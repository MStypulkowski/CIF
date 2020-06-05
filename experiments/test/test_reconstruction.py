import argparse
import torch
import yaml
import numpy as np
from models.architecture import Embeddings4Recon
from models.models import model_load
from utils.plotting_tools import plot_points
from models.flows import F_inv_flow_new, F_inv_flow
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['use_random_dataloader']:
        tr_sample_size = 1
        te_sample_size = 1
    else:
        tr_sample_size = config['tr_sample_size']
        te_sample_size = config['te_sample_size']

    test_cloud = ShapeNet15kPointClouds(
        tr_sample_size=tr_sample_size,
        te_sample_size=te_sample_size,
        root_dir=config["root_dir"],
        root_embs_dir=config["root_embs_dir"],
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
        split="train",
        scale=config["scale"],
        categories=config["categories"],
        random_subsample=True,
    )

    if config['use_random_dataloader']:
        test_cloud = CIFDatasetDecorator(test_cloud)

    if (
            config["resume_dataset_mean"] is not None
            and config["resume_dataset_std"] is not None
    ):
        mean = np.load(config["resume_dataset_mean"])
        std = np.load(config["resume_dataset_std"])
        test_cloud.renormalize(mean, std)

    n_test_clouds, cloud_size, _ = test_cloud[0]["test_points"].shape

    F_flows, _, _, _ = model_load(config, device, train=False)
    embs4recon = Embeddings4Recon(1, config['emb_dim']).to(device)

    embs4recon.load_state_dict(torch.load(config['embs_dir'] + r'embs.pth'))

    data = (
        torch.tensor(test_cloud[config["id4recon"]]["test_points"]).float()
    ).to(device)

    for key in F_flows:
        F_flows[key].eval()
    embs4recon.eval()

    z = torch.randn(config['n_points'], 3).to(device).float()

    mean = (
        torch.from_numpy(test_cloud.all_points_mean)
            .float()
            .to(device)
            .squeeze(dim=0)
    )
    std = (
        torch.from_numpy(test_cloud.all_points_std)
            .float()
            .to(device)
            .squeeze(dim=0)
    )

    with torch.no_grad():
        targets = torch.LongTensor(config['n_points'], 1).fill_(0)
        embeddings = embs4recon(targets).view(-1, config['emb_dim'])

        if config['use_new_f']:
            z = F_inv_flow_new(z, embeddings, F_flows, config['n_flows_F'])
        else:
            z = F_inv_flow(z, embeddings, F_flows, config['n_flows_F'])
        z = z * std + mean
    data = data * std + mean
    plot_points(z.cpu().numpy(), config, save_name='test_recon_' + str(config['id4recon']), show=False)
    plot_points(data.cpu().numpy(), config, save_name='test_ref_' + str(config['id4recon']), show=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
