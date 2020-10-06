import argparse
import torch
import yaml
import tqdm
import os
import numpy as np
from models.flows import F_inv_flow_new, F_inv_flow
from models.models import model_load
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, G_flows, _, _ = model_load(config, device, train=False)

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

    for key in F_flows:
        F_flows[key].eval()
    for key in G_flows:
        G_flows[key].eval()

    embs4g = config['prior_e_var'] * torch.randn(config['n_samples'], config['emb_dim']).to(device)

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

    samples = []
    for sample_index in tqdm.trange(config["n_samples"], desc="Sample"):
        z = config['prior_z_var'] * torch.randn(config['n_points'], 3).to(device).float()

        with torch.no_grad():
            targets = torch.LongTensor(config['n_points'], 1).fill_(sample_index)
            embeddings4g = embs4g[targets].view(-1, config['emb_dim'])
            z = F_inv_flow(z, embeddings4g, F_flows, config['n_flows_F'])
            z = z * std + mean
        samples.append(z.cpu())
    samples = torch.cat(samples, 0).view(-1, config['n_points'], 3)
    torch.save(samples, config['load_models_dir'] + 'sampling_samples.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
