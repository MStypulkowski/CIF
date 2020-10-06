import argparse
import torch
import yaml
import os
import tqdm
import numpy as np
from utils.metrics import MMD, pairwise_MMD
from models.models import model_load
from models.flows import F_inv_flow, G_flow
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
from models.pointnet import Encoder


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, G_flows, _, _ = model_load(config, device, train=False)

    pointnet = Encoder(
        load_pretrained=config["load_pretrained"],
        pretrained_path=config["pretrained_path"],
        zdim=config["emb_dim"],
    ).to(device)

    pointnet.load_state_dict(
            torch.load(os.path.join(config["load_models_dir"], "pointnet.pth"))
    )

    if config['use_random_dataloader']:
        tr_sample_size = 1
        te_sample_size = 1
    else:
        tr_sample_size = config['tr_sample_size']
        te_sample_size = config['te_sample_size']

    cloud_pointflow = ShapeNet15kPointClouds(
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
        cloud_pointflow = CIFDatasetDecorator(cloud_pointflow)

    if (
            config["resume_dataset_mean"] is not None
            and config["resume_dataset_std"] is not None
    ):
        mean = np.load(config["resume_dataset_mean"])
        std = np.load(config["resume_dataset_std"])
        cloud_pointflow.renormalize(mean, std)

        mean = torch.from_numpy(mean).to(device)
        std = torch.from_numpy(std).to(device)

    for key in F_flows:
        F_flows[key].eval()
    for key in G_flows:
        G_flows[key].eval()
    pointnet.eval()

    ref_samples = torch.from_numpy(cloud_pointflow.all_points[:, :2048, :]).float().to(device)
    ref_samples = ref_samples[:100]

    n_samples, cloud_size = ref_samples.shape[:2]
    print(n_samples, cloud_size)

    samples = []
    with torch.no_grad():
        for ref in tqdm.tqdm(ref_samples, desc='Reconstructions'):
            # ref = ref.to(device)
            w = pointnet(ref[None, :])
            e = G_flow(w, G_flows, config['n_flows_G'], config['emb_dim'])[0]
            e_mult = e.repeat(cloud_size, 1)
            
            z = torch.randn(ref.shape).to(device).float()
            
            x = F_inv_flow(z, e_mult, F_flows, config['n_flows_F'])
            x = x * std + mean
            samples.append(x)
    
    samples = (
                    torch.cat(samples, dim=0)
                        .reshape((n_samples, cloud_size, 3))
                        .to(device)
                )
    print(samples.shape, ref_samples.shape)

    ref_samples = ref_samples * std + mean

    torch.save(samples, config['load_models_dir'] + 'reconstructions_train.pth')
    torch.save(ref_samples, config['load_models_dir'] + 'references_train.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))