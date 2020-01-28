import argparse
import torch
import yaml
import tqdm
import numpy as np
from utils.metrics import MMD, coverage
from models.models import model_load
from models.flows import F_inv_flow_new, F_inv_flow
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, _, _, _, w = model_load(config, device, train=False)

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

    n_test_clouds = len(test_cloud)
    cloud_size = test_cloud[0]["test_points"].shape[0]
    n_samples = n_test_clouds

    samples = []
    embs4g = torch.randn(n_samples, config['emb_dim']).to(device)

    for sample_index in tqdm.trange(n_samples, desc="Sample"):
        z = torch.randn(cloud_size, 3).to(device).float()
        with torch.no_grad():
            targets = torch.LongTensor(cloud_size, 1).fill_(sample_index)
            embeddings4g = embs4g[targets].view(-1, config['emb_dim'])

            if config['use_new_f']:
                z = F_inv_flow_new(z, embeddings4g, F_flows, config['n_flows_F'])
            else:
                z = F_inv_flow(z, embeddings4g, F_flows, config['n_flows_F'])
            z = z * std + mean
            samples.append(z)

    samples = (
        torch.cat(samples, dim=0)
            .reshape((n_samples, cloud_size, 3))
            .to(device)
    )

    ref_samples = torch.from_numpy(test_cloud.all_points).float().to(device)
    ref_samples = ref_samples * std + mean

    if config["use_EMD"]:
        print(
            "Coverage (EMD): {:.4f}%".format(
                coverage(samples, ref_samples) * 100
            )
        )
        print("MMD (EMD): {:.4f}".format(MMD(samples, ref_samples).item()))

    else:
        print(
            "Coverage (CD): {:.4f}%".format(
                coverage(samples, ref_samples, use_EMD=False) * 100
            )
        )
        print(
            "MMD (CD): {:.4f}".format(
                MMD(samples, ref_samples, use_EMD=False).item()
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
