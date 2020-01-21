# Note that in the current state all points are used
# For the PointFlow train/test split it is 15k points
import argparse

import numpy as np
import torch
import tqdm
import yaml

from data.datasets_pointflow import ShapeNet15kPointClouds
from models.flows import F_inv_flow
from models.models import model_load
from utils.metrics import MMD, coverage


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, _, _, _, w = model_load(config, device, train=False)

    test_cloud = ShapeNet15kPointClouds(
        tr_sample_size=config["tr_sample_size"],
        te_sample_size=config["te_sample_size"],
        root_dir=config["root_dir"],
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
        split="test",
        scale=config["scale"],
        categories=config["categories"],
    )

    if (
        config["resume_dataset_mean"] is not None
        and config["resume_dataset_std"] is not None
    ):
        mean = np.load(config["resume_dataset_mean"])
        std = np.load(config["resume_dataset_std"])
        test_cloud.renormalize(mean, std)

    for key in F_flows:
        F_flows[key].eval()

    n_test_clouds, cloud_size, _ = test_cloud[0]["test_points"].shape
    n_samples = 3 * n_test_clouds

    samples = []
    embs4g = torch.randn(n_samples, config["emb_dim"]).to(device)

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

    for sample_index in tqdm.trange(n_samples, desc="Sample"):
        z = torch.randn(cloud_size, 3).to(device).float()
        with torch.no_grad():
            targets = torch.empty((cloud_size, 1), dtype=torch.long).fill_(
                sample_index
            )
            embeddings4g = embs4g[targets].view(-1, config["emb_dim"])

            z = F_inv_flow(z, embeddings4g, F_flows, config["n_flows_F"])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
