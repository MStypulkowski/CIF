import argparse

import numpy as np
import torch
import tqdm
import yaml

from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
from models.flows import F_inv_flow, G_flow
from models.models import model_load
from utils.plotting_tools import plot_points


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, G_flows, _, _, w = model_load(config, device, train=False)

    test_cloud = CIFDatasetDecorator(
        ShapeNet15kPointClouds(
            tr_sample_size=config["tr_sample_size"],
            te_sample_size=config["te_sample_size"],
            root_dir=config["root_dir"],
            normalize_per_shape=config["normalize_per_shape"],
            normalize_std_per_axis=config["normalize_std_per_axis"],
            split="test",
            scale=config["scale"],
            categories=config["categories"],
        )
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
    for key in G_flows:
        G_flows[key].eval()

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
    for sample_index in tqdm.trange(10):
        z = torch.randn(config["n_points"], 3).to(device).float()
        with torch.no_grad():
            targets = torch.tensor(
                (config["n_points"], 1), dtype=torch.long
            ).fill_(sample_index)
            embeddings = w[targets].view(-1, config["emb_dim"])

            e, _ = G_flow(
                embeddings, G_flows, config["n_flows_G"], config["emb_dim"]
            )
            z = F_inv_flow(z, e, F_flows, config["n_flows_F"])
            z = z * std + mean

        plot_points(
            z.cpu().numpy(),
            config,
            save_name="recon_" + str(sample_index),
            show=False,
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
