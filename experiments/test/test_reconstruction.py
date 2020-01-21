import argparse

import numpy as np
import torch
import yaml

from data.datasets_pointflow import ShapeNet15kPointClouds
from models.architecture import Embeddings4Recon
from models.flows import F_inv_flow
from models.models import model_load
from utils.plotting_tools import plot_points


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    n_test_clouds, cloud_size, _ = test_cloud.all_points.shape

    F_flows, _, _, _, w = model_load(config, device, train=False)
    embs4recon = Embeddings4Recon(1, config["emb_dim"]).to(device)

    embs4recon.load_state_dict(torch.load(config["embs_dir"] + r"embs.pth"))

    data = (
        torch.tensor(test_cloud.all_points[config["id4recon"]]).float()
    ).to(device)

    for key in F_flows:
        F_flows[key].eval()
    embs4recon.eval()

    z = torch.randn(config["n_points"], 3).to(device).float()

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
        targets = torch.LongTensor(config["n_points"], 1).fill_(0)
        embeddings = embs4recon(targets).view(-1, config["emb_dim"])

        z = F_inv_flow(z, embeddings, F_flows, config["n_flows_F"])
        z = z * std + mean

    data = data * std + mean
    plot_points(
        z.cpu().numpy(),
        config,
        save_name="test_recon_" + str(config["id4recon"]),
        show=False,
    )
    plot_points(
        data.cpu().numpy(),
        config,
        save_name="test_ref_" + str(config["id4recon"]),
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
