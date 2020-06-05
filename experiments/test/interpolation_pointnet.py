import argparse

import numpy as np
import torch
import tqdm
import yaml

from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
from models.flows import F_inv_flow, G_flow
from models.models import model_load


# from utils.plotting_tools import get_rotation_matrix
# from utils.plotting_tools import plot_points


def main(config: argparse.Namespace):
    def get_embs(ids):
        embs = []
        for i in ids:
            ref = ref_samples[i]
            ref = ref.to(device)
            with torch.no_grad():
                w = pointnet(ref[None, :])
                e = G_flow(w, G_flows, config["n_flows_G"], config["emb_dim"])[
                    0
                ]
                embs.append(e)
        return embs

    def reconstruct_from_emb(e, cloud_size=2048):
        e_mult = e.repeat(cloud_size, 1)

        z = torch.randn(cloud_size, 3).to(device).float()

        with torch.no_grad():
            x = F_inv_flow(z, e_mult, F_flows, config["n_flows_F"])
        x = x * std + mean
        return x[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, G_flows, pointnet, _, _ = model_load(config, device, train=False)

    for key in F_flows:
        F_flows[key].eval()
    for key in G_flows:
        G_flows[key].eval()
    pointnet.eval()

    if config["use_random_dataloader"]:
        tr_sample_size = 1
        te_sample_size = 1
    else:
        tr_sample_size = config["tr_sample_size"]
        te_sample_size = config["te_sample_size"]

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

    if config["use_random_dataloader"]:
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
    else:
        mean = 0
        std = 1

    ref_samples = torch.from_numpy(
        cloud_pointflow.all_points[:, :2048, :]
    ).float()
    cloud_size = config["n_points"]

    embs = get_embs(config["interpolation_ids"])
    start_embs = embs
    stop_embs = embs[1:] + [embs[0]]

    inter_samples = []
    for start, stop in tqdm.tqdm(
        zip(start_embs, stop_embs),
        total=len(start_embs),
        desc="Interpolations",
    ):
        n_midsamples = config["n_midsamples"]
        step_vector = (stop - start) / (n_midsamples + 1)

        for i in range(n_midsamples + 1):
            emb = start + i * step_vector
            midsample = reconstruct_from_emb(emb)
            inter_samples.append(midsample.cpu())

    inter_samples = (
        torch.cat(inter_samples, dim=0).reshape((-1, cloud_size, 3))
        # .to(device)
    )

    torch.save(
        inter_samples, config["load_models_dir"] + "interpolation_samples.pth"
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
