"""Rendering is performed using:
https://github.com/kacperkan/mitsuba-flask-service """
import argparse
import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import tqdm
import yaml

from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
from models.flows import F_inv_flow, F_inv_flow_new
from models.models import model_load
from utils.visualize_points import (
    colormap,
    decode_image,
    standardize_bbox,
    xml_ball_segment,
    xml_head,
    xml_tail,
)


def visualize_single(
    points: np.ndarray,
    port: int,
    is_rotated: bool,
    colors: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, point_indices = standardize_bbox(
        points, 2048, return_point_indices=True
    )

    if not is_rotated:
        points = points[:, [2, 0, 1]]
        points[:, 0] *= -1
    else:
        points[:, 1] *= -1
        points = points[:, [1, 0, 2]]

    points[:, 2] += 0.0125
    xml_segments = [xml_head]
    create_new_colors = colors is None
    if create_new_colors:
        colors = []

    for i in range(points.shape[0]):
        if create_new_colors:
            color = colormap(
                points[i, 0] + 0.5,
                points[i, 1] + 0.5,
                points[i, 2] + 0.5 - 0.0125,
            )
            colors.append(color)
        else:
            color = colors[i]
        xml_segments.append(
            xml_ball_segment.format(
                points[i, 0], points[i, 1], points[i, 2], *color
            )
        )
    xml_segments.append(xml_tail)

    xml_content = str.join("", xml_segments)
    result = requests.post(f"http://localhost:{port}/render", data=xml_content)
    data = json.loads(result.content)
    an_img = decode_image(data)
    return an_img, np.array(colors), point_indices


def get_visualizations(
    points: np.ndarray,
    original_points: np.ndarray,
    port: int,
    is_rotated: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    orig_img, colors, point_indices = visualize_single(
        original_points, port, is_rotated, None
    )
    trans_img, _, _ = visualize_single(
        points[point_indices], port, is_rotated, colors
    )

    return orig_img, trans_img


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, G_flows, _, _, _ = model_load(config, device, train=False)

    if config["use_random_dataloader"]:
        tr_sample_size = 1
        te_sample_size = 1
    else:
        tr_sample_size = config["tr_sample_size"]
        te_sample_size = config["te_sample_size"]

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

    if config["use_random_dataloader"]:
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

    embs4g = config["prior_e_var"] * torch.randn(
        config["n_samples"], config["emb_dim"]
    ).to(device)

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
    save_vis_dir = os.path.join(
        config["renders_save_dir"], config["categories"][0], "point-colouring"
    )
    os.makedirs(save_vis_dir, exist_ok=True)

    for sample_index in tqdm.trange(config["n_samples"], desc="Sample"):
        z = (
            config["prior_z_var"]
            * torch.randn(config["n_points"], 3).to(device).float()
        )
        orig_points = z.clone()

        with torch.no_grad():
            targets = torch.empty(
                (config["n_points"], 1), dtype=torch.long
            ).fill_(sample_index)
            embeddings4g = embs4g[targets].view(-1, config["emb_dim"])

            if config["use_new_f"]:
                z = F_inv_flow_new(
                    z, embeddings4g, F_flows, config["n_flows_F"]
                )
            else:
                z = F_inv_flow(z, embeddings4g, F_flows, config["n_flows_F"])
            z = z * std + mean
            orig_points = orig_points * std + mean
        point_cloud = z.view((config["n_points"], 3))
        orig_point_cloud = orig_points.view((config["n_points"], 3))
        orig_vis, trans_vis = get_visualizations(
            point_cloud.detach().cpu().numpy(),
            orig_point_cloud.detach().cpu().numpy(),
            config["renderer_port"],
            is_rotated=config["are_shapes_rotated_by_default"],
        )

        cv2.imwrite(
            os.path.join(save_vis_dir, f"{sample_index}_trans.png"), trans_vis
        )
        cv2.imwrite(
            os.path.join(save_vis_dir, f"{sample_index}_orig.png"), orig_vis
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
