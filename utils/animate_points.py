import argparse
import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import tqdm

from utils.visualize_points import process_single


def visualize(
    path: str,
    out_image_path: str,
    port: int,
    num_frames: int,
    indices: Optional[List[int]],
):
    points = torch.load(path, map_location="cpu").detach().cpu().numpy()

    if indices is None:
        indices = list(range(points))
    start_position = (3, 3, 3)
    radius = np.linalg.norm(start_position)
    start_theta = np.arctan(start_position[1] / start_position[0])
    phi = np.arccos(start_position[-1] / radius)

    out_path = Path(out_image_path)
    if not out_path.exists():
        out_path.mkdir(parents=True)

    pbar = tqdm.tqdm(total=len(indices))
    for i in indices:
        pts = points[i]
        out_folder = out_path / f"{i}"
        out_folder.mkdir(exist_ok=True, parents=True)

        for j, offset in enumerate(
            tqdm.tqdm(np.linspace(0, 2 * math.pi, num=num_frames + 1))
        ):
            theta = start_theta + offset
            camera_params = (
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi),
            )
            frame = process_single(
                pts, port, False, camera_params=camera_params
            )

            cv2.imwrite(
                (out_folder / f"{j}.png").as_posix(),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
        pbar.update(1)
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", help="Path to file from PointFlow dataset"
    )

    parser.add_argument("out", help="Path to image file of rendered points")
    parser.add_argument(
        "--frames",
        default=120,
        type=int,
        help="Number of frames around the object",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port of the rendering service"
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        help="Indices from the tensor to render",
    )
    args = parser.parse_args()

    visualize(args.file_path, args.out, args.port, args.frames, args.indices)


if __name__ == "__main__":
    main()
