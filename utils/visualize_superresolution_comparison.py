import argparse
import itertools
import os
import shutil
import subprocess
import typing as t
import zipfile
from pathlib import Path

import click
import cv2
import numpy as np
import tqdm

from utils.visualize_points import process_single


@click.group()
def cli():
    pass


@cli.command("unpack")
@click.option(
    "--path", type=Path, help="Path to directory with zips", required=True
)
def unpack(path: Path):
    files = path.glob("*.zip")
    f: Path
    for f in files:
        subprocess.call(["unzip", "-d", f.with_suffix(""), f.absolute()])
        os.remove(f.absolute())
    print("Folder unpacked in {}".format(path.absolute()))


@cli.command("visualize")
@click.option(
    "--in_dir", type=Path, help="Path to folders with data to visualize"
)
@click.option("--out_dir", type=Path, help="Path to folders with output data")
@click.option(
    "--port", type=int, default=8000, help="Port value of the renderer"
)
def visualize(in_dir: Path, out_dir: Path, port: int):
    dirs = list(in_dir.rglob("*.npy"))
    if out_dir.exists():
        shutil.rmtree(out_dir.as_posix())
    with tqdm.tqdm(dirs) as bar:
        f: Path
        for f in bar:
            bar.set_description_str(f"Processing: {f.parent.name}/{f.name}")
            points = np.load(f.as_posix())[0]
            img = process_single(
                points,
                port=port,
                is_rotated=False,
                limit_points=-1,
                point_size=0.015,
            )

            out_file = out_dir / f.parent.name / f.with_suffix(".png").name
            out_file.parent.mkdir(exist_ok=True, parents=True)

            cv2.imwrite(
                out_file.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )


@cli.command("clean")
@click.option("--out_dir", type=Path, help="Ouput dire")
def to_separate_folde(out_dir: Path):
    folders_to_filter = ["4096", "16384", "65536"]
    for to_fix_dir in folders_to_filter:
        temp_dir = out_dir / to_fix_dir
        images = temp_dir.rglob("*.png")
        image: Path
        for image in images:
            proper_name = "pointflow_{name}_recon".format(
                name=image.name.split("_")[0]
            )

            class_folder = out_dir / proper_name
            output_name = image.with_suffix("").name.replace(
                "ref", "recon"
            ) + "_{}".format(image.parent.name)

            class_folder.mkdir(exist_ok=True, parents=True)
            shutil.move(
                image.as_posix(),
                (class_folder / output_name).with_suffix(".png").as_posix(),
            )
        shutil.rmtree(temp_dir.as_posix())


if __name__ == "__main__":
    cli()
