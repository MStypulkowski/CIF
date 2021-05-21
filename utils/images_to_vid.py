import argparse
import os
import shlex
import subprocess
from pathlib import Path


def generate(in_folder: str):
    input_path = os.path.join(in_folder, "%d.png")
    out_path = Path(in_folder).with_suffix(".mp4").as_posix()
    command = "ffmpeg -framerate 24 -i {} -c:v libx264 -pix_fmt yuv420p -crf 18 -y {}".format(
        input_path, out_path
    )
    subprocess.call(shlex.split(command))


def main():
    parser = argparse.ArgumentParser(
        description="A script to generation movies from image sequences"
    )
    parser.add_argument("in_folder")
    args = parser.parse_args()
    generate(args.in_folder)


if __name__ == "__main__":
    main()
