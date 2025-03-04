#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import glob
import json
import os

import click
import yaml

import dot

"""
Usage:
    python image_swap.py
    -c <path/to/config>
    -s <path/to/source/images>
    -t <path/to/target/images>
    -o <path/to/output/folder>
    -l 5(Optional limit total swaps)
"""


@click.command()
@click.option("-c", "--config", default="./src/dot/simswap/configs/config.yaml")
@click.option("-s", "--source", required=True)
@click.option("-t", "--target", required=True)
@click.option("-o", "--save_folder", required=False)
@click.option("-l", "--limit", type=int, required=False)
def main(
    config: str, source: str, target: str, save_folder: str, limit: int = False
) -> None:
    """Performs face-swap given a `source/target` image(s). Saves JSON file of (un)successful swaps.
    Args:
        config (str): Path to DOT configuration yaml file.
        source (str): Path to source images folder or certain image file.
        target (str): Path to target images folder or certain image file.
        save_folder (str): Output folder to store face-swaps and metadata file.
        limit (int, optional): Number of desired face-swaps. If not specified,
        all possible combinations of source/target pairs will be processed. Defaults to False.
    """

    print(f"Loading config: {config}")
    with open(config) as f:
        config = yaml.safe_load(f)

    _dot = dot.DOT(use_video=False, use_image=True, save_folder=save_folder)

    analysis_config = config["analysis"]["simswap"]
    option = _dot.simswap(
        use_gpu=analysis_config.get("use_gpu", False),
        use_mask=analysis_config.get("opt_use_mask", False),
        gpen_type=analysis_config.get("gpen", None),
        gpen_path=analysis_config.get("gpen_path", None),
        crop_size=analysis_config.get("opt_crop_size", 224),
    )

    swappedMD, rejectedMD = _dot.generate(
        option, source=source, target=target, limit=limit, **analysis_config
    )

    # save metadata file
    if swappedMD:
        with open(os.path.join(save_folder, "metadata.json"), "a") as fp:
            json.dump(swappedMD, fp, indent=4)

    # save rejected face-swaps
    if rejectedMD:
        with open(os.path.join(save_folder, "rejected.json"), "a") as fp:
            json.dump(rejectedMD, fp, indent=4)


def find_images_from_path(path):
    if os.path.isfile(path):
        return [path]

    try:
        return int(path)
    except ValueError:
        # supported extensions
        ext = ["png", "jpg", "jpeg"]
        files = []
        [files.extend(glob.glob(path + "**/*." + e, recursive=True)) for e in ext]

        return files


if __name__ == "__main__":
    main()
