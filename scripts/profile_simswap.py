#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import cProfile
import glob
import os
import pstats

import click
import yaml

import dot

# define globals
CONFIG = "./src/dot/simswap/configs/config.yaml"
SOURCE = "data/obama.jpg"
TARGET = "data/mona.jpg"
SAVE_FOLDER = "./profile_output/"
LIMIT = 1


@click.command()
@click.option("-c", "--config", default=CONFIG)
@click.option("--source", default=SOURCE)
@click.option("--target", default=TARGET)
@click.option("--save_folder", default=SAVE_FOLDER)
@click.option("--limit", type=int, default=LIMIT)
def main(
    config=CONFIG, source=SOURCE, target=TARGET, save_folder=SAVE_FOLDER, limit=LIMIT
):

    profiler = cProfile.Profile()

    with open(config) as f:
        config = yaml.safe_load(f)

    analysis_config = config["analysis"]["simswap"]
    _dot = dot.DOT(use_video=False, use_image=True, save_folder=save_folder)
    option = _dot.simswap(
        use_gpu=config["analysis"]["simswap"]["use_gpu"],
        gpen_type=config["analysis"]["simswap"]["gpen"],
        gpen_path=config["analysis"]["simswap"]["gpen_path"],
        use_mask=config["analysis"]["simswap"]["opt_use_mask"],
        crop_size=config["analysis"]["simswap"]["opt_crop_size"],
    )
    option.create_model(**analysis_config)
    profiler.enable()

    swappedMD, rejectedMD = _dot.generate(
        option,
        source=source,
        target=target,
        limit=limit,
        profiler=True,
        **analysis_config
    )
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("SimSwap_profiler.prof")


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
