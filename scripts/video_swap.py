#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import click
import yaml

import dot

"""
Usage:
    python video_swap.py
    -c <path/to/config>
    -s <path/to/source/images>
    -t <path/to/target/videos>
    -o <path/to/output/folder>
    -d 5(Optional trim video)
    -l 5(Optional limit total swaps)
"""


@click.command()
@click.option("-c", "--config", default="./src/dot/simswap/configs/config.yaml")
@click.option("-s", "--source_image_path", required=True)
@click.option("-t", "--target_video_path", required=True)
@click.option("-o", "--output", required=True)
@click.option("-d", "--duration_per_video", required=False)
@click.option("-l", "--limit", type=int, required=False)
def main(
    config: str,
    source_image_path: str,
    target_video_path: str,
    output: str,
    duration_per_video: int,
    limit: int = None,
):
    """Given `source` and `target` folders, performs face-swap on each video with randomly chosen
    image found `source` path.
    Supported image formats: `["jpg", "png", "jpeg"]`
    Supported video formats: `["avi", "mp4", "mov", "MOV"]`

    Args:
        config (str): Path to configuration file.
        source_image_path (str): Path to source images
        target_video_path (str): Path to target videos
        output (str): Output folder path.
        duration_per_video (int): Trim duration of target video in seconds.
        limit (int, optional): Limit number of video-swaps. Defaults to None.
    """
    print(f"Loading config: {config}")
    with open(config) as f:
        config = yaml.safe_load(f)

    _dot = dot.DOT(use_video=True, use_image=False, save_folder=output)

    analysis_config = config["analysis"]["simswap"]
    option = _dot.simswap(
        use_gpu=analysis_config.get("use_gpu", False),
        use_mask=analysis_config.get("opt_use_mask", False),
        gpen_type=analysis_config.get("gpen", None),
        gpen_path=analysis_config.get("gpen_path", None),
        crop_size=analysis_config.get("opt_crop_size", 224),
    )
    _dot.generate(
        option=option,
        source=source_image_path,
        target=target_video_path,
        duration=duration_per_video,
        limit=limit,
        **analysis_config,
    )


if __name__ == "__main__":
    main()
