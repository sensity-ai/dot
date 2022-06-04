#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

from typing import Union

import click

from .dot import DOT


@click.command()
@click.option(
    "--swap_type",
    "swap_type",
    type=click.Choice(["avatarify", "faceswap_cv2", "simswap"], case_sensitive=False),
    required=True,
)
@click.option(
    "--source",
    "source",
    required=True,
    help="Images to swap with target",
)
@click.option(
    "--target",
    "target",
    required=True,
    help="Cam ID or target media",
)
@click.option(
    "--model_path",
    "model_path",
    default=None,
    help="Path to 68-point facial landmark detector for FaceSwap-cv2 or to the model's weights for the FOM",
)
@click.option(
    "--parsing_model_path",
    "parsing_model_path",
    default=None,
    help="Path to the parsing model",
)
@click.option(
    "--arcface_model_path",
    "arcface_model_path",
    default=None,
    help="Path to arcface model",
)
@click.option(
    "--checkpoints_dir",
    "checkpoints_dir",
    default=None,
    help="models are saved here",
)
@click.option(
    "--gpen_type",
    "gpen_type",
    default=None,
    type=click.Choice(["gpen_256", "gpen_512"]),
)
@click.option(
    "--gpen_path",
    "gpen_path",
    default="./saved_models/gpen",
    help="Path to gpen models.",
)
@click.option("--crop_size", "crop_size", type=int, default=224)
@click.option("--save_folder", "save_folder", type=str, default=None)
@click.option(
    "--show_fps",
    "show_fps",
    type=bool,
    default=False,
    is_flag=True,
    help="Pass flag to show fps value.",
)
@click.option(
    "--use_gpu",
    "use_gpu",
    type=bool,
    default=False,
    is_flag=True,
    help="Pass flag to use GPU else use CPU.",
)
@click.option(
    "--use_video",
    "use_video",
    type=bool,
    default=False,
    is_flag=True,
    help="Pass flag to use video-swap pipeline.",
)
@click.option(
    "--use_image",
    "use_image",
    type=bool,
    default=False,
    is_flag=True,
    help="Pass flag to use image-swap pipeline.",
)
@click.option("--limit", "limit", type=int, default=None)
def main(
    swap_type: str,
    source: str,
    target: Union[int, str],
    model_path: str = None,
    parsing_model_path: str = None,
    arcface_model_path: str = None,
    checkpoints_dir: str = None,
    gpen_type: str = None,
    gpen_path: str = "./saved_models/gpen",
    crop_size: int = 224,
    save_folder: str = None,
    show_fps: bool = False,
    use_gpu: bool = False,
    use_video: bool = False,
    use_image: bool = False,
    limit: int = None,
):
    """CLI entrypoint for dot."""
    # initialize dot
    _dot = DOT(use_video=use_video, use_image=use_image, save_folder=save_folder)

    # build dot
    option = _dot.build_option(
        swap_type=swap_type,
        use_gpu=use_gpu,
        gpen_type=gpen_type,
        gpen_path=gpen_path,
        crop_size=crop_size,
    )

    # run dot
    _dot.generate(
        option=option,
        source=source,
        target=target,
        show_fps=show_fps,
        model_path=model_path,
        limit=limit,
        parsing_model_path=parsing_model_path,
        arcface_model_path=arcface_model_path,
        checkpoints_dir=checkpoints_dir,
        opt_crop_size=crop_size,
    )


if __name__ == "__main__":
    main()
