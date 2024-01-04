#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import traceback
from typing import Union

import click
import yaml

from .dot import DOT


def run(
    swap_type: str,
    source: str,
    target: Union[int, str],
    model_path: str = None,
    parsing_model_path: str = None,
    arcface_model_path: str = None,
    checkpoints_dir: str = None,
    gpen_type: str = None,
    gpen_path: str = "saved_models/gpen",
    crop_size: int = 224,
    head_pose: bool = False,
    save_folder: str = None,
    show_fps: bool = False,
    use_gpu: bool = False,
    use_video: bool = False,
    use_image: bool = False,
    limit: int = None,
):
    """Builds a DOT object and runs it.

    Args:
        swap_type (str): The type of swap to run.
        source (str): The source image or video.
        target (Union[int, str]): The target image or video.
        model_path (str, optional): The path to the model's weights. Defaults to None.
        parsing_model_path (str, optional): The path to the parsing model. Defaults to None.
        arcface_model_path (str, optional): The path to the arcface model. Defaults to None.
        checkpoints_dir (str, optional): The path to the checkpoints directory. Defaults to None.
        gpen_type (str, optional): The type of gpen model to use. Defaults to None.
        gpen_path (str, optional): The path to the gpen models. Defaults to "saved_models/gpen".
        crop_size (int, optional): The size to crop the images to. Defaults to 224.
        save_folder (str, optional): The path to the save folder. Defaults to None.
        show_fps (bool, optional): Pass flag to show fps value. Defaults to False.
        use_gpu (bool, optional): Pass flag to use GPU else use CPU. Defaults to False.
        use_video (bool, optional): Pass flag to use video-swap pipeline. Defaults to False.
        use_image (bool, optional): Pass flag to use image-swap pipeline. Defaults to False.
        limit (int, optional): The number of frames to process. Defaults to None.
    """
    try:
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
            head_pose=head_pose,
        )
    except:  # noqa
        print(traceback.format_exc())


@click.command()
@click.option(
    "--swap_type",
    "swap_type",
    type=click.Choice(["fomm", "faceswap_cv2", "simswap"], case_sensitive=False),
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
    default="saved_models/gpen",
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
@click.option(
    "-c",
    "--config",
    "config_file",
    help="Configuration file. Overrides duplicate options passed.",
    required=False,
    default=None,
)
def main(
    swap_type: str,
    source: str,
    target: Union[int, str],
    model_path: str = None,
    parsing_model_path: str = None,
    arcface_model_path: str = None,
    checkpoints_dir: str = None,
    gpen_type: str = None,
    gpen_path: str = "saved_models/gpen",
    crop_size: int = 224,
    save_folder: str = None,
    show_fps: bool = False,
    use_gpu: bool = False,
    use_video: bool = False,
    use_image: bool = False,
    limit: int = None,
    config_file: str = None,
):
    """CLI entrypoint for dot."""
    # load config, if provided
    config = {}
    if config_file is not None:
        with open(config_file) as f:
            config = yaml.safe_load(f)

    # run dot
    run(
        swap_type=config.get("swap_type", swap_type),
        source=source,
        target=target,
        model_path=config.get("model_path", model_path),
        parsing_model_path=config.get("parsing_model_path", parsing_model_path),
        arcface_model_path=config.get("arcface_model_path", arcface_model_path),
        checkpoints_dir=config.get("checkpoints_dir", checkpoints_dir),
        gpen_type=config.get("gpen_type", gpen_type),
        gpen_path=config.get("gpen_path", gpen_path),
        crop_size=config.get("crop_size", crop_size),
        head_pose=config.get("head_pose", False),
        save_folder=save_folder,
        show_fps=show_fps,
        use_gpu=use_gpu,
        use_video=use_video,
        use_image=use_image,
        limit=limit,
    )


if __name__ == "__main__":
    main()
