#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import traceback
from typing import Union

import click
import yaml

from .dot import DOT, AVAILABLE_SWAP_TYPES


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
    # Diffusion model options
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    use_lcm: bool = False,
    controlnet_conditioning_scale: float = 0.8,
    ip_adapter_scale: float = 0.8,
    use_sdxl: bool = False,
    use_face_restore: bool = True,
    face_restore_model: str = "codeformer",
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
        num_inference_steps (int, optional): Diffusion inference steps. Defaults to 20.
        guidance_scale (float, optional): Classifier-free guidance scale. Defaults to 7.5.
        use_lcm (bool, optional): Use LCM scheduler for faster inference. Defaults to False.
        controlnet_conditioning_scale (float, optional): ControlNet scale for InstantID. Defaults to 0.8.
        ip_adapter_scale (float, optional): IP-Adapter scale. Defaults to 0.8.
        use_sdxl (bool, optional): Use SDXL model for IP-Adapter. Defaults to False.
        use_face_restore (bool, optional): Use face restoration for ReActor. Defaults to True.
        face_restore_model (str, optional): Face restore model for ReActor. Defaults to "codeformer".
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
            # Diffusion options
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_lcm=use_lcm,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            use_sdxl=use_sdxl,
            use_face_restore=use_face_restore,
            face_restore_model=face_restore_model,
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
    type=click.Choice(AVAILABLE_SWAP_TYPES, case_sensitive=False),
    help=f"Swap engine to use. Available: {', '.join(AVAILABLE_SWAP_TYPES)}",
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
# Diffusion model options
@click.option(
    "--num_inference_steps",
    "num_inference_steps",
    type=int,
    default=20,
    help="Number of diffusion inference steps (for diffusion models).",
)
@click.option(
    "--guidance_scale",
    "guidance_scale",
    type=float,
    default=7.5,
    help="Classifier-free guidance scale (for diffusion models).",
)
@click.option(
    "--use_lcm",
    "use_lcm",
    type=bool,
    default=False,
    is_flag=True,
    help="Use LCM scheduler for faster inference (for diffusion models).",
)
@click.option(
    "--controlnet_scale",
    "controlnet_conditioning_scale",
    type=float,
    default=0.8,
    help="ControlNet conditioning scale (for InstantID).",
)
@click.option(
    "--ip_adapter_scale",
    "ip_adapter_scale",
    type=float,
    default=0.8,
    help="IP-Adapter scale (for InstantID and IP-Adapter Face).",
)
@click.option(
    "--use_sdxl",
    "use_sdxl",
    type=bool,
    default=False,
    is_flag=True,
    help="Use SDXL model (for IP-Adapter Face).",
)
@click.option(
    "--use_face_restore",
    "use_face_restore",
    type=bool,
    default=True,
    is_flag=True,
    help="Use face restoration (for ReActor).",
)
@click.option(
    "--face_restore_model",
    "face_restore_model",
    type=click.Choice(["codeformer", "gfpgan"]),
    default="codeformer",
    help="Face restoration model (for ReActor).",
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
    # Diffusion model options
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    use_lcm: bool = False,
    controlnet_conditioning_scale: float = 0.8,
    ip_adapter_scale: float = 0.8,
    use_sdxl: bool = False,
    use_face_restore: bool = True,
    face_restore_model: str = "codeformer",
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
        # Diffusion options
        num_inference_steps=config.get("num_inference_steps", num_inference_steps),
        guidance_scale=config.get("guidance_scale", guidance_scale),
        use_lcm=config.get("use_lcm", use_lcm),
        controlnet_conditioning_scale=config.get("controlnet_conditioning_scale", controlnet_conditioning_scale),
        ip_adapter_scale=config.get("ip_adapter_scale", ip_adapter_scale),
        use_sdxl=config.get("use_sdxl", use_sdxl),
        use_face_restore=config.get("use_face_restore", use_face_restore),
        face_restore_model=config.get("face_restore_model", face_restore_model),
    )


if __name__ == "__main__":
    main()
