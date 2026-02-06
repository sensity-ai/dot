#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

# Suppress diffusers CUDA autocast warning on macOS / non-CUDA systems.
# The library uses device_type="cuda"; we use MPS or CPU when CUDA is unavailable.
import warnings
warnings.filterwarnings(
    "ignore",
    message="CUDA is not available or torch_xla is imported. Disabling autocast.",
    category=UserWarning,
    module="diffusers",
)

import traceback
from typing import Union

import click
import yaml

from .commons.paths import resolve_app_path
from .dot import DOT, AVAILABLE_SWAP_TYPES
from .filters import AVAILABLE_FILTERS


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
    filter_type: str = None,
    # Diffusion model options
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    use_lcm: bool = False,
    controlnet_conditioning_scale: float = 0.8,
    ip_adapter_scale: float = 0.8,
    use_sdxl: bool = False,
    use_face_restore: bool = True,
    face_restore_model: str = "codeformer",
    use_virtual_cam: bool = False,
    virtual_cam_fps: int = 20,
    cam_width: int = 1280,
    cam_height: int = 720,
    live_settings: dict = None,
    gui_update_callback=None,
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
        filter_type (str, optional): Camera/film filter to apply. Defaults to None.
    """
    try:
        # Resolve relative paths when running as frozen app (PyInstaller)
        gpen_path = resolve_app_path(gpen_path)
        if model_path:
            model_path = resolve_app_path(model_path)
        if parsing_model_path:
            parsing_model_path = resolve_app_path(parsing_model_path)
        if arcface_model_path:
            arcface_model_path = resolve_app_path(arcface_model_path)
        if checkpoints_dir:
            checkpoints_dir = resolve_app_path(checkpoints_dir)

        # initialize dot
        _dot = DOT(use_video=use_video, use_image=use_image, save_folder=save_folder)

        # build dot
        option = _dot.build_option(
            swap_type=swap_type,
            use_gpu=use_gpu,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            crop_size=crop_size,
            filter_type=filter_type,
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
            use_virtual_cam=use_virtual_cam,
            virtual_cam_fps=virtual_cam_fps,
            cam_width=cam_width,
            cam_height=cam_height,
            live_settings=live_settings,
            gui_update_callback=gui_update_callback,
        )
    except:  # noqa
        print(traceback.format_exc())


@click.command()
@click.option(
    "--gui",
    "gui",
    is_flag=True,
    default=False,
    help="Launch the graphical interface (tint and other settings).",
)
@click.option(
    "--list-cameras",
    "list_cameras",
    is_flag=True,
    default=False,
    help="List available camera devices (indices and resolution) then exit.",
)
@click.option(
    "--swap_type",
    "swap_type",
    type=click.Choice(AVAILABLE_SWAP_TYPES, case_sensitive=False),
    default=None,
    help=f"Swap engine to use. Available: {', '.join(AVAILABLE_SWAP_TYPES)}",
)
@click.option(
    "--source",
    "source",
    default=None,
    help="Images to swap with target",
)
@click.option(
    "--target",
    "target",
    default=None,
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
    "--filter",
    "filter_type",
    type=click.Choice(AVAILABLE_FILTERS),
    default=None,
    help="Apply camera/film filter to make output more realistic.",
)
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
@click.option(
    "--virtual_cam",
    "use_virtual_cam",
    is_flag=True,
    default=False,
    help="Send output to a virtual camera device (FaceTime/Zoom). Requires: pip install pyvirtualcam; on macOS also install OBS.",
)
@click.option(
    "--virtual_cam_fps",
    "virtual_cam_fps",
    type=int,
    default=20,
    help="Virtual camera FPS when --virtual_cam is used (default 20).",
)
@click.option(
    "--cam_width",
    "cam_width",
    type=int,
    default=1280,
    help="Camera capture width (default 1280).",
)
@click.option(
    "--cam_height",
    "cam_height",
    type=int,
    default=720,
    help="Camera capture height (default 720).",
)
def main(
    gui: bool,
    list_cameras: bool,
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
    filter_type: str = None,
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
    use_virtual_cam: bool = False,
    virtual_cam_fps: int = 20,
    cam_width: int = 1920,
    cam_height: int = 1080,
):
    """CLI entrypoint for dot."""
    if gui:
        from dot.ui.ui import App
        app = App()
        app.mainloop()
        return

    if list_cameras:
        from dot.commons.video.videocaptureasync import list_cameras as list_cameras_fn
        cameras = list_cameras_fn(max_index=10)
        if not cameras:
            print("No cameras found (probed indices 0–9).")
        else:
            print("Camera devices (use --target <index> for camera mode):")
            for idx, res in cameras:
                print(f"  {idx}: {res or 'unknown resolution'}")
        return

    # load config, if provided
    config = {}
    if config_file is not None:
        config_path = resolve_app_path(config_file)
        with open(config_path) as f:
            config = yaml.safe_load(f)

    swap_type = config.get("swap_type", swap_type)
    source = config.get("source", source)
    target = config.get("target", target)
    if swap_type is None or source is None or target is None:
        raise click.UsageError(
            "Missing required options. Use --swap_type, --source, --target (or -c config.yaml). Use --gui to launch the graphical interface."
        )

    # run dot
    run(
        swap_type=swap_type,
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
        filter_type=config.get("filter_type", filter_type),
        # Diffusion options
        num_inference_steps=config.get("num_inference_steps", num_inference_steps),
        guidance_scale=config.get("guidance_scale", guidance_scale),
        use_lcm=config.get("use_lcm", use_lcm),
        controlnet_conditioning_scale=config.get("controlnet_conditioning_scale", controlnet_conditioning_scale),
        ip_adapter_scale=config.get("ip_adapter_scale", ip_adapter_scale),
        use_sdxl=config.get("use_sdxl", use_sdxl),
        use_face_restore=config.get("use_face_restore", use_face_restore),
        face_restore_model=config.get("face_restore_model", face_restore_model),
        use_virtual_cam=config.get("use_virtual_cam", use_virtual_cam),
        virtual_cam_fps=config.get("virtual_cam_fps", virtual_cam_fps),
        cam_width=config.get("cam_width", cam_width),
        cam_height=config.get("cam_height", cam_height),
    )


if __name__ == "__main__":
    main()
