#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import json
import os

import click
import numpy as np
import pandas as pd
import yaml

import dot

"""
Usage:
    python metadata_swap.py \
    --config <path_to_config/config.yaml> \
    --local_root_path <path_to_root_directory> \
    --metadata <path_to_metadata_file> \
    --set <train_or_test_dataset> \
    --save_folder <path_to_output_folder> \
    --limit 100
"""

# common face identity features
face_identity_features = {
    1: "ArchedEyebrows",
    2: "Attractive",
    3: "BagsUnderEyes",
    6: "BigLips",
    7: "BigNose",
    12: "BushyEyebrows",
    16: "Goatee",
    18: "HeavyMakeup",
    19: "HighCheekbones",
    22: "Mustache",
    23: "NarrowEyes",
    24: "NoBeard",
    27: "PointyNose",
}


@click.command()
@click.option("-c", "--config", default="./src/dot/simswap/configs/config.yaml")
@click.option("--local_root_path", required=True)
@click.option("--metadata", required=True)
@click.option("--set", required=True)
@click.option("-o", "--save_folder", required=False)
@click.option("--limit", type=int, required=False)
def main(
    config: str,
    local_root_path: str,
    metadata: str,
    set: str,
    save_folder: str,
    limit: bool = None,
) -> None:
    """Script is tailored to dictionary format as shown below. `key` is the relative path to image,
    `value` is a list of total 44 attributes.
    [0:40] `Face attributes`: 50'ClockShadow, ArchedEyebrows, Attractive, BagsUnderEyes, Bald,Bangs,BigLips,
    BigNose, BlackHair, BlondHair, Blurry, BrownHair, BushyEyebrows, Chubby, DoubleChin ,Eyeglasses,Goatee,
    GrayHair, HeavyMakeup, HighCheekbones, Male, MouthSlightlyOpen, Mustache, NarrowEyes, NoBeard, OvalFace,
    PaleSkin, PointyNose, RecedingHairline, RosyCheeks, Sideburns, Smiling, StraightHair, WavyHair, WearingEarrings,
    WearingHat, WearingLipstick, WearingNecklace, WearingNecktie, Young.
    [41] `Spoof type`: Live, Photo, Poster, A4, Face Mask, Upper Body Mask, Region Mask, PC, Pa, Phone, 3D Mask.
    [42] `Illumination`: Live, Normal, Strong, Back, Dark.
    [43] `Live/Spoof(binary)`: Live, Spoof.
    {
        "rel_path/img1.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2,2,1],
        "rel_path/img2.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,2,1]
        ....
    }
    It constructs a pd.DataFrame from `metadata` and filters rows where examples are under-aged(young==0).
    Face-swaps are performed randomly based on gender. The `result-swap` image shares common attributes with
    the `source` image which are defined in `face_identity_features` dict.
    Spoof-type of swapped image is defined at index 40 of attributes list and set to 11.
    Args:
        config (str): Path to DOT configuration yaml file.
        local_root_path (str): Root path of dataset.
        metadata (str): JSON metadata file path of dataset.
        set (str): Defines train/test dataset.
        save_folder (str): Output folder to store face-swaps and metadata file.
        limit (int, optional): Number of desired face-swaps. If not specified, will be set equal to DataFrame size.
        Defaults to False.
    """
    if limit and limit < 4:
        print("Error: limit should be >= 4")
        return

    output_data_folder = os.path.join(save_folder + f"Data/{set}/swap/")
    df = pd.read_json(metadata, orient="index")

    mapping = {
        df.columns[20]: "gender",
        df.columns[26]: "pale_skin",
        df.columns[39]: "young",
    }
    df = df.rename(columns=mapping)
    df.head()
    # keep only live images
    df = df.loc[df.index.str.contains("live")]
    # keep only adult images
    df = df.loc[df["young"] == 0]
    if not limit:
        limit = df.shape[0]
        print(f"Limit is set to: {limit}")

    filters = ["gender==1", "gender==0"]
    swaps = []
    for filter in filters:
        # get n random rows based on condition ==1(male)
        filtered = df.query(filter).sample(n=round(limit / len(filters)), replace=True)
        # shuffle again, keep only indices and convert to list
        filtered = filtered.sample(frac=1).index.tolist()
        # append local_root_path
        filtered = [os.path.join(local_root_path, p) for p in filtered]
        # split into two lists roughly equal size
        mid_index = round(len(filtered) / 2)
        src = filtered[0:mid_index]
        tar = filtered[mid_index:]
        swaps.append((src, tar))

    print(f"Loading config: {config}")
    with open(config) as f:
        config = yaml.safe_load(f)

    analysis_config = config["analysis"]["simswap"]
    _dot = dot.DOT(use_video=False, save_folder=output_data_folder)
    _dot.use_cam = False
    option = _dot.build_option(
        swap_type="simswap",
        use_gpu=analysis_config.get("use_gpu", False),
        use_mask=analysis_config.get("opt_use_mask", False),
        gpen_type=analysis_config.get("gpen", None),
        gpen_path=analysis_config.get("gpen_path", None),
        crop_size=analysis_config.get("opt_crop_size", 224),
    )
    total_succeed = {}
    total_failed = {}
    for swap in swaps:
        source_list = swap[0]
        target_list = swap[1]
        # perform faceswap
        for source, target in zip(source_list, target_list):
            success, rejections = _dot.generate(
                option,
                source=source,
                target=target,
                duration=None,
                **analysis_config,
            )

        total_succeed = {**total_succeed, **success}
        total_failed = {**total_failed, **rejections}

    # save succeed face-swaps file
    if total_succeed:
        # append attribute list for source/target images
        for key, value in total_succeed.items():
            src_attr = (
                df.loc[df.index == value["source"]["path"].replace(local_root_path, "")]
                .iloc[0, 0:]
                .tolist()
            )
            tar_attr = (
                df.loc[df.index == value["target"]["path"].replace(local_root_path, "")]
                .iloc[0, 0:]
                .tolist()
            )

            total_succeed[key]["source"]["attr"] = src_attr
            total_succeed[key]["target"]["attr"] = tar_attr

        with open(os.path.join(save_folder, "swaps_succeed.json"), "w") as fp:
            json.dump(total_succeed, fp)

    # save failed face-swaps file
    if total_failed:
        with open(os.path.join(save_folder, "swaps_failed.json"), "w") as fp:
            json.dump(total_failed, fp)

    # format metadata to appropriate format
    formatted = format_swaps(total_succeed)

    # save file
    if formatted:
        with open(os.path.join(save_folder, f"{set}_label_swap.json"), "w") as fp:
            json.dump(formatted, fp)


def format_swaps(succeeds):
    formatted = {}
    for key, value in succeeds.items():
        # attributes of source image
        src_attr = np.asarray(value["source"]["attr"])
        # attributes of target image
        tar_attr = np.asarray(value["target"]["attr"])
        # attributes of swapped image. copy from target image
        swap_attr = tar_attr
        # transfer facial attributes from source image
        for idx in face_identity_features.keys():
            swap_attr[idx] = src_attr[idx]

        # swap-spoof-type-11, FaceSwap
        swap_attr[40] = 11
        # store in dict
        formatted[key] = swap_attr.tolist()

    return formatted


if __name__ == "__main__":
    main()
