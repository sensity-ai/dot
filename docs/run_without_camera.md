
# Run dot on image and video files instead of camera feed

## Using Images

```bash
dot -c ./configs/simswap.yaml --target data/ --source "data/" --save_folder test_local/ --use_image --use_gpu
```

```bash
dot -c ./configs/faceswap_cv2.yaml --target data/ --source "data/" --save_folder test_local/ --use_image --use_gpu
```

## Using Videos

```
dot -c ./configs/simswap.yaml --target "/path/to/driving/video" --source "data/image.png"  --save_folder test_local/ --use_gpu --use_video
```

```
dot -c ./configs/fomm.yaml --target "/path/to/driving/video" --source "data/image.png"  --save_folder test_local/  --use_gpu --use_video
```

## Faceswap images from directory (Simswap)

You can pass a `--source` folder with images and some `--target` images. Faceswapped images will be generated at `--save_folder` including a metadata json file.

```bash
python scripts/image_swap.py --config <path_to_config/config.yaml> --source <path_to_source_images_folder> --target <path_to_target_images_folder> --save_folder <output_dir> --limit 100
```

## Faceswap images from metadata (SimSwap)

```bash
python scripts/metadata_swap.py --config <path_to_config/config.yaml> --local_root_path <path_to_root_directory> --metadata <path_to_metadata_file> --set <train_or_test_dataset> --save_folder <path_to_output_folder> --limit 100
```

## Faceswap on video files (SimSwap)

```bash
python scripts/video_swap.py -c <path_to_simpswap_config/config.yaml> -s <path_to_source_images> -t <path_to_target_videos> -o <path_to_output_folder> -d 5 -l 5
```

`-d 5` is optional to trim video in seconds
`-l 5` is optional limit total swaps
