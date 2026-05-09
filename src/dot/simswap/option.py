#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dot.commons.utils import get_device, get_model_base_path, VIDEO_EXTENSIONS
from dot.commons.shared_detector import get_static_detector, get_live_detector
from dot.simswap.fs_model import create_model
from dot.simswap.parsing_model.model import BiSeNet
from dot.simswap.util.norm import SpecificNorm
from dot.simswap.util.reverse2original import reverse2wholeimage
from dot.simswap.util.util import _totensor



class SimswapOption:
    """SimSwap model wrapper for the live webcam path."""

    def __init__(
        self,
        use_gpu=True,
        use_mask=False,
        crop_size=224,
    ):
        self.use_gpu = use_gpu
        self.crop_size = crop_size
        self.use_mask = use_mask

    def create_model(  # type: ignore
        self,
        detection_threshold=0.6,
        det_size=(640, 640),
        opt_verbose=False,
        opt_crop_size=224,
        opt_gpu_ids=[0],
        opt_fp16=False,
        checkpoints_dir=None,
        opt_name="people",
        opt_resize_or_crop="scale_width",
        opt_load_pretrain="",
        opt_which_epoch="latest",
        opt_continue_train="store_true",
        parsing_model_path=None,
        arcface_model_path=None,
        max_num_faces=1,
        min_detection_confidence=None,
        min_tracking_confidence=0.5,
        **kwargs
    ) -> None:
        model_base = get_model_base_path()
        if parsing_model_path is None:
            parsing_model_path = f"{model_base}/parsing_model/checkpoint/79999_iter.pth"
        if arcface_model_path is None:
            arcface_model_path = f"{model_base}/arcface_model/arcface_checkpoint.tar"
        if checkpoints_dir is None:
            checkpoints_dir = f"{model_base}/checkpoints"

        # preprocess_f
        self.transformer_Arcface = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if opt_crop_size == 512:
            opt_which_epoch = 550000
            opt_name = "512"
            self.mode = "ffhq"
        else:
            self.mode = "None"

        # For camera mode: use video-stream tracking. For batch: static mode is faster/lighter.
        is_camera = not (hasattr(self, '_source_display_frame'))
        detection_confidence = (
            detection_threshold
            if min_detection_confidence is None
            else min_detection_confidence
        )
        self.detect_model = get_live_detector(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) if is_camera else get_static_detector(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            mode=self.mode,
        )

        # Tod check if we need this
        self.spNorm = SpecificNorm(use_gpu=self.use_gpu)
        if self.use_mask:
            n_classes = 19
            self.net = BiSeNet(n_classes=n_classes)
            if self.use_gpu:
                device = get_device()
                self.net.to(device)
                self.net.load_state_dict(
                    torch.load(parsing_model_path, weights_only=False, map_location=device)
                )
            else:
                self.net.cpu()
                self.net.load_state_dict(
                    torch.load(parsing_model_path, weights_only=False, map_location="cpu")
                )

            self.net.eval()
        else:
            self.net = None

        torch.nn.Module.dump_patches = False

        self.model = create_model(
            opt_verbose,
            opt_crop_size,
            opt_fp16,
            opt_gpu_ids,
            checkpoints_dir,
            opt_name,
            opt_resize_or_crop,
            opt_load_pretrain,
            opt_which_epoch,
            opt_continue_train,
            arcface_model_path,
            use_gpu=self.use_gpu,
        )
        self.model.eval()

    def _embed_single_frame(self, bgr_frame: np.ndarray):
        """Compute ArcFace embedding for one BGR frame. Returns (embedding, crop) or None."""
        result = self.detect_model.get(bgr_frame, self.crop_size)
        if result is None:
            return None
        crop_bgr = result[0][0]
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        img_a = self.transformer_Arcface(pil_img)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        device = get_device() if self.use_gpu else "cpu"
        img_id = img_id.to(device)
        img_id_ds = F.interpolate(img_id, size=(112, 112))
        emb = self.model.netArc(img_id_ds).detach().cpu()
        emb = emb / np.linalg.norm(emb.numpy(), axis=1, keepdims=True)
        return emb, crop_bgr

    def _set_embedding(self, embedding: "torch.Tensor") -> None:
        """Normalise and store the final source embedding on the correct device."""
        device = get_device() if self.use_gpu else "cpu"
        self.source_image = embedding.to(device)

    # ── public API ─────────────────────────────────────────────────────────────

    def change_option(self, image, num_video_samples: int = 20, **kwargs) -> None:
        """Sets the source identity.

        Accepts:
          • np.ndarray  – a single BGR image (existing behaviour)
          • str         – path to an image OR a video file.
                          When a video is given, embeddings are averaged across
                          up to `num_video_samples` evenly-spaced frames for a
                          much more robust identity representation.
        """
        if isinstance(image, str) and os.path.splitext(image)[1] in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(image)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {image}")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # WebM and some containers report -1 or garbage for frame count.
            # Fall back to sequential read-and-sample in that case.
            can_seek = total > 0 and total < 1_000_000

            embeddings = []
            display_crop = None
            fname = os.path.basename(image)

            if can_seek:
                indices = np.linspace(0, total - 1,
                                      min(num_video_samples, total), dtype=int)
                print(f"[dot] Sampling {len(indices)} frames from {fname} …")
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    out = self._embed_single_frame(frame)
                    if out is None:
                        continue
                    emb, crop = out
                    embeddings.append(emb)
                    del frame, crop
                    if display_crop is None:
                        display_crop = crop
            else:
                # Sequential scan — sample every Nth frame until we have enough
                print(f"[dot] Scanning {fname} sequentially (WebM/no seek) …")
                frame_idx = 0
                step = 3  # sample every 3rd frame for speed
                while len(embeddings) < num_video_samples:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if frame_idx % step == 0:
                        out = self._embed_single_frame(frame)
                        if out is not None:
                            emb, crop = out
                            embeddings.append(emb)
                            if display_crop is None:
                                display_crop = crop
                    frame_idx += 1

            cap.release()

            if not embeddings:
                raise ValueError(f"No faces detected in video: {image}")

            print(f"[dot] Averaged {len(embeddings)} face embeddings from {fname} ✅")
            avg = torch.stack(embeddings).mean(dim=0)
            avg = avg / np.linalg.norm(avg.numpy(), axis=1, keepdims=True)
            self._source_display_frame = display_crop
            self._set_embedding(avg)
            return

        # ── image path ─────────────────────────────────────────────────────────
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot read image: {image}")

        # ── numpy array (original single-image path) ───────────────────────────
        result = self.detect_model.get(image, self.crop_size)
        if result is None:
            raise ValueError("No face detected in source image.")
        img_a_align_crop = result[0]
        img_a_align_crop_pil = Image.fromarray(
            cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB)
        )
        img_a = self.transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        device = get_device() if self.use_gpu else "cpu"
        img_id = img_id.to(device)

        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        source_image = self.model.netArc(img_id_downsample)
        source_image = source_image.detach().cpu()
        source_image = source_image / np.linalg.norm(
            source_image.numpy(), axis=1, keepdims=True
        )
        self._source_display_frame = None
        self._set_embedding(source_image)

    def process_image(self, image: np.array, **kwargs) -> np.array:
        """Main process of simswap method. There are 3 main steps:
        * face detection and alignment of target image.
        * swap with `self.source_image`.
        * face segmentation and reverse to whole image.

        Args:
            image (np.array): Target frame where face from `self.source_image` will be swapped with.

        Returns:
            np.array: Resulted face-swap image
        """

        detect_results = self.detect_model.get(image, self.crop_size)
        if detect_results is not None:
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []
            frame_align_crop_tenor_list = []
            for frame_align_crop in frame_align_crop_list:
                if self.use_gpu:
                    frame_align_crop_tenor = _totensor(
                        cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB)
                    )[None, ...].to(get_device())
                else:
                    frame_align_crop_tenor = _totensor(
                        cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB)
                    )[None, ...].cpu()

                swap_result = self.model(
                    None, frame_align_crop_tenor, self.source_image, None, True
                )[0]
                swap_result_list.append(swap_result)
                frame_align_crop_tenor_list.append(frame_align_crop_tenor)

            result_frame = reverse2wholeimage(
                frame_align_crop_tenor_list,
                swap_result_list,
                frame_mat_list,
                self.crop_size,
                image,
                pasring_model=self.net,
                use_mask=self.use_mask,
                norm=self.spNorm,
                use_gpu=self.use_gpu,
                use_cam=kwargs.get("use_cam", True),
                color_match=kwargs.get("natural_color_match", False),
                color_match_strength=kwargs.get("natural_color_match_strength", 0.65),
                detail_enhance=kwargs.get("natural_detail_enhance", False),
                detail_enhance_strength=kwargs.get(
                    "natural_detail_enhance_strength", 0.12
                ),
                preserve_occluders=kwargs.get("natural_preserve_occluders", False),
                occluder_threshold=kwargs.get("natural_occluder_threshold", 0.10),
                occluder_strength=kwargs.get("natural_occluder_strength", 0.90),
                blend_mode=kwargs.get("natural_blend_mode", "alpha"),
                blend_strength=kwargs.get("natural_blend_strength", 1.0),
                mask_blur=kwargs.get("natural_mask_blur", 0),
            )
            return result_frame
        else:
            return image
