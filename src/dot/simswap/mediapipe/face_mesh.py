#!/usr/bin/env python3

import os
import urllib.request
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .utils import face_align_ffhqandnewarc as face_align
from .utils import mediapipe_landmarks

# Model file path and download URL
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "saved_models", "mediapipe")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _ensure_model_downloaded():
    """Download the face landmarker model if not present."""
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"Downloading face landmarker model to {_MODEL_PATH}...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Download complete.")


class FaceMesh:
    """Wrapper class of Mediapipe's FaceMesh module. Extracts facial landmarks
    and performs face alignment.

    Args:
        static_image_mode (bool, optional):
            Indicates whether to treat input images as separated images(not video-stream). Defaults to True.
        max_num_faces (int, optional):
            Maximum allowed faces to examine in single image. Defaults to 1.
        refine_landmarks (bool, optional):
            Used to reduce jitter across multiple input images. Ignored if `static_image_mode = True`. Defaults to True.
        min_detection_confidence (float, optional):
            Threshold for a detection to considered successfull. Defaults to 0.5.
        mode (str, optional):
            Either ['None' | 'ffhq']. Instructs `estimate_norm` function for face alignment mode. Defaults to "None".
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        mode: str = "None",
    ):
        self.MediaPipeIds = mediapipe_landmarks.MediaPipeLandmarks
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.mode = mode

    def _get_centroid(self, landmarks: List[Any]) -> Tuple[float, float]:
        """Given a set of normalized landmarks/points finds centroid point

        Args:
            landmarks (List[Any]): List of relative points that form a polygon

        Returns:
            Tuple[float, float]: x,y coordinates of polygon centroid
        """
        x_li = [landmark.x for landmark in landmarks]
        y_li = [landmark.y for landmark in landmarks]
        _len = len(landmarks)
        return sum(x_li) / _len, sum(y_li) / _len

    def _create_face_landmarker(self) -> vision.FaceLandmarker:
        """Create and return a FaceLandmarker instance."""
        _ensure_model_downloaded()
        base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def get_face_landmarks(self, image: np.ndarray) -> Optional[np.array]:
        """Calls FaceMesh module from Mediapipe and retrieves related landmarks.
        The order of landmarks is important for face alignment

        landmarks: [
            [
                Left Eye,
                Right Eye,
                Nose Tip,
                Left Mouth Tip,
                Right Mouth Tip
            ]
        ]
        Extracted landmarks are normalized points based on width/height of the image

        @Eyes, Mediapipe returns a list of landmarks that forms a polygon
        `_get_centroid` method returns middle point
        @Mouth, Mediapipe returns a list of landmarks that forms a polygon.
        Only edge points are needed, `min/max` on x-axis


        Args:
            image (np.ndarray): [description]

        Returns:
            Optional[np.array]: [description]
        """
        # keypoints for all detected faces
        detection_kpss = []

        # Create landmarker instance
        landmarker = self._create_face_landmarker()

        # convert BGR image to RGB before processing.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detection = landmarker.detect(mp_image)
        if not detection.face_landmarks:
            return None

        # get width/height to de-normalize relative points
        height, width, _ = image.shape
        for face_landmarks in detection.face_landmarks:
            landmark_arr = np.empty((5, 2))
            # left eye: gets landmarks(polygon) and calculates center point
            left_eye = [
                face_landmarks[pt]
                for pt in self.MediaPipeIds.LEFT_EYE_OUTER
            ]
            centroid_left_eye = self._get_centroid(left_eye)
            landmark_arr[0] = np.array(
                (centroid_left_eye[0] * width, centroid_left_eye[1] * height)
            )

            # right eye: gets landmarks(polygon) and calculates center point
            right_eye = [
                face_landmarks[pt]
                for pt in self.MediaPipeIds.RIGHT_EYE_OUTER
            ]
            centroid_right_eye = self._get_centroid(right_eye)
            landmark_arr[1] = np.array(
                (centroid_right_eye[0] * width, centroid_right_eye[1] * height)
            )

            # nose tip
            nose_landmark = face_landmarks[self.MediaPipeIds.NOSE_TIP]
            landmark_arr[2] = np.array(
                (nose_landmark.x * width, nose_landmark.y * height)
            )

            # mouth region: finds the most left and most right point of outer lips region
            lips_outer_landmarks = [
                face_landmarks[pt] for pt in self.MediaPipeIds.LIPS_OUTER
            ]
            mouth_most_left_point = min(lips_outer_landmarks, key=lambda x: x.x)
            mouth_most_right_point = max(lips_outer_landmarks, key=lambda x: x.x)
            landmark_arr[3] = np.array(
                (
                    mouth_most_left_point.x * width,
                    mouth_most_left_point.y * height,
                )
            )
            landmark_arr[4] = np.array(
                (
                    mouth_most_right_point.x * width,
                    mouth_most_right_point.y * height,
                )
            )
            detection_kpss.append(landmark_arr)

        return np.array(detection_kpss)

    def get(
        self, image: np.ndarray, crop_size: Tuple[int, int]
    ) -> Optional[Tuple[List, List]]:
        """Driver method of face alignment

        Args:
            image (np.ndarray): raw cv2 image
            crop_size (Tuple[int, int]): face alignment crop size

        Returns:
            Optional[Tuple[List, List]]: List of face aligned images for each detected person
        """
        # gets facial landmarks using Face_Mesh model from MediaPipe
        landmarks = self.get_face_landmarks(image)
        if landmarks is None:
            print("ERROR: No face detected!")
            return None

        align_img_list = []
        M_list = []
        for i in range(landmarks.shape[0]):
            kps = landmarks[i]
            M, _ = face_align.estimate_norm(kps, crop_size, self.mode)
            align_img = cv2.warpAffine(
                image, M, (crop_size, crop_size), borderValue=0.0
            )
            align_img_list.append(align_img)
            M_list.append(M)

        return align_img_list, M_list
