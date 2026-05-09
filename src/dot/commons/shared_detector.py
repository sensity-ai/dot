#!/usr/bin/env python3

# Lazy import to avoid circular dependency:
#   shared_detector -> face_mesh -> simswap.__init__ -> option -> shared_detector
# FaceMesh is imported on first call, not at module load time.


class SharedFaceDetector:
    _instances = {}

    @classmethod
    def get_detector(cls, static_image_mode=True, **kwargs):
        key = (static_image_mode, frozenset(kwargs.items()))
        if key not in cls._instances:
            from ..simswap.mediapipe.face_mesh import FaceMesh
            cls._instances[key] = FaceMesh(static_image_mode=static_image_mode, **kwargs)
        return cls._instances[key]


# Convenience functions
def get_static_detector(**kwargs):
    return SharedFaceDetector.get_detector(static_image_mode=True, **kwargs)


def get_live_detector(**kwargs):
    return SharedFaceDetector.get_detector(static_image_mode=False, **kwargs)
