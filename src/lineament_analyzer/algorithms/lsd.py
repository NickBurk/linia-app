"""LSD segment detection using OpenCV's image processing primitive."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LsdParameters:
    min_length: float = 10.0
    scale: float = 0.8


def detect(image: np.ndarray, valid: np.ndarray, params: LsdParameters) -> np.ndarray:
    """Return Nx4 pixel endpoint coordinates of candidate segments."""
    if params.min_length < 0 or not 0 < params.scale <= 1:
        raise ValueError("Invalid LSD parameters")
    finite = np.isfinite(image) & valid
    if not np.any(finite):
        return np.empty((0, 4), dtype=float)
    low, high = np.percentile(image[finite], [2, 98])
    normalized = np.clip((np.where(finite, image, low) - low) / max(high - low, 1e-12) * 255, 0, 255)
    raster = normalized.astype(np.uint8)
    raster[~finite] = 0
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, scale=params.scale).detect(raster)[0]
    if lines is None:
        return np.empty((0, 4), dtype=float)
    segments = lines.reshape(-1, 4).astype(float)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    return segments[lengths >= params.min_length]
