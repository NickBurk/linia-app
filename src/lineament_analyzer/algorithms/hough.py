"""Standalone probabilistic Hough line extraction from an edge image."""

from dataclasses import dataclass

import cv2
import numpy as np

from lineament_analyzer.core.windowed import check_cancel


@dataclass(frozen=True)
class HoughParameters:
    vote_threshold: int = 20
    min_segment_length_m: float = 100.0
    max_gap_m: float = 20.0

    def validate(self) -> None:
        if self.vote_threshold < 1:
            raise ValueError("Hough threshold must be a positive integer")
        if not np.isfinite(self.min_segment_length_m) or self.min_segment_length_m <= 0:
            raise ValueError("Minimum segment length must be positive and finite")
        if not np.isfinite(self.max_gap_m) or self.max_gap_m < 0:
            raise ValueError("Maximum gap must be non-negative and finite")


def detect_lines(
    edges: np.ndarray,
    parameters: HoughParameters,
    *,
    pixel_size_m: float,
    cancel=None,
) -> np.ndarray:
    """Return probabilistic Hough line candidates as Nx4 pixel endpoints."""
    parameters.validate()
    edge_image = np.asarray(edges)
    if edge_image.ndim != 2 or edge_image.dtype != np.uint8:
        raise ValueError("Probabilistic Hough requires a 2D uint8 edge image")
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("Pixel size in metres must be positive and finite")
    check_cancel(cancel)
    minimum_pixels = max(1, round(parameters.min_segment_length_m / pixel_size_m))
    maximum_gap_pixels = max(0, round(parameters.max_gap_m / pixel_size_m))
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=parameters.vote_threshold,
        minLineLength=minimum_pixels,
        maxLineGap=maximum_gap_pixels,
    )
    check_cancel(cancel)
    if lines is None:
        return np.empty((0, 4), dtype=float)
    return lines.reshape(-1, 4).astype(float)
