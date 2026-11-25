"""Standalone Canny edge detection for masked DEM intensity images."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from lineament_analyzer.core.windowed import check_cancel


@dataclass(frozen=True)
class CannyParameters:
    low_threshold: float = 30.0
    high_threshold: float = 90.0
    aperture_size: int = 3
    l2_gradient: bool = True

    def validate(self) -> None:
        if (
            not np.isfinite(self.low_threshold)
            or not np.isfinite(self.high_threshold)
            or self.low_threshold < 0
            or self.high_threshold <= self.low_threshold
            or self.high_threshold > 255
        ):
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
        if self.aperture_size not in {3, 5, 7}:
            raise ValueError("Canny aperture size must be 3, 5 or 7")
        if not isinstance(self.l2_gradient, bool):
            raise TypeError("Canny L2 gradient option must be boolean")


def normalize_image(image: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Scale finite valid values and extend nearest valid values across NoData."""
    values = np.asarray(image, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("Canny requires a 2D image and matching validity mask")
    mask = mask & np.isfinite(values)
    if not np.any(mask):
        return None
    low, high = np.percentile(values[mask], [2, 98])
    if high <= low:
        low, high = float(np.min(values[mask])), float(np.max(values[mask]))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8), mask
    values = np.where(mask, values, low)
    scaled = np.clip((values - low) / (high - low) * 255.0, 0, 255)
    if np.any(~mask):
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        scaled[~mask] = scaled[tuple(indices[:, ~mask])]
    return scaled.astype(np.uint8), mask


def detect_edges(
    image: np.ndarray,
    valid: np.ndarray,
    parameters: CannyParameters,
    *,
    cancel=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a uint8 Canny edge image and the effective finite validity mask."""
    parameters.validate()
    check_cancel(cancel)
    normalized = normalize_image(image, valid)
    if normalized is None:
        return np.zeros(np.shape(image), dtype=np.uint8), np.zeros(np.shape(image), dtype=bool)
    raster, mask = normalized
    edges = cv2.Canny(
        raster,
        parameters.low_threshold,
        parameters.high_threshold,
        apertureSize=parameters.aperture_size,
        L2gradient=parameters.l2_gradient,
    )
    check_cancel(cancel)
    return edges, mask
