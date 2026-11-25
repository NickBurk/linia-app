"""Shared NoData support checks for image-based line detectors."""

import numpy as np
from scipy.ndimage import binary_erosion


def detector_support_mask(valid: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Prepare an eroded support mask once per detector invocation."""
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Validity mask must be 2D")
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return binary_erosion(mask, iterations=radius, border_value=1) if radius else mask


def sample_mask_support(
    support: np.ndarray,
    segment: np.ndarray,
    *,
    step: float = 0.25,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample a segment densely and reject it near NoData or outside the image.

    Image borders remain valid because they are actual raster edges. Returned
    integer arrays index the sampled pixels in the supplied support mask.
    """
    mask = np.asarray(support, dtype=bool)
    line = np.asarray(segment, dtype=float)
    if mask.ndim != 2 or line.shape != (4,) or not np.isfinite(line).all():
        raise ValueError("Expected a 2D validity mask and four finite segment coordinates")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Sample step must be positive")
    if not np.any(mask):
        return None

    x1, y1, x2, y2 = line
    length = float(np.hypot(x2 - x1, y2 - y1))
    count = max(2, int(np.ceil(length / step)) + 1)
    x = np.rint(np.linspace(x1, x2, count)).astype(int)
    y = np.rint(np.linspace(y1, y2, count)).astype(int)
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    if not inside.all() or not mask[y, x].all():
        return None
    return x, y


def sample_valid_support(
    valid: np.ndarray,
    segment: np.ndarray,
    *,
    radius: int = 1,
    step: float = 0.25,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convenience wrapper to erode a validity mask and sample one segment."""
    return sample_mask_support(
        detector_support_mask(valid, radius=radius), segment, step=step
    )
