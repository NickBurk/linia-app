"""Basic terrain derivatives for projected DEMs."""

import numpy as np
from scipy.ndimage import uniform_filter

from lineament_analyzer.data.dem import Dem


def derivative(dem: Dem, kind: str, *, azimuth: float = 315, altitude: float = 45,
               tpi_size: int = 9, vertical_scale: float = 1) -> np.ndarray:
    """Calculate a terrain derivative, returning NaN at invalid pixels."""
    if dem.crs is not None and dem.crs.is_geographic:
        raise ValueError("Reproject geographic DEM to a projected CRS before terrain analysis")
    if vertical_scale <= 0 or tpi_size < 3 or tpi_size % 2 == 0:
        raise ValueError("Invalid scale or TPI window")
    z = np.where(dem.valid, dem.values * vertical_scale, np.nan)
    filled = np.where(dem.valid, z, 0)
    if kind == "tpi":
        count = uniform_filter(dem.valid.astype(float), size=tpi_size, mode="nearest")
        mean = uniform_filter(filled, size=tpi_size, mode="nearest") / np.maximum(count, 1e-12)
        result = z - mean
    else:
        dy, dx = np.gradient(filled, dem.resolution[1], dem.resolution[0])
        slope = np.arctan(np.hypot(dx, dy))
        if kind == "slope":
            result = np.degrees(slope)
        elif kind == "aspect":
            result = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360
        elif kind == "hillshade":
            zenith = np.radians(90 - altitude)
            light = np.radians(azimuth)
            aspect = np.arctan2(-dx, dy)
            result = 255 * (np.cos(zenith) * np.cos(slope)
                            + np.sin(zenith) * np.sin(slope) * np.cos(light - aspect))
            result = np.clip(result, 0, 255)
        else:
            raise ValueError(f"Unknown derivative: {kind}")
    result[~dem.valid] = np.nan
    return result
