"""Metric terrain derivatives with explicit units and conservative NoData support."""

import warnings

import numpy as np
from pyproj import CRS
from scipy.ndimage import uniform_filter

from lineament_analyzer.data.dem import Dem


class UnitWarning(UserWarning):
    """Input units are missing and a documented assumption was used."""


def elevation_scale(unit: str | None) -> float:
    units = {
        "m": 1,
        "metre": 1,
        "meter": 1,
        "meters": 1,
        "metres": 1,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
        "us survey foot": 1200 / 3937,
        "cm": 0.01,
        "mm": 0.001,
    }
    if unit is not None and unit.lower().strip() in units:
        return units[unit.lower().strip()]
    warnings.warn(
        f"Unknown elevation unit {unit!r}; assuming metres. "
        "Set the elevation unit and Z-factor before interpreting terrain.",
        UnitWarning,
        stacklevel=2,
    )
    return 1.0


def metric_jacobian(dem: Dem):
    """Return pixel-to-local east/north Jacobian, in metres per pixel."""
    t = dem.transform
    if dem.crs is None:
        warnings.warn(
            "Missing horizontal CRS; assuming map units are metres. "
            "Assign the source CRS before interpreting terrain.",
            UnitWarning,
            stacklevel=2,
        )
        return t.a, t.b, t.d, t.e
    crs = CRS(dem.crs)
    if crs.is_projected:
        factors = [axis.unit_conversion_factor for axis in crs.axis_info[:2]]
        if len(factors) != 2 or not all(np.isfinite(factors)):
            raise ValueError("Unknown horizontal CRS units")
        return t.a * factors[0], t.b * factors[0], t.d * factors[1], t.e * factors[1]
    if not crs.is_geographic:
        raise ValueError("Terrain requires a projected or geographic horizontal CRS")
    # Geographic rasters use x=longitude, y=latitude irrespective of EPSG axis order.
    if any(axis.unit_name != "degree" for axis in crs.axis_info[:2]):
        raise ValueError("Geographic raster angular units must be degrees")
    row, col = np.indices(dem.values.shape, dtype=float)
    latitude = t.d * (col + 0.5) + t.e * (row + 0.5) + t.f
    if np.any(np.abs(latitude) >= 90):
        raise ValueError("Geographic DEM reaches a pole or has invalid latitude")
    phi = np.radians(latitude)
    ellipsoid = crs.ellipsoid
    a = ellipsoid.semi_major_metre
    e2 = 1 - (ellipsoid.semi_minor_metre / a) ** 2
    denominator = 1 - e2 * np.sin(phi) ** 2
    east = np.pi / 180 * a / np.sqrt(denominator) * np.cos(phi)
    north = np.pi / 180 * a * (1 - e2) / denominator**1.5
    return east * t.a, east * t.b, north * t.d, north * t.e


def derivative(
    dem: Dem,
    kind: str,
    *,
    azimuth: float = 315,
    altitude: float = 45,
    tpi_size: int = 9,
    vertical_scale: float = 1,
    elevation_unit: str | None = None,
) -> np.ndarray:
    """Use physical metres, with Z-factor applied after elevation unit conversion.

    Gradients use one-sided differences at true raster edges and require valid
    central samples and both derivative stencils. TPI requires a complete window,
    so missing samples and unsupported raster borders produce NaN.
    """
    if kind not in {"hillshade", "slope", "aspect", "tpi"}:
        raise ValueError(f"Unknown derivative: {kind}")
    if not np.isfinite(vertical_scale) or vertical_scale <= 0:
        raise ValueError("Z-factor must be finite and positive")
    if not isinstance(tpi_size, int) or tpi_size < 3 or tpi_size % 2 == 0:
        raise ValueError("TPI window must be an odd integer of at least 3")
    if not np.isfinite([azimuth, altitude]).all() or not 0 <= altitude <= 90:
        raise ValueError("Invalid illumination angles")
    valid = dem.valid & np.isfinite(dem.values)
    z = np.where(
        valid,
        dem.values * elevation_scale(elevation_unit or dem.elevation_unit) * vertical_scale,
        np.nan,
    )
    if kind == "tpi":
        support = uniform_filter(valid.astype(float), size=tpi_size, mode="constant", cval=0)
        mean = uniform_filter(np.where(valid, z, 0), size=tpi_size, mode="constant", cval=0)
        return np.where(valid & np.isclose(support, 1, atol=1e-12, rtol=0), z - mean, np.nan)
    if min(z.shape) < 2:
        raise ValueError("Terrain gradients require at least two rows and columns")
    a, b, d, e = metric_jacobian(dem)
    determinant = a * e - b * d
    if np.any(~np.isfinite(determinant)) or np.any(np.abs(determinant) < 1e-15):
        raise ValueError("Singular pixel transform")
    row_gradient, col_gradient = np.gradient(z)
    east = (e * col_gradient - d * row_gradient) / determinant
    north = (-b * col_gradient + a * row_gradient) / determinant
    slope = np.arctan(np.hypot(east, north))
    aspect = np.arctan2(-east, -north)
    if kind == "slope":
        result = np.degrees(slope)
    elif kind == "aspect":
        result = np.degrees(aspect) % 360
        result[np.hypot(east, north) < 1e-12] = np.nan
    else:
        zenith = np.radians(90 - altitude)
        result = np.clip(
            255
            * (
                np.cos(zenith) * np.cos(slope)
                + np.sin(zenith) * np.sin(slope) * np.cos(np.radians(azimuth) - aspect)
            ),
            0,
            255,
        )
    result[~valid] = np.nan
    return result
