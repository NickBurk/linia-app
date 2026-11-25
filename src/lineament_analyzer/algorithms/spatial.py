"""CRS-aware raster sampling helpers shared by independent detectors."""

import numpy as np
from pyproj import CRS as ProjCRS
from rasterio.warp import transform as reproject_points

from lineament_analyzer.data.dem import Dem


def pixel_size_metres(dem: Dem) -> float:
    """Estimate geometric-mean ground sample distance at the DEM centre."""
    if dem.crs is None:
        raise ValueError("A DEM CRS is required for metre-based detector parameters")
    crs = ProjCRS.from_user_input(dem.crs)
    geod = crs.get_geod()
    col = max(0.0, (dem.values.shape[1] - 1) / 2)
    row = max(0.0, (dem.values.shape[0] - 1) / 2)
    center = dem.transform * (col + 0.5, row + 0.5)
    across = dem.transform * (col + 1.5, row + 0.5)
    down = dem.transform * (col + 0.5, row + 1.5)
    xs, ys = reproject_points(
        crs,
        "EPSG:4326",
        [center[0], across[0], center[0], down[0]],
        [center[1], across[1], center[1], down[1]],
    )
    if (
        not np.isfinite(np.column_stack((xs, ys))).all()
        or np.any(np.abs(xs) > 180)
        or np.any(np.abs(ys) > 90)
    ):
        raise ValueError("DEM affine produced invalid geographic coordinates")
    x_size = geod.inv(xs[0], ys[0], xs[1], ys[1])[2]
    y_size = geod.inv(xs[2], ys[2], xs[3], ys[3])[2]
    size = float(np.sqrt(abs(x_size * y_size)))
    if not np.isfinite(size) or size <= 0:
        raise ValueError("DEM pixel size must be finite and non-zero")
    return size
