"""Shared export checks for legacy and source/preview detection grids."""

from pathlib import Path

import numpy as np
import rasterio


def validate_result_grid(result, dem):
    if result.source_crs_wkt != dem.crs.to_wkt():
        raise ValueError("Detection result CRS does not match the export DEM")
    if (result.source_transform == tuple(dem.transform)[:6]
            and result.source_shape == dem.values.shape):
        return
    source = result.analysis_metadata.get("dem")
    if source is None or Path(source).resolve() != dem.path.resolve():
        raise ValueError("Detection result raster does not match the export DEM")
    with rasterio.open(source) as dataset:
        if result.source_shape != dataset.shape or result.source_transform != tuple(dataset.transform)[:6]:
            raise ValueError("Detection result grid does not match either the source or preview DEM")


def validate_coordinates(coordinates):
    values = np.asarray(coordinates, dtype=float)
    if (
        values.ndim != 2 or values.shape[1] != 2 or len(values) < 2
        or not np.isfinite(values).all()
        or not np.any(np.diff(values, axis=0))
    ):
        raise ValueError("Line geometry requires at least two distinct finite vertices")
    return values
