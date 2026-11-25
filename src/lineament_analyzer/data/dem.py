"""GeoTIFF DEM loading with mask and georeferencing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


@dataclass(frozen=True)
class Dem:
    values: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: CRS | None
    path: Path

    elevation_unit: str | None = None

    @property
    def resolution(self) -> tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)


def load_dem(path: str | Path) -> Dem:
    """Read a single-band DEM, retaining its mask and spatial metadata."""
    source = Path(path)
    with rasterio.open(source) as dataset:
        if dataset.count < 1:
            raise ValueError("GeoTIFF has no raster bands")
        data = dataset.read(1, masked=True).astype(np.float64)
        valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
        return Dem(
            data.data * dataset.scales[0] + dataset.offsets[0],
            valid,
            dataset.transform,
            dataset.crs,
            source,
            dataset.units[0],
        )


def load_preview(path: str | Path, max_size: int = 1600) -> Dem:
    """Read a bounded nearest-neighbour preview with its own pixel transform."""
    if max_size < 2:
        raise ValueError("Preview size must be at least 2")
    with rasterio.open(path) as dataset:
        ratio = min(1, max_size / max(dataset.width, dataset.height))
        width = max(1, round(dataset.width * ratio))
        height = max(1, round(dataset.height * ratio))
        data = dataset.read(1, out_shape=(height, width), masked=True).astype(float)
        values = data.data * dataset.scales[0] + dataset.offsets[0]
        valid = ~np.ma.getmaskarray(data) & np.isfinite(values)
        transform = dataset.transform * Affine.scale(dataset.width / width, dataset.height / height)
        return Dem(values, valid, transform, dataset.crs, Path(path), dataset.units[0])
