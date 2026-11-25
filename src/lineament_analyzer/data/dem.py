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
        return Dem(data.data, valid, dataset.transform, dataset.crs, source)
