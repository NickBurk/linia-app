"""Bounded-memory raster processing with halo and atomic output publication."""

import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import rasterio
from rasterio.windows import Window

from lineament_analyzer.core.terrain import (
    UnitWarning,
    derivative,
    elevation_scale,
    metric_jacobian,
)
from lineament_analyzer.data.dem import Dem


class CancelledError(Exception):
    """Cooperative cancellation requested by the caller."""


def check_cancel(cancel):
    if cancel is not None and cancel():
        raise CancelledError("Processing cancelled")


def process_dem(
    source,
    destination,
    kind,
    *,
    block_size=512,
    memory_limit_mb=256,
    progress=None,
    cancel=None,
    **parameters,
):
    """Read halo windows, crop local results, and write a georeferenced float64 TIFF.

    The budget uses a conservative 256 bytes per buffered pixel for NumPy/SciPy
    intermediates. GDAL cache is bounded separately. Output is replaced only on
    success. Cancellation and failure preserve any existing destination.
    """
    halo = parameters.get("tpi_size", 9) // 2 if kind == "tpi" else 1
    if block_size < 2 or memory_limit_mb < 1 or halo < 1:
        raise ValueError("Invalid block size or memory budget")
    side = int(np.sqrt(memory_limit_mb * 1024**2 / 256)) - 2 * halo
    tile = min(block_size, side)
    if tile < 2:
        raise ValueError("Memory budget too small for the operation halo")
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".tif", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with rasterio.Env(GDAL_CACHEMAX=16 * 1024**2), rasterio.open(source) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float64",
                nodata=np.nan,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="IF_SAFER",
            )
            total = ((src.width + tile - 1) // tile) * ((src.height + tile - 1) // tile)
            completed = 0
            # Validate metadata once, avoiding one warning per block.
            elevation_scale(parameters.get("elevation_unit") or src.units[0])
            metadata = Dem(
                np.empty((1, 1)),
                np.ones((1, 1), bool),
                src.transform,
                src.crs,
                source,
                src.units[0],
            )
            if kind != "tpi":
                metric_jacobian(metadata)
            with rasterio.open(temporary, "w", **profile) as dst:
                dst.update_tags(
                    source=str(source),
                    derivative=kind,
                    z_factor=parameters.get("vertical_scale", 1),
                )
                dst.set_band_unit(
                    1, "m" if kind == "tpi" else "degree" if kind in {"slope", "aspect"} else "1"
                )
                for row in range(0, src.height, tile):
                    for col in range(0, src.width, tile):
                        check_cancel(cancel)
                        width, height = min(tile, src.width - col), min(tile, src.height - row)
                        left, top = max(0, col - halo), max(0, row - halo)
                        right = min(src.width, col + width + halo)
                        bottom = min(src.height, row + height + halo)
                        window = Window(left, top, right - left, bottom - top)
                        data = src.read(1, window=window, masked=True).astype(float)
                        values = data.data * src.scales[0] + src.offsets[0]
                        valid = ~np.ma.getmaskarray(data) & np.isfinite(values)
                        # Unit conversion once here avoids repeated unknown-unit warnings.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UnitWarning)
                            result = derivative(
                                Dem(
                                    values,
                                    valid,
                                    src.window_transform(window),
                                    src.crs,
                                    source,
                                    src.units[0],
                                ),
                                kind,
                                **parameters,
                            )
                        cropped = result[
                            row - top : row - top + height, col - left : col - left + width
                        ]
                        check_cancel(cancel)
                        dst.write(cropped, 1, window=Window(col, row, width, height))
                        completed += 1
                        if progress is not None:
                            progress(int(100 * completed / total))
            check_cancel(cancel)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
