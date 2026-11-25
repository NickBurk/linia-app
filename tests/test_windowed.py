import numpy as np
import pytest
import rasterio
from affine import Affine

from lineament_analyzer.core.terrain import derivative
from lineament_analyzer.core.windowed import CancelledError, process_dem
from lineament_analyzer.data.dem import load_dem, load_preview


@pytest.fixture
def raster(tmp_path):
    path = tmp_path / "input.tif"
    y, x = np.indices((57, 61))
    z = (np.sin(x / 6) * 20 + y * 2).astype(float)
    z[15:19, 16:20] = -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=61,
        height=57,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        nodata=-9999,
        transform=Affine(0.001, 0, 10, 0, -0.001, 60),
    ) as dst:
        dst.write(z, 1)
        dst.set_band_unit(1, "ft")
        dst.scales = (2,)
        dst.offsets = (10,)
    return path


@pytest.mark.parametrize("kind", ["slope", "aspect", "hillshade", "tpi"])
@pytest.mark.parametrize("block", [7, 16, 64])
def test_window_matches_whole(raster, tmp_path, kind, block):
    target = tmp_path / "result.tif"
    progress = []
    process_dem(
        raster,
        target,
        kind,
        block_size=block,
        memory_limit_mb=1,
        progress=progress.append,
        tpi_size=5,
    )
    dem = load_dem(raster)
    result = load_dem(target)
    np.testing.assert_allclose(
        result.values, derivative(dem, kind, tpi_size=5), equal_nan=True, atol=1e-8
    )
    assert result.crs == dem.crs
    assert result.transform == dem.transform
    assert progress[-1] == 100


def test_cancel_preserves_destination(raster, tmp_path):
    target = tmp_path / "existing.tif"
    target.write_bytes(b"original")
    with pytest.raises(CancelledError):
        process_dem(raster, target, "slope", cancel=lambda: True)
    assert target.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["existing.tif", "input.tif"]


def test_preview_transform(raster):
    preview = load_preview(raster, 20)
    original = load_dem(raster)
    assert max(preview.values.shape) <= 20
    assert preview.transform * (preview.values.shape[1], preview.values.shape[0]) == (
        original.transform * (61, 57)
    )


def test_reads_are_bounded_windows(raster, tmp_path, monkeypatch):
    original_open = rasterio.open
    reads = []

    class Reader:
        def __init__(self, dataset):
            self.dataset = dataset

        def __getattr__(self, name):
            return getattr(self.dataset, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.dataset.close()

        def read(self, *args, **kwargs):
            window = kwargs.get("window")
            assert window is not None
            assert max(window.width, window.height) <= 18  # tile 16 + gradient halo
            reads.append(window)
            return self.dataset.read(*args, **kwargs)

    def opened(path, *args, **kwargs):
        dataset = original_open(path, *args, **kwargs)
        return Reader(dataset) if Path(path) == raster else dataset

    from pathlib import Path

    monkeypatch.setattr(rasterio, "open", opened)
    process_dem(raster, tmp_path / "result.tif", "slope", block_size=16)
    assert len(reads) == 16


def test_midway_failure_preserves_destination(raster, tmp_path):
    target = tmp_path / "existing.tif"
    target.write_bytes(b"original")

    def progress(value):
        raise OSError("synthetic write failure")

    with pytest.raises(OSError, match="synthetic"):
        process_dem(raster, target, "slope", block_size=16, progress=progress)
    assert target.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["existing.tif", "input.tif"]
