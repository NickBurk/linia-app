from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.algorithms.filter_banks import (
    DirectionalParameters,
    FilterDetector,
    LogGaborParameters,
    ShearletParameters,
    WaveletParameters,
    reconstruct_shearlets,
    response_map,
    shearlet_components,
    wavelet_components,
)
from lineament_analyzer.core.windowed import CancelledError
from lineament_analyzer.data.dem import Dem


@pytest.mark.parametrize("shape", [(51, 68), (64, 65), (1, 13)])
def test_decompositions_reconstruct(shape):
    image = np.random.default_rng(20).normal(size=shape)
    assert np.allclose(sum(band for _, band in wavelet_components(image, 3)), image)
    assert np.allclose(reconstruct_shearlets(shearlet_components(image, 3)), image, atol=1e-11)


@pytest.mark.parametrize(("method", "p"), [
    ("directional", DirectionalParameters(levels=2)),
    ("wavelet", WaveletParameters(levels=2)),
    ("log-gabor", LogGaborParameters(levels=2)),
    ("shearlet", ShearletParameters(levels=2)),
])
@pytest.mark.parametrize("vertical", [False, True])
def test_response_and_detections(method, p, vertical):
    y, x = np.indices((80, 100))
    z = np.exp(-((x - 50) / 2.5)**2 if vertical else -((y - 40) / 2.5)**2)
    valid = np.ones(z.shape, dtype=bool)
    valid[5:12, 5:12] = False
    z[~valid] = np.nan
    dem = Dem(z, valid, Affine(0.0001, 0, 10, 0, -0.0001, 60),
              CRS.from_epsg(4326), Path("ridge.tif"), "m")
    response = response_map(dem, p, method)
    assert response.strength[~valid].max() == 0
    assert response.strength[40, 50] > 0.7
    assert abs(np.cos(response.tangent_rad[40, 50] - vertical * np.pi / 2)) >= 0.9
    detector = FilterDetector(method)
    result = detector.run(dem, p)
    assert result == detector.run(dem, p)
    assert result.segments
    for line in result.segments:
        assert line.length_m > 40
        assert line.coordinates[0] == dem.transform * tuple(np.array(line.pixels[0]) + 0.5)
    with pytest.raises(CancelledError):
        detector.run(dem, p, cancel=lambda: True)
