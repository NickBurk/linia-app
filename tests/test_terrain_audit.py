import numpy as np
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.core.terrain import derivative
from lineament_analyzer.data.dem import Dem


def test_plane_and_nodata():
    y, x = np.mgrid[:9, :9]
    values = (2 * x + y).astype(float)
    valid = np.ones(values.shape, dtype=bool)
    valid[4, 4] = False
    dem = Dem(values, valid, Affine(1, 0, 0, 0, -1, 9), CRS.from_epsg(32633), None)
    slope = derivative(dem, "slope")
    assert np.isclose(slope[0, 0], np.degrees(np.arctan(np.sqrt(5))))
    assert np.isnan(slope[4, 5])
    assert np.isnan(slope[4, 4])
    assert np.isclose(slope[2, 2], slope[0, 0])


def test_flat_aspect_undefined():
    values = np.ones((5, 5))
    dem = Dem(values, np.ones_like(values, dtype=bool), Affine.identity(), None, None)
    assert np.isnan(derivative(dem, "aspect")).all()
