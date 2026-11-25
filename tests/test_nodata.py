import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.core.terrain import derivative
from lineament_analyzer.data.dem import Dem


def test_tpi_complete_support_and_no_edge_extension():
    y, x = np.indices((21, 21))
    z = (2 * x + y).astype(float)
    valid = np.ones_like(z, bool)
    valid[10, 10] = False
    z[10, 10] = -99999
    dem = Dem(z, valid, Affine(1, 0, 0, 0, -1, 21), CRS.from_epsg(32633), None, "m")
    tpi = derivative(dem, "tpi", tpi_size=5)
    assert np.isnan(tpi[:2]).all()
    assert np.isnan(tpi[-2:]).all()
    assert np.isnan(tpi[8:13, 8:13]).all()
    np.testing.assert_allclose(tpi[np.isfinite(tpi)], 0, atol=1e-12)


@pytest.mark.parametrize("kind", ["hillshade", "slope", "aspect"])
def test_hole_and_true_edges(kind):
    y, x = np.indices((13, 13))
    z = (2 * x + y).astype(float)
    valid = np.ones_like(z, bool)
    valid[6, 6] = False
    z[2, 2] = np.nan  # Non-finite samples must also invalidate their stencils.
    dem = Dem(z, valid, Affine(1, 0, 0, 0, -1, 13), CRS.from_epsg(32633), None, "m")
    result = derivative(dem, kind)
    assert np.isnan(result[6, 5:8]).all()
    assert np.isnan(result[5:8, 6]).all()
    assert np.isnan(result[2, 1:4]).all()
    finite = result[np.isfinite(result)]
    np.testing.assert_allclose(finite, result[0, 0])
