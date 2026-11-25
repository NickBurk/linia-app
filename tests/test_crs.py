from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS as ProjCRS
from rasterio.crs import CRS

from lineament_analyzer.core.terrain import UnitWarning, derivative
from lineament_analyzer.data.dem import Dem


def plane(transform, crs, unit="m", factor=1):
    row, col = np.indices((17, 19))
    east = transform.a * col + transform.b * row
    north = transform.d * col + transform.e * row
    values = (0.2 * east + 0.1 * north) / factor
    return Dem(
        values, np.ones_like(values, dtype=bool), transform, CRS.from_epsg(crs), Path("plane"), unit
    )


@pytest.mark.parametrize(
    "transform", [Affine(10, 0, 0, 0, -20, 1000), Affine(8, 6, 0, 6, -8, 1000)]
)
def test_rotated_projected_plane(transform):
    dem = plane(transform, 32633)
    np.testing.assert_allclose(
        dem.resolution, (np.hypot(transform.a, transform.d), np.hypot(transform.b, transform.e))
    )
    slope = np.arctan(np.hypot(0.2, 0.1))
    aspect = np.arctan2(-0.2, -0.1)
    np.testing.assert_allclose(derivative(dem, "slope"), np.degrees(slope))
    np.testing.assert_allclose(derivative(dem, "aspect"), np.degrees(aspect) % 360)
    expected = 255 * (
        np.cos(np.pi / 4) * np.cos(slope)
        + np.sin(np.pi / 4) * np.sin(slope) * np.cos(np.radians(315) - aspect)
    )
    np.testing.assert_allclose(derivative(dem, "hillshade"), expected)


def test_horizontal_and_vertical_feet_and_z_factor():
    unit_factor = ProjCRS.from_epsg(2277).axis_info[0].unit_conversion_factor
    dem = plane(Affine(10, 0, 0, 0, -10, 1000), 2277, "ft", 0.3048 / unit_factor)
    np.testing.assert_allclose(
        derivative(dem, "slope", vertical_scale=2), np.degrees(np.arctan(np.hypot(0.4, 0.2)))
    )


@pytest.mark.parametrize("latitude", [0, 45, 75])
def test_geographic_latitude_scale(latitude):
    t = Affine(0.0001, 0, 10, 0, -0.0001, latitude + 0.001)
    row, col = np.indices((17, 19))
    phi = np.radians(t.f + t.e * (row + 0.5))
    a = 6378137
    e2 = 6.6943799901413165e-3
    metres_per_degree = np.pi / 180 * a / np.sqrt(1 - e2 * np.sin(phi) ** 2) * np.cos(phi)
    z = 0.2 * metres_per_degree * t.a * (col - 9)
    dem = Dem(z, np.ones_like(z, dtype=bool), t, CRS.from_epsg(4326), None, "m")
    np.testing.assert_allclose(derivative(dem, "slope")[:, 9], np.degrees(np.arctan(0.2)))
    np.testing.assert_allclose(derivative(dem, "aspect")[:, 9], 270)


def test_unknown_units_warn():
    dem = Dem(np.ones((5, 5)), np.ones((5, 5), bool), Affine.identity(), None, None)
    with pytest.warns(UnitWarning) as caught:
        derivative(dem, "slope")
    assert len(caught) == 2


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_bad_z_factor(scale):
    with pytest.raises(ValueError, match="Z-factor"):
        derivative(plane(Affine.identity(), 32633), "slope", vertical_scale=scale)


@pytest.mark.parametrize(
    "gradient,aspect", [((0, -1), 0), ((-1, 0), 90), ((0, 1), 180), ((1, 0), 270)]
)
def test_cardinal_aspect(gradient, aspect):
    y, x = np.indices((7, 7))
    z = gradient[0] * x - gradient[1] * y
    dem = Dem(z, np.ones_like(z, bool), Affine(1, 0, 0, 0, -1, 7), CRS.from_epsg(32633), None, "m")
    np.testing.assert_allclose(derivative(dem, "aspect"), aspect)
    np.testing.assert_allclose(derivative(dem, "slope"), 45)


def test_explicit_vertical_unit_override():
    dem = plane(Affine.identity(), 32633, "unrecognized")
    np.testing.assert_allclose(
        derivative(dem, "slope", elevation_unit="ft"),
        np.degrees(np.arctan(np.hypot(0.2, 0.1) * 0.3048)),
    )
