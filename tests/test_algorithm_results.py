import numpy as np
import pytest
from affine import Affine
from pyproj import Geod, Transformer
from rasterio.crs import CRS

from lineament_analyzer.algorithms.lsd import LsdDetector, LsdParameters
from lineament_analyzer.algorithms.lsd import detect as detect_lsd
from lineament_analyzer.algorithms.results import LineDetector, build_detection_result
from lineament_analyzer.core.windowed import CancelledError
from lineament_analyzer.data.dem import Dem


@pytest.mark.parametrize("epsg", [32633, 4326, 2277])
def test_result_uses_pixel_centres_affine_and_geodesic_metrics(epsg):
    transform = (
        Affine(0.0012, 0.0003, 10, 0.0002, -0.0008, 50)
        if epsg == 4326
        else Affine(12, 3, 500000, 2, -8, 4500000)
    )
    crs = CRS.from_epsg(epsg)
    dem = Dem(np.zeros((12, 15)), np.ones((12, 15), bool), transform, crs, None)
    pixels = np.array([[1.25, 2.5, 9.5, 7.25]])
    result = build_detection_result(
        dem,
        pixels,
        algorithm_id="test",
        parameters={"threshold": 5},
        confidences=[0.75],
        properties=[{"votes": 12}],
    )

    segment = result.segments[0]
    first = transform * (1.25 + 0.5, 2.5 + 0.5)
    second = transform * (9.5 + 0.5, 7.25 + 0.5)
    np.testing.assert_allclose(segment.source_start, first)
    np.testing.assert_allclose(segment.source_end, second)
    np.testing.assert_allclose(result.as_pixel_array(), pixels)
    assert segment.length_px == pytest.approx(np.hypot(8.25, 4.75))

    to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon1, lat1 = to_wgs84.transform(*first)
    lon2, lat2 = to_wgs84.transform(*second)
    azimuth, _, length = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    assert segment.length_m == pytest.approx(length)
    assert segment.azimuth_deg == pytest.approx(azimuth % 360)
    assert segment.orientation_deg == pytest.approx(azimuth % 180)
    assert segment.algorithm_id == result.algorithm_id == "test"
    assert result.parameters == {"threshold": 5}
    assert segment.confidence == 0.75
    assert segment.properties == {"votes": 12}
    assert result.source_crs_wkt == crs.to_wkt()
    with pytest.raises(TypeError):
        result.parameters["threshold"] = 6
    with pytest.raises(TypeError):
        segment.properties["votes"] = 20


def test_unknown_crs_keeps_source_coordinates_and_marks_metrics_unknown():
    transform = Affine(30, 0, 100, 0, -20, 500)
    dem = Dem(np.zeros((10, 10)), np.ones((10, 10), bool), transform, None, None)
    result = build_detection_result(dem, [[0, 0, 3, 4]], algorithm_id="test", parameters={})
    segment = result.segments[0]
    assert segment.source_start == (115, 490)
    assert segment.source_end == (205, 410)
    assert segment.length_m is None
    assert segment.azimuth_deg is None
    assert segment.orientation_deg is None
    assert result.source_crs_wkt is None


def test_empty_and_invalid_result_inputs():
    dem = Dem(
        np.zeros((5, 5)), np.ones((5, 5), bool), Affine.identity(), CRS.from_epsg(32633), None
    )
    empty = build_detection_result(dem, np.empty((0, 4)), algorithm_id="test", parameters={})
    assert empty.segments == ()
    assert empty.as_pixel_array().shape == (0, 4)

    for lines in (np.array([1, 2, 3, 4]), np.array([[0, 0, np.nan, 1]])):
        with pytest.raises(ValueError, match="Segments"):
            build_detection_result(dem, lines, algorithm_id="test", parameters={})
    with pytest.raises(ValueError, match="outside raster"):
        build_detection_result(dem, [[0, 0, 5, 4]], algorithm_id="test", parameters={})
    with pytest.raises(ValueError, match="Confidence count"):
        build_detection_result(
            dem, [[0, 0, 1, 1]], algorithm_id="test", parameters={}, confidences=[]
        )
    with pytest.raises(ValueError, match="Confidence values"):
        build_detection_result(
            dem, [[0, 0, 1, 1]], algorithm_id="test", parameters={}, confidences=[1.1]
        )


def test_invalid_geographic_coordinates_are_rejected():
    dem = Dem(
        np.zeros((5, 5)),
        np.ones((5, 5), bool),
        Affine(1, 0, 500000, 0, -1, 4500000),
        CRS.from_epsg(4326),
        None,
    )
    with pytest.raises(ValueError, match="invalid geographic coordinates"):
        build_detection_result(dem, [[0, 0, 1, 1]], algorithm_id="test", parameters={})


def test_lsd_adapter_implements_common_result_contract():
    image = np.zeros((100, 120), dtype=float)
    image[48:53, 10:110] = 255
    dem = Dem(
        image,
        np.ones_like(image, dtype=bool),
        Affine(10, 0, 500000, 0, -10, 4500000),
        CRS.from_epsg(32633),
        None,
    )
    parameters = LsdParameters(min_length=30)
    detector = LsdDetector()
    assert isinstance(detector, LineDetector)
    expected = detect_lsd(image, dem.valid, parameters)
    progress = []
    result = detector.run(dem, parameters, progress=progress.append)
    np.testing.assert_allclose(result.as_pixel_array(), expected)
    assert result.algorithm_id == "lsd"
    assert result.parameters == {"min_length": 30, "scale": 0.8}
    assert progress == [0, 100]
    assert all(segment.length_m and segment.confidence is None for segment in result.segments)
    with pytest.raises(CancelledError):
        detector.run(dem, parameters, cancel=lambda: True)
