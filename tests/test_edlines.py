import cv2
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.algorithms.edlines import (
    EDGE_OPERATORS,
    EdLinesDetector,
    EdLinesParameters,
    detect,
)
from lineament_analyzer.core.windowed import CancelledError
from lineament_analyzer.data.dem import Dem


def make_dem(image, *, pixel_size=5, valid=None):
    if valid is None:
        valid = np.ones(image.shape, dtype=bool)
    return Dem(
        image.astype(float),
        valid,
        Affine(pixel_size, 0, 500000, 0, -pixel_size, 4500000),
        CRS.from_epsg(32633),
        None,
    )


def params(**changes):
    values = {
        "gradient_threshold": 15,
        "min_path_length_px": 8,
        "min_segment_length_m": 300,
        "nfa_validation": False,
    }
    values.update(changes)
    return EdLinesParameters(**values)


def orientation_distance(first, second):
    return abs((first - second + 90) % 180 - 90)


@pytest.mark.parametrize(
    "start,end,expected",
    [((20, 40), (150, 40), 90), ((70, 20), (70, 150), 0), ((20, 20), (150, 150), 135)],
)
def test_actual_edlines_backend_detects_cardinal_and_diagonal_lines(start, end, expected):
    image = np.zeros((180, 180), dtype=float)
    cv2.line(image, start, end, 100, 4)
    result = EdLinesDetector().run(make_dem(image), params())
    assert result.segments
    assert min(orientation_distance(s.orientation_deg, expected) for s in result.segments) < 3
    assert all(s.algorithm_id == "edlines" for s in result.segments)
    assert all(s.length_m >= 300 for s in result.segments)
    assert all(s.confidence is not None and 0 <= s.confidence <= 1 for s in result.segments)
    assert all("edge_support_fraction" in s.properties for s in result.segments)


def test_crossing_and_parallel_lines():
    image = np.zeros((180, 180), dtype=float)
    cv2.line(image, (15, 45), (165, 45), 100, 3)
    cv2.line(image, (15, 135), (165, 135), 100, 3)
    cv2.line(image, (90, 15), (90, 165), 100, 3)
    result = EdLinesDetector().run(
        make_dem(image), params(min_segment_length_m=300, gradient_threshold=12)
    )
    directions = [segment.orientation_deg for segment in result.segments]
    assert any(orientation_distance(value, 0) < 3 for value in directions)
    assert any(orientation_distance(value, 90) < 3 for value in directions)


def test_seeded_noise_preserves_long_line():
    rng = np.random.default_rng(45)
    image = rng.normal(30, 2, (180, 180))
    cv2.line(image, (20, 90), (160, 90), 100, 4)
    result = EdLinesDetector().run(
        make_dem(image), params(gradient_threshold=20, min_segment_length_m=500)
    )
    assert any(
        orientation_distance(segment.orientation_deg, 90) < 3 and segment.length_m >= 500
        for segment in result.segments
    )


def test_nodata_support_rejects_segments_touching_gap():
    image = np.zeros((150, 180), dtype=float)
    cv2.line(image, (10, 75), (170, 75), 100, 4)
    valid = np.ones_like(image, dtype=bool)
    valid[65:86, 80:100] = False
    result = EdLinesDetector().run(
        make_dem(image, valid=valid),
        params(min_segment_length_m=200, max_distance_between_lines_px=100),
    )
    assert all(
        not (
            min(segment.pixel_start[0], segment.pixel_end[0]) < 100
            and max(segment.pixel_start[0], segment.pixel_end[0]) > 80
        )
        for segment in result.segments
    )


def test_length_units_scale_across_dem_resolutions():
    detected_lengths = []
    for size, pixel, start, end, thickness in [
        (200, 5, (20, 100), (180, 100), 6),
        (100, 10, (10, 50), (90, 50), 3),
    ]:
        image = np.zeros((size, size), dtype=float)
        cv2.line(image, start, end, 100, thickness)
        result = EdLinesDetector().run(
            make_dem(image, pixel_size=pixel),
            params(min_segment_length_m=500),
        )
        assert result.segments
        detected_lengths.append(max(segment.length_m for segment in result.segments))
    assert detected_lengths[0] == pytest.approx(detected_lengths[1], rel=0.08)


def test_short_segments_are_filtered_using_metric_length():
    image = np.zeros((120, 180), dtype=float)
    cv2.line(image, (15, 35), (60, 35), 100, 4)
    cv2.line(image, (15, 85), (165, 85), 100, 4)
    result = EdLinesDetector().run(make_dem(image), params(min_segment_length_m=300))
    assert result.segments
    assert all(segment.length_m >= 300 for segment in result.segments)
    assert all(
        not (
            min(segment.pixel_start[0], segment.pixel_end[0]) <= 60
            and max(segment.pixel_start[0], segment.pixel_end[0]) >= 15
            and abs(segment.pixel_start[1] - 35) < 10
            and abs(segment.pixel_end[1] - 35) < 10
        )
        for segment in result.segments
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"sigma": 0},
        {"gradient_threshold": 0},
        {"anchor_threshold": -1},
        {"min_path_length_px": 0},
        {"min_segment_length_m": 0},
        {"line_fit_error_threshold": 0},
        {"max_error_threshold": 0},
        {"max_distance_between_lines_px": -1},
        {"edge_operator": "LSD"},
        {"scan_interval": 0},
        {"pf_mode": 1},
    ],
)
def test_independent_parameters_are_validated(changes):
    with pytest.raises((TypeError, ValueError)):
        params(**changes).validate()


def test_parameter_values_are_passed_to_edlines_backend(monkeypatch):
    captured = {}
    parameter_factory = cv2.ximgproc.createEdgeDrawing().Params

    class FakeBackend:
        def Params(self):
            return parameter_factory()

        def setParams(self, config):
            captured.update(
                sigma=config.Sigma,
                minimum_length=config.MinLineLength,
                gradient_threshold=config.GradientThresholdValue,
                operator=config.EdgeDetectionOperator,
            )

        def detectEdges(self, raster):
            captured["input_dtype"] = raster.dtype

        def detectLines(self):
            return np.array([[[20, 20, 100, 20]]], dtype=np.float32)

        def getEdgeImage(self):
            edge_image = np.zeros((120, 120), dtype=np.uint8)
            edge_image[20, 20:101] = 255
            return edge_image

    monkeypatch.setattr(cv2.ximgproc, "createEdgeDrawing", lambda: FakeBackend())
    image = np.zeros((120, 120), dtype=float)
    image[18:23, 20:101] = 100
    result = detect(
        image,
        np.ones_like(image, dtype=bool),
        params(sigma=1.4, gradient_threshold=17, edge_operator="PREWITT"),
        pixel_size_m=5,
    )
    assert result.shape == (1, 4)
    assert captured["sigma"] == pytest.approx(1.4)
    assert captured["minimum_length"] == 60
    assert captured["gradient_threshold"] == 17
    assert captured["operator"] == EDGE_OPERATORS["PREWITT"]
    assert captured["input_dtype"] == np.uint8


def test_backend_absence_fails_clearly_without_fallback(monkeypatch):
    monkeypatch.setattr(cv2, "ximgproc", None)
    with pytest.raises(RuntimeError, match="requires OpenCV contrib"):
        detect(
            np.pad(np.ones((10, 10)), 5),
            np.ones((20, 20), bool),
            params(),
            pixel_size_m=5,
        )


def test_no_data_empty_result_and_cancellation():
    empty = EdLinesDetector().run(
        make_dem(np.zeros((30, 30)), valid=np.zeros((30, 30), bool)), params()
    )
    assert empty.segments == ()
    with pytest.raises(CancelledError, match="cancelled"):
        detect(
            np.zeros((20, 20)),
            np.ones((20, 20), bool),
            params(),
            pixel_size_m=5,
            cancel=lambda: True,
        )
