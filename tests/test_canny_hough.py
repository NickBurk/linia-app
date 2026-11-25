import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.algorithms.canny_hough import (
    CannyHoughDetector,
    CannyHoughParameters,
    detect,
    pixel_size_metres,
)
from lineament_analyzer.core.windowed import CancelledError
from lineament_analyzer.data.dem import Dem


def make_dem(image, *, pixel_size=5, crs=32633, valid=None, transform=None):
    if valid is None:
        valid = np.ones(image.shape, dtype=bool)
    if transform is None:
        origin_x, origin_y = (10, 50) if crs == 4326 else (500000, 4500000)
        transform = Affine(pixel_size, 0, origin_x, 0, -pixel_size, origin_y)
    return Dem(
        image.astype(float),
        valid,
        transform,
        CRS.from_epsg(crs),
        None,
    )


def params(**changes):
    values = {
        "canny_low_threshold": 30,
        "canny_high_threshold": 90,
        "hough_threshold": 10,
        "min_segment_length_m": 300,
        "max_gap_m": 25,
    }
    values.update(changes)
    return CannyHoughParameters(**values)


def orientation_distance(first, second):
    return abs((first - second + 90) % 180 - 90)


@pytest.mark.parametrize(
    "start,end,expected",
    [((20, 40), (150, 40), 90), ((70, 20), (70, 150), 0), ((20, 20), (150, 150), 135)],
)
def test_detects_horizontal_vertical_and_diagonal_ridges(start, end, expected):
    image = np.zeros((180, 180), dtype=float)
    import cv2

    cv2.line(image, start, end, 100, 4)
    result = CannyHoughDetector().run(make_dem(image), params())
    assert result.segments
    assert min(orientation_distance(s.orientation_deg, expected) for s in result.segments) < 2
    assert all(s.length_m >= 300 for s in result.segments)
    assert all(s.algorithm_id == "canny-hough" for s in result.segments)
    assert all("edge_support_fraction" in s.properties for s in result.segments)
    assert all(s.confidence is not None and 0 <= s.confidence <= 1 for s in result.segments)


def test_crossing_and_parallel_structures_keep_both_orientations():
    import cv2

    image = np.zeros((180, 180), dtype=float)
    cv2.line(image, (15, 45), (165, 45), 100, 3)
    cv2.line(image, (15, 135), (165, 135), 100, 3)
    cv2.line(image, (90, 15), (90, 165), 100, 3)
    result = CannyHoughDetector().run(
        make_dem(image), params(hough_threshold=12, min_segment_length_m=400)
    )
    directions = [s.orientation_deg for s in result.segments]
    assert any(orientation_distance(value, 0) < 2 for value in directions)
    assert any(orientation_distance(value, 90) < 2 for value in directions)


def test_seeded_noise_does_not_hide_long_line():
    import cv2

    rng = np.random.default_rng(23)
    image = rng.normal(30, 3, (180, 180))
    cv2.line(image, (20, 90), (160, 90), 100, 4)
    result = CannyHoughDetector().run(
        make_dem(image),
        params(canny_low_threshold=45, canny_high_threshold=100, hough_threshold=25),
    )
    assert any(
        orientation_distance(s.orientation_deg, 90) < 2 and s.length_m >= 500
        for s in result.segments
    )


def test_nodata_gap_and_hole_boundaries_are_not_reported_as_supported_lines():
    import cv2

    image = np.zeros((150, 180), dtype=float)
    cv2.line(image, (10, 75), (170, 75), 100, 4)
    valid = np.ones_like(image, dtype=bool)
    valid[65:86, 80:100] = False
    result = CannyHoughDetector().run(
        make_dem(image, valid=valid), params(max_gap_m=100, min_segment_length_m=200)
    )
    for segment in result.segments:
        samples = np.linspace(
            [segment.pixel_start[0], segment.pixel_start[1]],
            [segment.pixel_end[0], segment.pixel_end[1]],
            300,
        )
        x, y = np.rint(samples).astype(int).T
        assert valid[y, x].all()
    assert all(
        not (
            min(s.pixel_start[0], s.pixel_end[0]) < 100
            and max(s.pixel_start[0], s.pixel_end[0]) > 80
        )
        for s in result.segments
    )


def test_maximum_gap_is_converted_from_metres_to_preview_pixels():
    import cv2

    image = np.zeros((160, 180), dtype=float)
    cv2.line(image, (10, 80), (70, 80), 100, 4)
    cv2.line(image, (78, 80), (170, 80), 100, 4)
    dem = make_dem(image)
    short_gap = CannyHoughDetector().run(
        dem, params(min_segment_length_m=500, max_gap_m=20)
    )
    long_gap = CannyHoughDetector().run(
        dem, params(min_segment_length_m=500, max_gap_m=40)
    )
    assert not short_gap.segments
    assert long_gap.segments
    assert max(segment.length_m for segment in long_gap.segments) > 750


def test_map_length_parameters_scale_across_dem_resolutions():
    import cv2

    detected_lengths = []
    for size, psize, start, end, thickness in [
        (200, 5, (20, 100), (180, 100), 6),
        (100, 10, (10, 50), (90, 50), 3),
    ]:
        image = np.zeros((size, size), dtype=float)
        cv2.line(image, start, end, 100, thickness)
        result = CannyHoughDetector().run(
            make_dem(image, pixel_size=psize),
            params(min_segment_length_m=500, max_gap_m=25),
        )
        assert result.segments
        detected_lengths.append(max(s.length_m for s in result.segments))
    assert detected_lengths[0] == pytest.approx(detected_lengths[1], rel=0.08)
    assert all(length >= 500 for length in detected_lengths)


def test_short_segments_are_filtered_using_metric_length():
    import cv2

    image = np.zeros((120, 180), dtype=float)
    cv2.line(image, (15, 35), (60, 35), 100, 4)
    cv2.line(image, (15, 85), (165, 85), 100, 4)
    result = CannyHoughDetector().run(make_dem(image), params(min_segment_length_m=300))
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


def test_rotated_and_geographic_pixel_scale_and_no_crs_error():
    rotated = make_dem(
        np.zeros((20, 20)),
        transform=Affine(8, 6, 500000, 6, -8, 4500000),
    )
    assert pixel_size_metres(rotated) == pytest.approx(10, rel=0.002)
    geographic = make_dem(np.zeros((20, 20)), pixel_size=0.0001, crs=4326)
    assert 5 < pixel_size_metres(geographic) < 15
    unknown = Dem(
        np.zeros((20, 20)),
        np.ones((20, 20), bool),
        Affine(5, 0, 0, 0, -5, 100),
        None,
        None,
    )
    with pytest.raises(ValueError, match="CRS is required"):
        pixel_size_metres(unknown)


@pytest.mark.parametrize(
    "values",
    [
        {"canny_low_threshold": -1},
        {"canny_low_threshold": 90, "canny_high_threshold": 80},
        {"canny_high_threshold": 256},
        {"aperture_size": 4},
        {"hough_threshold": 0},
        {"min_segment_length_m": 0},
        {"max_gap_m": -1},
    ],
)
def test_invalid_parameters_are_rejected(values):
    with pytest.raises(ValueError):
        params(**values).validate()


def test_empty_valid_mask_returns_empty_segments():
    result = CannyHoughDetector().run(
        make_dem(np.zeros((100, 100)), valid=np.zeros((100, 100), bool)), params()
    )
    assert result.segments == ()


def test_low_level_cancellation_is_checked_before_canny():
    image = np.zeros((20, 20))
    with pytest.raises(CancelledError, match="cancelled"):
        detect(
            image,
            np.ones_like(image, bool),
            params(),
            pixel_size_m=5,
            cancel=lambda: True,
        )
