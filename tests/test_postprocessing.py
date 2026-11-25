import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.algorithms.postprocessing import (
    PostprocessParameters,
    classify_segments,
    filter_segments,
    merge_segments,
    postprocess_result,
    summarize_results,
)
from lineament_analyzer.algorithms.results import build_detection_result
from lineament_analyzer.core.windowed import CancelledError
from lineament_analyzer.data.dem import Dem


def make_dem(crs=None, transform=None):
    crs = crs or CRS.from_epsg(32633)
    return Dem(
        np.zeros((100, 100), dtype=np.float32),
        np.ones((100, 100), dtype=bool),
        transform or Affine(10, 2, 500000, 1, -10, 4500000),
        crs,
        None,
    )


def result_for(lines, *, confidences=None, crs=None, transform=None):
    return build_detection_result(
        make_dem(crs, transform),
        lines,
        algorithm_id="synthetic-detector",
        parameters={"threshold": 12},
        confidences=confidences,
    )


def test_merge_uses_independent_angle_gap_distance_and_collinearity_limits():
    result = result_for([[10, 20, 30, 20], [32, 20, 50, 20]])
    params = PostprocessParameters(
        max_angle_difference_deg=2,
        max_endpoint_distance_px=4,
        max_gap_px=3,
        max_collinearity_distance_px=0.5,
    )
    merged = merge_segments(result, params)
    assert len(merged.segments) == 1
    np.testing.assert_allclose(merged.as_pixel_array(), [[10, 20, 50, 20]])
    assert merged.algorithm_id == result.algorithm_id
    assert merged.parameters == result.parameters
    assert merged.postprocessing_parameters["max_gap_px"] == 3

    too_far = merge_segments(
        result,
        PostprocessParameters(max_endpoint_distance_px=1, max_gap_px=20),
    )
    assert len(too_far.segments) == 2
    gap_limited = merge_segments(
        result,
        PostprocessParameters(max_endpoint_distance_px=10, max_gap_px=1),
    )
    assert len(gap_limited.segments) == 2


def test_intersections_and_non_collinear_lines_are_not_joined():
    result = result_for([[10, 20, 30, 20], [20, 10, 20, 30]])
    merged = merge_segments(
        result,
        PostprocessParameters(
            max_angle_difference_deg=5,
            max_endpoint_distance_px=100,
            max_gap_px=100,
            max_collinearity_distance_px=100,
        ),
    )
    assert len(merged.segments) == 2


def test_filter_deduplicates_reversed_lines_and_keeps_missing_confidence():
    result = result_for(
        [[10, 10, 30, 10], [30, 10, 10, 10], [5, 5, 7, 5], [10, 30, 30, 30], [10, 40, 30, 40]],
        confidences=[None, 0.9, 0.2, 0.4, None],
    )
    filtered = filter_segments(
        result,
        PostprocessParameters(
            minimum_length_px=5,
            maximum_length_px=25,
            minimum_confidence=0.5,
            duplicate_tolerance_px=0.1,
        ),
    )
    assert len(filtered.segments) == 2
    assert {segment.confidence for segment in filtered.segments} == {None, 0.9}
    assert filtered.segments[0].pixel_start == (30.0, 10.0)


def test_axial_rose_and_statistics_are_numerically_consistent():
    dem_crs = CRS.from_epsg(4326)
    transform = Affine(0.001, 0.0001, 10, 0.0001, -0.001, 50)
    result = result_for(
        [[20, 20, 20, 40], [20, 40, 20, 20], [10, 50, 40, 50]],
        crs=dem_crs,
        transform=transform,
    )
    # Opposite endpoint order remains the same axial orientation.
    assert result.segments[0].orientation_deg == pytest.approx(
        result.segments[1].orientation_deg, abs=0.01
    )
    classified = classify_segments(result, PostprocessParameters(density_radius_px=5))
    stats = summarize_results(classified, direction_bin_count=18)
    assert stats["object_count"] == 3
    assert stats["rose_orientation_range_deg"] == [0.0, 180.0]
    assert sum(stats["rose_counts"]) == 3
    assert stats["total_length_m"] == pytest.approx(
        sum(segment.length_m for segment in result.segments)
    )
    assert stats["density_per_km2"] is not None
    assert len(stats["class_statistics"]) >= 1
    assert all("class_id" in segment.properties for segment in classified.segments)


def test_postprocess_cancellation_and_invalid_parameters():
    result = result_for([[10, 10, 30, 10], [32, 10, 50, 10]])
    with pytest.raises(CancelledError):
        postprocess_result(result, PostprocessParameters(), cancel=lambda: True)
    with pytest.raises(ValueError, match="maximum_length_px"):
        PostprocessParameters(minimum_length_px=5, maximum_length_px=2)


@pytest.mark.parametrize("crs", [None, CRS.from_epsg(32633), CRS.from_epsg(4326)])
def test_postprocessing_preserves_rotated_georeferencing_and_confidence(crs):
    transform = Affine(0.001, 0.0004, 10, 0.0002, -0.001, 50) if crs and crs.is_geographic else Affine(10, 2, 500000, 1, -10, 4500000)
    result = result_for([[10, 10, 30, 10], [32, 10, 50, 10]], confidences=[0.8, None], crs=crs, transform=transform)
    processed = postprocess_result(
        result,
        PostprocessParameters(max_endpoint_distance_px=5, max_gap_px=5),
    )
    assert processed.source_transform == result.source_transform
    assert processed.source_shape == result.source_shape
    assert processed.source_crs_wkt == result.source_crs_wkt
    assert processed.segments[0].confidence == 0.8
