from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from lineament_analyzer.algorithms.postprocessing import PostprocessParameters, postprocess_result
from lineament_analyzer.algorithms.ransac import RansacDetector, RansacParameters
from lineament_analyzer.algorithms.tracing import (
    AStarDetector,
    AStarParameters,
    SegmentTracingDetector,
    TraceParameters,
)
from lineament_analyzer.core.windowed import CancelledError
from lineament_analyzer.data.dem import Dem


@pytest.fixture
def ridge():
    y, x = np.indices((100, 120))
    values = 20.0 * np.exp(-((y - 50) / 2) ** 2)
    valid = ~((x > 55) & (x < 63) & (y > 40) & (y < 60))
    return Dem(values, valid, Affine(3, 0.2, 500000, 0.1, -4, 5000000),
               CRS.from_epsg(32633), Path("fixture.tif"), "m")


@pytest.mark.parametrize(("detector", "parameters"), [
    (AStarDetector(), AStarParameters(max_steps=60)),
    (SegmentTracingDetector(), TraceParameters(max_steps=60)),
    (RansacDetector(), RansacParameters()),
])
def test_ridge_coordinates_mask_and_reproducibility(ridge, detector, parameters):
    result = detector.run(ridge, parameters)
    assert result.segments
    assert result == detector.run(ridge, parameters)
    for line in result.segments:
        assert line.length_m > 20
        assert line.orientation_deg == pytest.approx(90, abs=12)
        assert line.coordinates[0] == ridge.transform * tuple(np.array(line.pixels[0]) + 0.5)
        x, y = np.rint(np.asarray(line.pixels)).astype(int).T
        assert ridge.valid[y, x].all()
        assert not (x.min() < 55 and x.max() > 63)
    with pytest.raises(CancelledError):
        detector.run(ridge, parameters, cancel=lambda: True)


def test_astar_interactive_requires_supported_path(ridge):
    detector = AStarDetector()
    parameters = AStarParameters()
    result = detector.run(ridge, parameters, endpoints=((4, 48), (42, 48)))
    assert len(result.segments) == 1
    line = result.segments[0]
    assert line.length_px == pytest.approx(38)
    assert line.properties["search_cost"] >= line.length_px
    assert not detector.run(ridge, parameters, endpoints=((4, 48), (100, 48))).segments
    assert not detector.run(ridge, parameters, endpoints=((-1000, 48), (12, 48))).segments
    geographic = replace(ridge, crs=CRS.from_epsg(4326),
                         transform=Affine(0.0001, 0, 10, 0, -0.0001, 60))
    assert detector.run(geographic, parameters, endpoints=((4, 48), (42, 48))).segments[0].length_m > 200


def test_postprocessing_keeps_actual_path(ridge):
    result = SegmentTracingDetector().run(ridge, TraceParameters(max_steps=48))
    processed = postprocess_result(result, PostprocessParameters())
    assert processed.segments
    assert all(line.pixel_path for line in processed.segments)


def test_trace_join_respects_mask_and_tip_alignment():
    from lineament_analyzer.algorithms.tracing import _join_fragments

    response = np.ones((12, 24))
    valid = response.astype(bool)
    a = [(x, 6) for x in range(3, 10)]
    b = [(x, 6) for x in range(12, 20)]
    p = TraceParameters()
    field = response, np.zeros_like(response), valid
    paths = _join_fragments([a, b], field, p, None)
    assert paths == [[(x, 6) for x in range(3, 20)]]
    valid[6, 10] = False
    assert _join_fragments([a, b], field, p, None) == [a, b]
