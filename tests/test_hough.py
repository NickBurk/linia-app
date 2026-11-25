import cv2
import numpy as np
import pytest

from lineament_analyzer.algorithms.hough import HoughParameters, detect_lines
from lineament_analyzer.core.windowed import CancelledError


def test_probabilistic_hough_returns_line_candidates():
    edges = np.zeros((120, 140), dtype=np.uint8)
    cv2.line(edges, (10, 60), (130, 60), 255, 1)
    lines = detect_lines(
        edges,
        HoughParameters(vote_threshold=10, min_segment_length_m=400, max_gap_m=0),
        pixel_size_m=5,
    )
    assert lines.ndim == 2 and lines.shape[1] == 4
    assert len(lines) > 0
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    assert np.all(lengths >= 80)
    assert np.all(np.abs(lines[:, 3] - lines[:, 1]) <= 1)


def test_probabilistic_hough_returns_empty_for_no_edges():
    lines = detect_lines(np.zeros((20, 20), dtype=np.uint8), HoughParameters(), pixel_size_m=1)
    assert lines.shape == (0, 4)


@pytest.mark.parametrize(
    "parameters",
    [
        HoughParameters(vote_threshold=0),
        HoughParameters(min_segment_length_m=0),
        HoughParameters(max_gap_m=-1),
    ],
)
def test_hough_parameters_are_validated(parameters):
    with pytest.raises(ValueError):
        parameters.validate()


def test_hough_rejects_invalid_edge_image_and_checks_cancellation():
    with pytest.raises(ValueError, match="uint8"):
        detect_lines(np.zeros((20, 20), dtype=float), HoughParameters(), pixel_size_m=1)
    with pytest.raises(CancelledError):
        detect_lines(
            np.zeros((20, 20), dtype=np.uint8),
            HoughParameters(),
            pixel_size_m=1,
            cancel=lambda: True,
        )
