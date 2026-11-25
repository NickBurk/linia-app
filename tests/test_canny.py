import cv2
import numpy as np
import pytest

from lineament_analyzer.algorithms.canny import CannyParameters, detect_edges
from lineament_analyzer.core.windowed import CancelledError


def test_canny_component_detects_edges_and_respects_effective_mask():
    image = np.zeros((100, 100), dtype=float)
    cv2.line(image, (10, 50), (90, 50), 100, 4)
    valid = np.ones_like(image, dtype=bool)
    valid[:, 60:65] = False
    edges, effective = detect_edges(image, valid, CannyParameters())
    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
    assert np.count_nonzero(edges) > 0
    np.testing.assert_array_equal(effective, valid)


def test_canny_constant_and_empty_inputs_return_no_edges():
    constant = np.ones((40, 40))
    edges, mask = detect_edges(constant, np.ones_like(constant, bool), CannyParameters())
    assert not np.any(edges)
    assert mask.all()
    empty, valid = detect_edges(constant, np.zeros_like(constant, bool), CannyParameters())
    assert not np.any(empty)
    assert not np.any(valid)


@pytest.mark.parametrize(
    "parameters",
    [
        CannyParameters(-1, 10),
        CannyParameters(10, 10),
        CannyParameters(10, 256),
        CannyParameters(10, 20, 4),
        CannyParameters(10, 20, l2_gradient=1),
    ],
)
def test_canny_parameters_are_validated(parameters):
    with pytest.raises((TypeError, ValueError)):
        parameters.validate()


def test_canny_checks_cancellation():
    image = np.zeros((10, 10))
    with pytest.raises(CancelledError):
        detect_edges(
            image,
            np.ones_like(image, bool),
            CannyParameters(),
            cancel=lambda: True,
        )
