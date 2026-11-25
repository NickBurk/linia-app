"""Spatial line comparison in a DEM-centred, metre-based local projection."""

from dataclasses import dataclass
from math import atan2, degrees

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from scipy.sparse import coo_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from shapely import LineString, STRtree, transform

from lineament_analyzer.core.windowed import check_cancel


@dataclass(frozen=True)
class MatchParameters:
    tolerance_m: float = 20.0
    orientation_deg: float = 20.0
    minimum_coverage: float = 0.55
    minimum_length_ratio: float = 0.45

    def validate(self):
        if not (
            np.isfinite(self.tolerance_m) and self.tolerance_m > 0
            and 0 <= self.orientation_deg <= 90
            and 0 < self.minimum_coverage <= 1
            and 0 < self.minimum_length_ratio <= 1
        ):
            raise ValueError("Invalid line matching tolerances")


class MetricGrid:
    """A local azimuthal equidistant frame with explicit extent limits."""

    def __init__(self, source_crs, affine, shape):
        if not source_crs:
            raise ValueError("A DEM CRS is required for metre-based geometry comparison")
        source = CRS.from_user_input(source_crs)
        geo = Transformer.from_crs(source, "EPSG:4326", always_xy=True)
        height, width = shape
        longitude, latitude = geo.transform(*(affine * (width / 2, height / 2)), errcheck=True)
        self.crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} +datum=WGS84 +units=m"
        )
        self.forward = Transformer.from_crs(source, self.crs, always_xy=True)
        self.inverse = Transformer.from_crs(self.crs, source, always_xy=True)
        corners = (affine * point for point in ((0, 0), (0, height), (width, height), (width, 0)))
        xy = self.forward.transform(*zip(*corners), errcheck=True)
        if np.hypot(*xy).max() > 500_000:
            raise ValueError("Clip the DEM to a radius of 500 km or less for local validation")

    @classmethod
    def from_result(cls, result):
        return cls(result.source_crs_wkt, Affine(*result.source_transform), result.source_shape)

    def project(self, geometry):
        return transform(geometry, self.forward.transform, interleaved=False)


def orientation(line):
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return degrees(atan2(x1 - x0, y1 - y0)) % 180


def axial_difference(first, second):
    return abs((first - second + 90) % 180 - 90)


@dataclass(frozen=True)
class Pair:
    detected: int
    reference: int
    detected_coverage: float
    reference_coverage: float
    orientation_error_deg: float
    length_ratio: float
    score: float


def candidate_pairs(detected, reference, p, cancel=None, *, same_collection=False):
    """Find nearby, direction-compatible candidates with directed coverage."""
    p.validate()
    tree = STRtree(reference)
    ref_angles = [orientation(g) for g in reference]
    ref_lengths = [g.length for g in reference]
    buffers = {}
    for i, line in enumerate(detected):
        check_cancel(cancel)
        if line.length <= 0:
            continue
        candidates = sorted(tree.query(line, predicate="dwithin", distance=p.tolerance_m))
        if not candidates:
            continue
        first_buffer = line.buffer(p.tolerance_m)
        first_angle = orientation(line)
        for j in candidates:
            if same_collection and j <= i:
                continue
            check_cancel(cancel)
            other = reference[j]
            angle = axial_difference(first_angle, ref_angles[j])
            if angle > p.orientation_deg or ref_lengths[j] <= 0:
                continue
            if j not in buffers:
                buffers[j] = other.buffer(p.tolerance_m)
            detected_coverage = min(1.0, line.intersection(buffers[j]).length / line.length)
            if detected_coverage < p.minimum_coverage:
                continue
            ref_coverage = min(1.0, other.intersection(first_buffer).length / ref_lengths[j])
            ratio = min(line.length, ref_lengths[j]) / max(line.length, ref_lengths[j])
            score = min(detected_coverage, ref_coverage) * ratio * (1 - angle / 180)
            yield Pair(i, int(j), detected_coverage, ref_coverage, angle, ratio, score)


def select_matching(pairs, n_detected, n_reference):
    """Maximum-cardinality, maximum-score one-to-one sparse assignment."""
    if not pairs:
        return ()
    rows = [pair.detected for pair in pairs] + list(range(n_detected))
    cols = [pair.reference for pair in pairs] + list(range(n_reference, n_reference + n_detected))
    costs = [2 - pair.score for pair in pairs] + [n_detected + 2.0] * n_detected
    shape = (n_detected, n_reference + n_detected)
    graph = coo_array((costs, (rows, cols)), shape=shape).tocsr()
    matched_rows, matched_cols = min_weight_full_bipartite_matching(graph)
    lookup = {(pair.detected, pair.reference): pair for pair in pairs}
    return tuple(lookup[i, j] for i, j in zip(matched_rows, matched_cols) if j < n_reference)


def project_result(result, grid):
    projected = [grid.project(LineString(line.coordinates)) for line in result.segments]
    if any(not line.is_valid or line.is_empty or not np.isfinite(line.length) for line in projected):
        raise ValueError("Invalid projected detection geometry")
    return projected
