"""Algorithm agreement, one-to-one validation, and geometric diagnostics."""

from collections import defaultdict
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from pyproj import CRS
from shapely import LineString, union_all

from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.engine import METHODS
from lineament_analyzer.validation.geometry import (
    MatchParameters,
    MetricGrid,
    candidate_pairs,
    orientation,
    project_result,
    select_matching,
)

DEFAULT_MATCH = MatchParameters()

@dataclass(frozen=True)
class Evaluation:
    metrics: dict
    matches: tuple
    orientation_errors_deg: tuple
    detection_rose: tuple
    reference_rose: tuple
    detected_coordinates: tuple
    reference_coordinates: tuple


def _rose(lines):
    return tuple(map(int, np.histogram([orientation(line) for line in lines],
                                       bins=np.arange(0, 181, 15))[0]))


def _position_errors(line, reference, step):
    count = max(2, min(4096, int(np.ceil(line.length / step)) + 1))
    return [line.interpolate(t, normalized=True).distance(reference)
            for t in np.linspace(0, 1, count)]


def evaluate(result, reference=None, parameters=DEFAULT_MATCH, *, cancel=None):
    """Evaluate against reference source-CRS lines; omit accuracy without a reference."""
    parameters.validate()
    lengths = [line.length_m for line in result.segments if line.length_m is not None]
    metrics = {"count": len(result.segments),
               "mean_length_m": float(np.mean(lengths)) if lengths else None}
    source_detected = tuple(line.coordinates for line in result.segments)
    if not reference:
        return Evaluation(metrics, (), (), (), (), source_detected, ())
    grid = MetricGrid.from_result(result)
    found = project_result(result, grid)
    expected = [grid.project(line) for line in reference]
    pairs = tuple(candidate_pairs(found, expected, parameters, cancel))
    eligible = tuple(pair for pair in pairs if
                     pair.reference_coverage >= parameters.minimum_coverage
                     and pair.length_ratio >= parameters.minimum_length_ratio)
    matches = select_matching(eligible, len(found), len(expected))
    detected_count, reference_count, tp = len(found), len(expected), len(matches)
    precision = tp / detected_count if detected_count else 0.0
    recall = tp / reference_count
    orientation_errors, length_errors, position_errors = [], [], []
    for pair in matches:
        check_cancel(cancel)
        first, second = found[pair.detected], expected[pair.reference]
        orientation_errors.append(pair.orientation_error_deg)
        length_errors.append(abs(first.length - second.length))
        step = parameters.tolerance_m / 2
        position_errors.extend(_position_errors(first, second, step))
        position_errors.extend(_position_errors(second, first, step))
    matched_found = {pair.detected for pair in matches}
    duplicate_indices = {pair.detected for pair in eligible if pair.detected not in matched_found
                         and pair.reference in {match.reference for match in matches}}
    by_ref = defaultdict(set)
    for pair in pairs:
        if pair.detected not in duplicate_indices:
            by_ref[pair.reference].add(pair.detected)
    check_cancel(cancel)
    a = union_all([line.buffer(parameters.tolerance_m) for line in found])
    check_cancel(cancel)
    b = union_all([line.buffer(parameters.tolerance_m) for line in expected])
    check_cancel(cancel)
    total_area = a.union(b).area
    metrics.update(
        reference_count=reference_count, true_positive=tp,
        false_positive=detected_count - tp, false_negative=reference_count - tp,
        precision=precision, recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        buffered_iou=a.intersection(b).area / total_area if total_area else 0.0,
        mean_position_m=float(np.mean(position_errors)) if position_errors else None,
        median_position_m=float(np.median(position_errors)) if position_errors else None,
        mean_orientation_deg=float(np.mean(orientation_errors)) if orientation_errors else None,
        mean_length_error_m=float(np.mean(length_errors)) if length_errors else None,
        fragmented_references=sum(len(indices) > 1 for indices in by_ref.values()),
        duplicate_rate=len(duplicate_indices) / detected_count if detected_count else 0.0,
    )
    return Evaluation(metrics, matches, tuple(orientation_errors), _rose(found), _rose(expected),
                      source_detected, tuple(tuple(line.coords) for line in reference))


def compare_results(results, parameters=DEFAULT_MATCH, *, minimum_families=2, cancel=None):
    """Form complete-link groups, preserve one representative, and expose unmatched IDs."""
    if len(results) < 2 or minimum_families < 2:
        raise ValueError("Consensus requires at least two methods and two families")
    base = results[0]
    crs = CRS.from_user_input(base.source_crs_wkt)
    for result in results:
        if (
            CRS.from_user_input(result.source_crs_wkt) != crs
            or result.source_transform != base.source_transform
            or result.source_shape != base.source_shape
            or result.analysis_metadata.get("dem") != base.analysis_metadata.get("dem")
        ):
            raise ValueError("Consensus requires results on the same analysis grid")
    if len({r.algorithm_id for r in results}) != len(results):
        raise ValueError("Select one result per algorithm")
    grid = MetricGrid.from_result(base)
    records = [(result.algorithm_id, index, line)
               for result in results for index, line in enumerate(result.segments)]
    shapes = [grid.project(LineString(line.coordinates)) for _, _, line in records]
    pairs = {
        (pair.detected, pair.reference): pair.score
        for pair in candidate_pairs(shapes, shapes, parameters, cancel, same_collection=True)
        if records[pair.detected][0] != records[pair.reference][0]
        and pair.reference_coverage >= parameters.minimum_coverage
        and pair.length_ratio >= parameters.minimum_length_ratio
    }
    groups = {i: {i} for i in range(len(records))}
    owners = list(range(len(records)))
    for a, b in sorted(pairs, key=lambda key: (-pairs[key], key)):
        check_cancel(cancel)
        g1, g2 = owners[a], owners[b]
        if g1 == g2:
            continue
        first, second = groups[g1], groups[g2]
        if all(tuple(sorted((i, j))) in pairs for i in first for j in second):
            first.update(second)
            for i in second:
                owners[i] = g1
            del groups[g2]
    consensus, unique = [], defaultdict(list)
    for group in groups.values():
        check_cancel(cancel)
        algorithms = sorted({records[i][0] for i in group})
        families = sorted({METHODS[algorithm].family for algorithm in algorithms})
        members = sorted(group)
        if len(families) < minimum_families:
            for i in members:
                method, index, line = records[i]
                unique[method].append(line.properties.get("object_id", index))
            continue
        chosen = max(members, key=lambda i: (
            sum(pairs[tuple(sorted((i, j)))] for j in members if j != i), -i
        ))
        line = records[chosen][2]
        provenance = [{"algorithm": records[i][0],
                       "object_id": records[i][2].properties.get("object_id", records[i][1]),
                       "source_geometry": records[i][2].coordinates} for i in members]
        consensus.append(replace(
            line, algorithm_id="consensus", confidence=None,
            properties=MappingProxyType({"object_id": f"consensus-{len(consensus)+1:06d}",
                                         "supporting_algorithms": algorithms,
                                         "supporting_families": families,
                                         "support_count": len(algorithms),
                                         "provenance": provenance}),
        ))
    result = replace(base, algorithm_id="consensus", segments=tuple(consensus),
                     parameters=MappingProxyType({**vars(parameters),
                                                   "minimum_families": minimum_families}))
    return result, dict(unique)
