"""Qt-independent lineament merging, filtering, classification and statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import atan2, degrees, hypot
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from lineament_analyzer.algorithms.results import (
    DetectionResult,
    LineSegment,
    build_detection_result,
)
from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.data.dem import Dem


@dataclass(frozen=True)
class PostprocessParameters:
    """Independent merge, filter and classification controls.

    Distance and length thresholds are in preview pixels. This keeps the
    operation usable for DEMs with no CRS and makes the GUI scale explicit.
    Metric segment lengths remain geodesic values in the result/export.
    """

    merge_enabled: bool = True
    max_angle_difference_deg: float = 12.0
    max_endpoint_distance_px: float = 15.0
    max_collinearity_distance_px: float = 4.0
    max_gap_px: float = 12.0
    preserve_crossings: bool = True
    minimum_length_px: float = 0.0
    maximum_length_px: float | None = None
    minimum_orientation_deg: float | None = None
    maximum_orientation_deg: float | None = None
    minimum_confidence: float | None = None
    duplicate_tolerance_px: float = 0.5
    direction_bin_count: int = 8
    short_length_px: float = 20.0
    long_length_px: float = 80.0
    density_radius_px: float = 50.0
    density_min_samples: int = 2

    def __post_init__(self) -> None:
        finite_nonnegative = (
            "max_angle_difference_deg",
            "max_endpoint_distance_px",
            "max_collinearity_distance_px",
            "max_gap_px",
            "minimum_length_px",
            "duplicate_tolerance_px",
            "short_length_px",
            "long_length_px",
            "density_radius_px",
        )
        for name in finite_nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_angle_difference_deg > 90:
            raise ValueError("max_angle_difference_deg must not exceed 90")
        if self.maximum_length_px is not None and (
            not np.isfinite(self.maximum_length_px)
            or self.maximum_length_px < self.minimum_length_px
        ):
            raise ValueError("maximum_length_px must be finite and >= minimum_length_px")
        if self.minimum_confidence is not None and (
            not np.isfinite(self.minimum_confidence)
            or not 0 <= self.minimum_confidence <= 1
        ):
            raise ValueError("minimum_confidence must be in [0, 1] or None")
        for name in ("minimum_orientation_deg", "maximum_orientation_deg"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 <= value < 180):
                raise ValueError(f"{name} must be in [0, 180) or None")
        if self.direction_bin_count < 1 or self.density_min_samples < 1:
            raise ValueError("Direction bins and density samples must be positive")
        if self.short_length_px > self.long_length_px:
            raise ValueError("short_length_px must not exceed long_length_px")


def _coords(segment: LineSegment) -> tuple[float, float, float, float]:
    return (*segment.pixel_start, *segment.pixel_end)


def _axial_angle(line: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = line
    return degrees(atan2(y2 - y1, x2 - x1)) % 180.0


def _pixel_bearing(line: tuple[float, float, float, float]) -> float:
    """Approximate clockwise-from-north axial bearing in image coordinates."""
    x1, y1, x2, y2 = line
    return degrees(atan2(x2 - x1, y1 - y2)) % 180.0


def _axial_difference(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _point_segment_distance(point, start, end) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    denominator = vx * vx + vy * vy
    if denominator == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    fraction = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / denominator))
    return hypot(point[0] - (start[0] + fraction * vx), point[1] - (start[1] + fraction * vy))


def _crosses_interior(first, second) -> bool:
    """Return true for a proper interior/interior intersection."""
    a = np.asarray(first[:2], dtype=float)
    b = np.asarray(first[2:], dtype=float)
    c = np.asarray(second[:2], dtype=float)
    d = np.asarray(second[2:], dtype=float)
    r, s = b - a, d - c
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(float(denominator)) < 1e-10:
        return False
    delta = c - a
    cross_delta_s = delta[0] * s[1] - delta[1] * s[0]
    cross_delta_r = delta[0] * r[1] - delta[1] * r[0]
    t = float(cross_delta_s / denominator)
    u = float(cross_delta_r / denominator)
    return 1e-8 < t < 1 - 1e-8 and 1e-8 < u < 1 - 1e-8


def _merge_pair(first, second, params: PostprocessParameters):
    angle1, angle2 = _axial_angle(first), _axial_angle(second)
    if _axial_difference(angle1, angle2) > params.max_angle_difference_deg:
        return None
    if params.preserve_crossings and _crosses_interior(first, second):
        return None
    endpoints1 = (first[:2], first[2:])
    endpoints2 = (second[:2], second[2:])
    choices = sorted(
        (hypot(a[0] - b[0], a[1] - b[1]), i, j)
        for i, a in enumerate(endpoints1)
        for j, b in enumerate(endpoints2)
    )
    gap, i, j = choices[0]
    if gap > params.max_endpoint_distance_px:
        return None
    a = endpoints1[i]
    b = endpoints2[j]
    direction = np.asarray([np.cos(np.deg2rad(angle1)), np.sin(np.deg2rad(angle1))])
    normal = np.asarray([-direction[1], direction[0]])
    if abs(float(np.dot(np.asarray(b) - np.asarray(a), direction))) > params.max_gap_px:
        return None
    offset = abs(float(np.dot(np.asarray(b) - np.asarray(a), normal)))
    if offset > params.max_collinearity_distance_px:
        return None
    candidates = [endpoints1[1 - i], a, b, endpoints2[1 - j]]
    projections = [float(np.dot(point, direction)) for point in candidates]
    low, high = int(np.argmin(projections)), int(np.argmax(projections))
    return (*candidates[low], *candidates[high])


def _make_result(
    result: DetectionResult,
    segments: tuple[LineSegment, ...],
    postprocessing: dict[str, Any],
) -> DetectionResult:
    return replace(
        result,
        segments=segments,
        postprocessing_parameters=MappingProxyType(dict(postprocessing)),
    )


def merge_segments(
    result: DetectionResult,
    parameters: PostprocessParameters,
    *,
    cancel=None,
) -> DetectionResult:
    """Greedily bridge near-collinear endpoints within independent limits."""
    if not parameters.merge_enabled or len(result.segments) < 2:
        return _make_result(result, result.segments, asdict(parameters))
    lines = [_coords(segment) for segment in result.segments]
    confidences = [segment.confidence for segment in result.segments]
    source_members = [[index] for index in range(len(lines))]
    changed = True
    while changed:
        check_cancel(cancel)
        changed = False
        endpoint_array = np.asarray([point for line in lines for point in (line[:2], line[2:])])
        check_cancel(cancel)
        endpoint_pairs = cKDTree(endpoint_array).query_pairs(
            parameters.max_endpoint_distance_px, output_type="ndarray"
        )
        check_cancel(cancel)
        candidate_indices = {
            tuple(sorted((int(first // 2), int(second // 2))))
            for first, second in endpoint_pairs
            if first // 2 != second // 2
        }
        pairs = []
        for pair_number, (first_index, second_index) in enumerate(sorted(candidate_indices)):
            if pair_number % 64 == 0:
                check_cancel(cancel)
            if any(
                result.segments[member].pixel_path
                for index in (first_index, second_index) for member in source_members[index]
            ):
                continue
            merged = _merge_pair(lines[first_index], lines[second_index], parameters)
            if merged is not None:
                endpoints_a = (lines[first_index][:2], lines[first_index][2:])
                endpoints_b = (lines[second_index][:2], lines[second_index][2:])
                gap = min(
                    hypot(a[0] - b[0], a[1] - b[1])
                    for a in endpoints_a
                    for b in endpoints_b
                )
                pairs.append((gap, first_index, second_index, merged))
        if pairs:
            scheduled = {}
            consumed = set()
            for _, first_index, second_index, merged in sorted(pairs, key=lambda item: item[0]):
                if first_index in consumed or second_index in consumed:
                    continue
                scheduled[first_index] = (second_index, merged)
                consumed.update((first_index, second_index))
            if scheduled:
                new_lines = []
                new_members = []
                new_confidences = []
                skipped = set()
                for index, line in enumerate(lines):
                    if index in skipped:
                        continue
                    if index in scheduled:
                        second_index, merged = scheduled[index]
                        members = source_members[index] + source_members[second_index]
                        known = [
                            result.segments[member].confidence
                            for member in members
                            if result.segments[member].confidence is not None
                        ]
                        new_lines.append(merged)
                        new_members.append(members)
                        new_confidences.append(float(np.mean(known)) if known else None)
                        skipped.add(second_index)
                    elif index not in consumed:
                        new_lines.append(line)
                        new_members.append(source_members[index])
                        new_confidences.append(confidences[index])
                lines, source_members, confidences = new_lines, new_members, new_confidences
                changed = True
    dem = _dem_from_result(result)
    # This reconstructs source coordinates from the exact source transform,
    # preserving the established pixel-centre convention and CRS metrics.
    built = build_detection_result(
        dem,
        lines,
        algorithm_id=result.algorithm_id,
        parameters=result.parameters,
        confidences=confidences,
        properties=[
            {
                **dict(result.segments[members[0]].properties),
                **({"edge_support_fraction": float(np.mean([
                    result.segments[index].confidence
                    for index in members
                    if result.segments[index].confidence is not None
                ]))} if any(result.segments[index].confidence is not None for index in members) else {}),
                "merged_from": members,
                "original_geometries": [result.segments[index].coordinates for index in members],
            }
            for members in source_members
        ],
    )
    return replace(
        result,
        segments=tuple(
            result.segments[members[0]] if result.segments[members[0]].pixel_path else segment
            for segment, members in zip(built.segments, source_members, strict=True)
        ),
        postprocessing_parameters=MappingProxyType(asdict(parameters)),
    )


def _dem_from_result(result: DetectionResult) -> Dem:
    """Create transform metadata needed by the shared result builder."""
    from affine import Affine
    from rasterio.crs import CRS

    crs = CRS.from_wkt(result.source_crs_wkt) if result.source_crs_wkt else None
    return Dem(
        values=np.broadcast_to(np.uint8(0), result.source_shape),
        valid=np.broadcast_to(True, result.source_shape),
        transform=Affine(*result.source_transform),
        crs=crs,
        path=None,
    )


def filter_segments(
    result: DetectionResult,
    parameters: PostprocessParameters,
    *,
    cancel=None,
) -> DetectionResult:
    """Filter by length, axial orientation, optional confidence and duplicates."""
    kept: list[LineSegment] = []
    seen_lines: list[LineSegment] = []
    confidence_threshold = parameters.minimum_confidence
    ordered = sorted(
        enumerate(result.segments),
        key=lambda item: (
            confidence_threshold is not None
            and item[1].confidence is not None
            and item[1].confidence < confidence_threshold,
            item[1].confidence is None,
            -(item[1].confidence if item[1].confidence is not None else 0.0),
            item[0],
        ),
    )
    for index, segment in ordered:
        if index % 64 == 0:
            check_cancel(cancel)
        if segment.length_px < parameters.minimum_length_px:
            continue
        if parameters.maximum_length_px is not None and segment.length_px > parameters.maximum_length_px:
            continue
        if (
            parameters.minimum_confidence is not None
            and segment.confidence is not None
            and segment.confidence < parameters.minimum_confidence
        ):
            continue
        orientation = segment.orientation_deg
        if orientation is None:
            orientation = _pixel_bearing(_coords(segment))
        if parameters.minimum_orientation_deg is not None and parameters.maximum_orientation_deg is not None:
            low, high = parameters.minimum_orientation_deg, parameters.maximum_orientation_deg
            inside = low <= orientation <= high if low <= high else orientation >= low or orientation <= high
            if not inside:
                continue
        elif (
            parameters.minimum_orientation_deg is not None
            and orientation < parameters.minimum_orientation_deg
        ) or (
            parameters.maximum_orientation_deg is not None
            and orientation > parameters.maximum_orientation_deg
        ):
            continue
        line = _coords(segment)
        def same_path(old, segment=segment):
            if not segment.pixel_path and not old.pixel_path:
                return True
            first, second = np.asarray(segment.pixels), np.asarray(old.pixels)
            return first.shape == second.shape and min(
                np.linalg.norm(first - second, axis=1).max(),
                np.linalg.norm(first - second[::-1], axis=1).max(),
            ) <= parameters.duplicate_tolerance_px

        duplicate = any(
            (
                hypot(line[0] - old[0], line[1] - old[1])
                <= parameters.duplicate_tolerance_px
                and hypot(line[2] - old[2], line[3] - old[3])
                <= parameters.duplicate_tolerance_px
            )
            or (
                hypot(line[0] - old[2], line[1] - old[3])
                <= parameters.duplicate_tolerance_px
                and hypot(line[2] - old[0], line[3] - old[1])
                <= parameters.duplicate_tolerance_px
            )
            for prior in seen_lines if same_path(prior)
            for old in (_coords(prior),)
        )
        if duplicate:
            continue
        seen_lines.append(segment)
        kept.append(segment)
    return _make_result(result, tuple(kept), asdict(parameters))


def classify_segments(
    result: DetectionResult,
    parameters: PostprocessParameters,
    *,
    cancel=None,
) -> DetectionResult:
    """Attach direction, length, shape and spatial density class attributes."""
    if not result.segments:
        return _make_result(result, (), asdict(parameters))
    centers = np.asarray(
        [
            ((segment.pixel_start[0] + segment.pixel_end[0]) / 2,
             (segment.pixel_start[1] + segment.pixel_end[1]) / 2)
            for segment in result.segments
        ],
        dtype=float,
    )
    check_cancel(cancel)
    tree = cKDTree(centers)
    neighbors = tree.query_ball_tree(tree, parameters.density_radius_px)
    check_cancel(cancel)
    labels = np.full(len(centers), -1, dtype=int)
    visited = np.zeros(len(centers), dtype=bool)
    cluster_id = 0
    for index in range(len(centers)):
        check_cancel(cancel)
        if visited[index]:
            continue
        visited[index] = True
        if len(neighbors[index]) < parameters.density_min_samples:
            continue
        labels[index] = cluster_id
        queue = list(neighbors[index])
        queued = set(queue)
        cursor = 0
        while cursor < len(queue):
            other = queue[cursor]
            cursor += 1
            if not visited[other]:
                visited[other] = True
                if len(neighbors[other]) >= parameters.density_min_samples:
                    for candidate in neighbors[other]:
                        if candidate not in queued:
                            queue.append(candidate)
                            queued.add(candidate)
            if labels[other] == -1:
                labels[other] = cluster_id
        cluster_id += 1

    result_segments = []
    width = 180.0 / parameters.direction_bin_count
    for index, segment in enumerate(result.segments):
        orientation = segment.orientation_deg
        if orientation is None:
            orientation = _pixel_bearing(_coords(segment))
        direction_bin = min(parameters.direction_bin_count - 1, int(orientation / width))
        length_class = (
            "short" if segment.length_px < parameters.short_length_px
            else "long" if segment.length_px >= parameters.long_length_px
            else "medium"
        )
        shape_class = "axial-vertical" if min(orientation, 180 - orientation) < 22.5 else (
            "axial-horizontal" if abs(orientation - 90) < 22.5 else "diagonal"
        )
        density_class = f"cluster-{labels[index]}" if labels[index] >= 0 else "isolated"
        props = dict(segment.properties)
        props.update(
            direction_class=f"direction-{direction_bin:02d}",
            length_class=length_class,
            geometry_class=shape_class,
            density_class=density_class,
        )
        props["class_id"] = "/".join((f"D{direction_bin:02d}", length_class, shape_class, density_class))
        result_segments.append(replace(segment, properties=MappingProxyType(props)))
    return _make_result(result, tuple(result_segments), asdict(parameters))


def postprocess_result(
    result: DetectionResult,
    parameters: PostprocessParameters,
    *,
    cancel=None,
) -> DetectionResult:
    """Run merge, filters and classification with cooperative cancellation."""
    check_cancel(cancel)
    merged = merge_segments(result, parameters, cancel=cancel)
    filtered = filter_segments(merged, parameters, cancel=cancel)
    return classify_segments(filtered, parameters, cancel=cancel)


def summarize_results(
    result: DetectionResult,
    *,
    direction_bin_count: int = 18,
    raster_area_km2: float | None = None,
) -> dict[str, Any]:
    """Return deterministic axial rose, length, direction and class statistics."""
    if direction_bin_count < 1:
        raise ValueError("direction_bin_count must be positive")
    orientations = [
        segment.orientation_deg
        if segment.orientation_deg is not None
        else _pixel_bearing(_coords(segment))
        for segment in result.segments
    ]
    bins = np.linspace(0.0, 180.0, direction_bin_count + 1)
    rose_counts, _ = np.histogram(orientations, bins=bins)
    lengths = [segment.length_m for segment in result.segments if segment.length_m is not None]
    length_hist, length_edges = np.histogram(lengths, bins=min(10, max(1, len(lengths)))) if lengths else (np.zeros(0, dtype=int), np.zeros(0))
    direction_stats = []
    for index, count in enumerate(rose_counts):
        start, end = bins[index], bins[index + 1]
        direction_stats.append({"start_deg": float(start), "end_deg": float(end), "count": int(count)})
    class_stats: dict[str, int] = {}
    for segment in result.segments:
        class_id = str(segment.properties.get("class_id", "unclassified"))
        class_stats[class_id] = class_stats.get(class_id, 0) + 1
    count = len(result.segments)
    lengths_total = float(sum(lengths)) if len(lengths) == count else None
    if raster_area_km2 is None:
        raster_area_km2 = _result_area_km2(result)
    density = count / raster_area_km2 if raster_area_km2 and raster_area_km2 > 0 else None
    return {
        "object_count": count,
        "known_metric_length_count": len(lengths),
        "total_length_m": lengths_total,
        "mean_length_m": float(np.mean(lengths)) if lengths else None,
        "median_length_m": float(np.median(lengths)) if lengths else None,
        "density_per_km2": float(density) if density is not None else None,
        "rose_orientation_range_deg": [0.0, 180.0],
        "rose_bin_edges_deg": bins.tolist(),
        "rose_counts": rose_counts.astype(int).tolist(),
        "direction_statistics": direction_stats,
        "length_distribution": {
            "bin_edges_m": length_edges.tolist(),
            "counts": length_hist.astype(int).tolist(),
        },
        "class_statistics": class_stats,
    }


def _result_area_km2(result: DetectionResult) -> float | None:
    """Estimate footprint area from affine corners and the CRS ellipsoid."""
    if not result.source_crs_wkt:
        return None
    try:
        from affine import Affine
        from pyproj import CRS, Geod, Transformer

        crs = CRS.from_wkt(result.source_crs_wkt)
        transform = Affine(*result.source_transform)
        height, width = result.source_shape
        corners = [
            transform * (0, 0),
            transform * (width, 0),
            transform * (width, height),
            transform * (0, height),
        ]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        longitude, latitude = transformer.transform(
            [point[0] for point in corners], [point[1] for point in corners]
        )
        if not np.isfinite(np.column_stack((longitude, latitude))).all():
            return None
        area, _ = Geod(ellps="WGS84").polygon_area_perimeter(longitude, latitude)
        area_km2 = abs(float(area)) / 1_000_000
        return area_km2 if area_km2 > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None
