"""Data-constrained grid paths and bidirectional segment tracing."""

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import cos, hypot, pi

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter

from lineament_analyzer.algorithms.results import build_path_result
from lineament_analyzer.algorithms.support import detector_support_mask
from lineament_analyzer.core.windowed import check_cancel

STEPS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


@dataclass(frozen=True)
class TraceParameters:
    min_length_px: int = 12
    max_steps: int = 120
    max_turn_deg: float = 45
    support_threshold: float = 0.2
    max_gap_px: int = 2
    max_paths: int = 32


@dataclass(frozen=True)
class AStarParameters(TraceParameters):
    turn_weight: float = 0.8
    orientation_weight: float = 1.2
    max_expansions: int = 12000


def _validate(p):
    if (
        not 2 <= p.min_length_px <= p.max_steps <= 4096
        or not 0 <= p.max_turn_deg <= 90
        or not 0 < p.support_threshold <= 1
        or not 0 <= p.max_gap_px <= 16 or not 1 <= p.max_paths <= 512
    ):
        raise ValueError("Invalid tracing limits")
    if isinstance(p, AStarParameters) and (
        not 0 <= p.turn_weight <= 100 or not 0 <= p.orientation_weight <= 100
        or not 1 <= p.max_expansions <= 1_000_000
    ):
        raise ValueError("Invalid A* search weights or expansion limit")


def gradient_field(dem):
    """Estimate edge strength and tangent angle on the pixel grid."""
    valid = dem.valid & np.isfinite(dem.values)
    strength = np.zeros(dem.values.shape)
    if min(valid.shape) < 2 or not valid.any():
        return strength, strength.copy(), np.zeros_like(valid)
    if valid.all():
        image = dem.values
    else:
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        image = dem.values[tuple(nearest)]
    gy, gx = np.gradient(gaussian_filter(image, 1.0))
    safe = detector_support_mask(valid, radius=4)
    strength = np.where(safe, np.hypot(gx, gy), 0)
    positive = strength[strength > np.finfo(float).eps]
    if positive.size:
        strength = np.minimum(strength / np.percentile(positive, 95), 1)
    return strength, np.arctan2(gx, -gy), safe


def _neighbors(state, strength, tangent, valid, p):
    y, x, heading, gap = state
    for direction, (dy, dx) in enumerate(STEPS):
        ny, nx = y + dy, x + dx
        turn = min((direction - heading) % 8, (heading - direction) % 8) if heading >= 0 else 0
        if turn * 45 > p.max_turn_deg or not (
            0 <= ny < valid.shape[0] and 0 <= nx < valid.shape[1] and valid[ny, nx]
        ):
            continue
        if dy and dx and not (valid[y, nx] and valid[ny, x]):
            continue
        value = strength[ny, nx]
        new_gap = gap + 1 if value < p.support_threshold else 0
        if new_gap <= p.max_gap_px:
            alignment = abs(cos(direction * pi / 4 - tangent[ny, nx]))
            yield (ny, nx, direction, new_gap), value, alignment, turn, hypot(dx, dy)


def _search(start, end, field, p, cancel):
    strength, tangent, valid = field
    if not all(
        0 <= y < valid.shape[0] and 0 <= x < valid.shape[1]
        and valid[y, x] and strength[y, x] >= p.support_threshold
        for y, x in (start, end)
    ):
        return None
    initial = (*start, -1, 0, 0)
    costs = {initial: 0.0}
    previous = {}
    queue = [(hypot(end[0] - start[0], end[1] - start[1]), 0.0, initial)]
    expansions = 0
    while queue and expansions < p.max_expansions:
        _, cost, state = heappop(queue)
        if cost != costs[state]:
            continue
        if expansions % 128 == 0:
            check_cancel(cancel)
        expansions += 1
        y, x, _, _, steps = state
        if (y, x) == end:
            path = [(x, y)]
            while state in previous:
                state = previous[state]
                path.append((state[1], state[0]))
            if len(set(path)) != len(path):
                return None
            return path[::-1], cost
        if steps >= p.max_steps:
            continue
        for neighbor, value, alignment, turn, distance in _neighbors(
            state[:4], strength, tangent, valid, p
        ):
            ny, nx = neighbor[:2]
            estimate = hypot(end[0] - ny, end[1] - nx)
            if max(abs(end[0] - ny), abs(end[1] - nx)) + steps >= p.max_steps:
                continue
            weight = 1 + 2 * (1 - value) + p.orientation_weight * (1 - alignment)
            new_cost = cost + distance * weight + p.turn_weight * turn**2
            target = (*neighbor, steps + 1)
            if new_cost < costs.get(target, float("inf")):
                costs[target] = new_cost
                previous[target] = state
                heappush(queue, (new_cost + estimate, new_cost, target))
    return None


def _trace(seed, field, p, occupied, cancel):
    strength, tangent, valid = field
    direction = round(tangent[seed] * 4 / pi) % 8
    seen = {seed}
    branches = []
    for offset in (0, 4):
        state = (*seed, (direction + offset) % 8, 0)
        branch = []
        supported = 0
        for step in range(p.max_steps // 2):
            if step % 64 == 0:
                check_cancel(cancel)
            choices = [
                (value + alignment - 0.2 * turn, neighbor)
                for neighbor, value, alignment, turn, _ in _neighbors(
                    state, strength, tangent, valid, p
                )
                if neighbor[:2] not in seen and not occupied[neighbor[:2]]
            ]
            if not choices:
                break
            _, state = max(choices)
            y, x, _, gap = state
            seen.add((y, x))
            branch.append((x, y))
            if gap == 0:
                supported = len(branch)
        branches.append(branch[:supported])
    return branches[0][::-1] + [(seed[1], seed[0])] + branches[1]


def _join_fragments(paths, field, p, cancel):
    """Join aligned tips over a short valid gap without introducing a loop."""
    strength, _, valid = field
    joined = list(paths)
    for i, first in enumerate(joined):
        check_cancel(cancel)
        if first is None:
            continue
        for j in range(i + 1, len(joined)):
            second = joined[j]
            if second is None or len(first) + len(second) > p.max_steps + 1:
                continue
            for a, b in ((first, second), (first, second[::-1]),
                         (first[::-1], second), (first[::-1], second[::-1])):
                tip, origin = np.asarray(a[-1]), np.asarray(b[0])
                gap = origin - tip
                distance = np.linalg.norm(gap)
                if distance == 0 or distance > p.max_gap_px + 1:
                    continue
                directions = (tip - a[-min(4, len(a))], np.asarray(b[min(3, len(b)-1)]) - origin)
                minimum_alignment = cos(p.max_turn_deg * pi / 180)
                if any(direction @ gap < minimum_alignment * np.linalg.norm(direction) * distance
                       for direction in directions):
                    continue
                steps = int(max(abs(gap)))
                bridge = [tuple(point) for point in np.rint(
                    np.linspace(tip, origin, steps + 1)[1:-1]
                ).astype(int)]
                if not all(valid[y, x] for x, y in bridge):
                    continue
                combined = a + bridge + b
                if len(combined) > p.max_steps + 1 or len(set(combined)) != len(combined):
                    continue
                if np.mean([strength[y, x] >= p.support_threshold for x, y in combined]) < 0.65:
                    continue
                first = joined[i] = combined
                joined[j] = None
                break
    return [path for path in joined if path is not None]


class SegmentTracingDetector:
    """Greedy tangent-following with bidirectional seeds and bounded gap bridging."""

    algorithm_id = "segment-tracing"

    def run(self, dem, parameters, *, progress=None, cancel=None):
        return _run(dem, parameters, self.algorithm_id, progress, cancel)


class AStarDetector:
    """Constrained A* on position, heading, gap and step count."""

    algorithm_id = "astar"

    def run(self, dem, parameters, *, endpoints=None, progress=None, cancel=None):
        """Use automatic endpoints or two zero-based (column, row) pixel centers."""
        return _run(dem, parameters, self.algorithm_id, progress, cancel, endpoints)


def _run(dem, p, algorithm_id, progress, cancel, endpoints=None):
    _validate(p)
    check_cancel(cancel)
    if progress:
        progress(0)
    field = gradient_field(dem)
    strength = field[0]
    paths, confidence, properties = [], [], []
    metadata = asdict(p)
    if endpoints is not None:
        points = np.asarray(endpoints, dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError("Select two finite raster pixel coordinates")
        seeds = [tuple(np.rint(points[0, ::-1]).astype(int))]
        fixed_end = tuple(np.rint(points[1, ::-1]).astype(int))
        for name, point in zip(("start", "end"), points, strict=True):
            metadata[f"{name}_x"], metadata[f"{name}_y"] = map(float, point)
    else:
        eligible = (strength >= p.support_threshold) & (strength == maximum_filter(strength, 3))
        indices = np.flatnonzero(eligible)
        if indices.size > p.max_paths * 256:
            ranked = np.lexsort((indices, -strength.ravel()[indices]))
            indices = indices[ranked[:p.max_paths * 256]]
        indices = indices[np.argsort(-strength.ravel()[indices], kind="stable")]
        seeds = list(zip(*np.unravel_index(indices, strength.shape)))
    occupied = np.zeros(strength.shape, dtype=bool)
    attempts = 0
    for y, x in seeds:
        check_cancel(cancel)
        if endpoints is None and occupied[y, x]:
            continue
        if len(paths) >= p.max_paths or attempts >= p.max_paths * 8:
            break
        attempts += 1
        cost = None
        if algorithm_id == "astar":
            if endpoints is not None:
                start, end = (y, x), fixed_end
            else:
                candidate = _trace((y, x), field, p, occupied, cancel)
                start, end = (candidate[0][::-1], candidate[-1][::-1])
                if hypot(start[0] - end[0], start[1] - end[1]) < p.min_length_px:
                    continue
            search = _search(start, end, field, p, cancel)
            if search is None:
                continue
            path, cost = search
        else:
            path = _trace((y, x), field, p, occupied, cancel)
        if len(path) < 2 or np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < p.min_length_px:
            continue
        xc, yc = np.asarray(path).T
        coverage = float(np.mean(strength[yc, xc] >= p.support_threshold))
        if coverage < 0.65:
            continue
        paths.append(path)
        confidence.append(coverage)
        properties.append({"support_fraction": coverage, "search_cost": cost})
        occupied[yc, xc] = True
        if progress:
            progress(min(95, round(100 * len(paths) / p.max_paths)))
    if algorithm_id == "segment-tracing":
        paths = _join_fragments(paths, field, p, cancel)
        confidence = [
            float(np.mean([strength[y, x] >= p.support_threshold for x, y in path]))
            for path in paths
        ]
        properties = [{"support_fraction": value} for value in confidence]
    check_cancel(cancel)
    result = build_path_result(
        dem, paths, algorithm_id=algorithm_id, parameters=metadata,
        confidences=confidence, properties=properties,
    )
    if progress:
        progress(100)
    return result
