"""Robust straight-line fitting on spatial groups of DEM edge samples."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label

from lineament_analyzer.algorithms.canny import CannyParameters, detect_edges
from lineament_analyzer.algorithms.results import build_detection_result
from lineament_analyzer.algorithms.support import detector_support_mask, sample_mask_support
from lineament_analyzer.core.windowed import check_cancel


@dataclass(frozen=True)
class RansacParameters:
    residual_px: float = 1.5
    iterations: int = 100
    min_inliers: int = 12
    max_gap_px: int = 4
    max_lines: int = 64
    seed: int = 0


def fit_line(points, parameters, rng, cancel=None):
    """Return a total least-squares line and inlier mask after two-point sampling."""
    p = parameters
    best = None
    score = (0, float("-inf"))
    for iteration in range(p.iterations):
        if iteration % 16 == 0:
            check_cancel(cancel)
        a, b = points[rng.choice(len(points), size=2, replace=False)]
        delta = b - a
        separation = np.linalg.norm(delta)
        if separation == 0:
            continue
        along = (points - a) @ (delta / separation)
        normal = np.array((-delta[1], delta[0])) / separation
        errors = abs((points - a) @ normal)
        inliers = (errors <= p.residual_px) & (
            np.abs(along - separation / 2) <= separation / 2 + p.max_gap_px
        )
        count = int(inliers.sum())
        if count >= p.min_inliers:
            candidate = (count, -float(np.mean(errors[inliers] ** 2)))
            if candidate > score:
                score, best = candidate, inliers
    if best is None:
        return None
    selected = points[best]
    center = selected.mean(axis=0)
    centered = selected - center
    _, vectors = np.linalg.eigh(centered.T @ centered)
    direction = vectors[:, -1]
    normal = np.array((-direction[1], direction[0]))
    errors = abs((points - center) @ normal)
    along = (points - center) @ direction
    lower, upper = sorted((float(along[best].min()), float(along[best].max())))
    inliers = (errors <= p.residual_px) & (along >= lower) & (along <= upper)
    return center, direction, along, errors, inliers


class RansacDetector:
    algorithm_id = "ransac"

    def run(self, dem, parameters, *, progress=None, cancel=None):
        p = parameters
        if (
            not 0.1 <= p.residual_px <= 16 or not 1 <= p.iterations <= 5000
            or not 2 <= p.min_inliers <= 4096 or not 1 <= p.max_gap_px <= 32
            or not 1 <= p.max_lines <= 512 or not 0 <= p.seed < 2**32
        ):
            raise ValueError("Invalid RANSAC parameters")
        check_cancel(cancel)
        if progress:
            progress(0)
        valid = dem.valid & np.isfinite(dem.values)
        safe = detector_support_mask(valid, radius=2)
        edge = detect_edges(dem.values, valid, CannyParameters(), cancel=cancel)[0] != 0
        edge &= safe
        groups, _ = label(binary_dilation(edge, iterations=max(1, p.max_gap_px // 2)) & valid)
        rng = np.random.default_rng(p.seed)
        lines, properties, confidence = [], [], []
        boxes = find_objects(groups)
        for index, box in enumerate(boxes, start=1):
            check_cancel(cancel)
            if box is None:
                continue
            y, x = np.nonzero(edge[box] & (groups[box] == index))
            points = np.column_stack((x + box[1].start, y + box[0].start)).astype(float)
            # Bound each randomized fitting pass over a very dense component.
            if len(points) > 32768:
                points = points[np.sort(rng.choice(len(points), size=32768, replace=False))]
            while len(points) >= p.min_inliers and len(lines) < p.max_lines:
                fitted = fit_line(points, p, rng, cancel)
                if fitted is None:
                    break
                center, direction, along, errors, inliers = fitted
                if inliers.sum() < p.min_inliers:
                    break
                ordered = np.flatnonzero(inliers)
                ordered = ordered[np.argsort(along[ordered], kind="stable")]
                for chunk in np.split(ordered, np.flatnonzero(np.diff(along[ordered]) > p.max_gap_px) + 1):
                    if len(chunk) < p.min_inliers:
                        continue
                    start, end = center + along[chunk[[0, -1]], None] * direction
                    line = np.r_[start, end]
                    if np.linalg.norm(end - start) < 1 or sample_mask_support(safe, line) is None:
                        continue
                    rmse = float(np.sqrt(np.mean(errors[chunk] ** 2)))
                    lines.append(line)
                    confidence.append(max(0.0, 1 - rmse / p.residual_px))
                    properties.append({"inliers": len(chunk), "rmse_px": rmse})
                    if len(lines) == p.max_lines:
                        break
                points = points[~inliers]
            if progress:
                progress(10 + round(85 * index / len(boxes)))
            if len(lines) == p.max_lines:
                break
        result = build_detection_result(
            dem, lines, algorithm_id=self.algorithm_id, parameters=p,
            confidences=confidence, properties=properties,
        )
        check_cancel(cancel)
        if progress:
            progress(100)
        return result
