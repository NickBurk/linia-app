"""Reproduce synthetic accuracy and Linux/Windows detector resource measurements."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from lineament_analyzer.algorithms.postprocessing import PostprocessParameters, postprocess_result
from lineament_analyzer.data.dem import load_preview
from lineament_analyzer.engine import METHODS, detector_response, measure, run_detector
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.validation.datasets import generate_synthetic
from lineament_analyzer.validation.experiment import run_experiment
from lineament_analyzer.validation.geometry import MatchParameters
from lineament_analyzer.validation.report import export_report


def write_profile_raster(path, size):
    """Write sparse, predictable ridges without constructing a full NumPy grid."""
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=1, dtype="float32",
        crs="EPSG:32633", transform=from_origin(420000, 5100000, 10, 10),
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dataset:
        x = np.arange(size)[None, :]
        for row in range(0, size, 128):
            height = min(128, size - row)
            y = np.arange(row, row + height)[:, None]
            values = (
                20 * np.exp(-((y - .28 * size) / 3)**2) +
                22 * np.exp(-((x - .62 * size) / 2)**2) +
                18 * np.exp(-((y - .45 * x - .20 * size) / 4)**2) +
                0.003 * (x + y)
            )
            dataset.write(values.astype(np.float32), 1, window=Window(0, row, size, height))
        dataset.set_band_unit(1, "m")


def profile(path, methods, budget):
    preview = load_preview(path)
    result_rows = []
    filters = {"directional", "wavelet", "log-gabor", "shearlet"}
    for method in methods:
        p = {"levels": 2} if method in filters else {}
        if method == "astar":
            p = {"max_paths": 8, "max_expansions": 5000}
        elif method == "segment-tracing":
            p = {"max_paths": 8}
        _, response = measure(lambda method=method, p=p: detector_response(
            preview, method, METHODS[method].parameters(**p)
        )) if method in filters else (None, None)
        raw, detection = measure(lambda method=method, p=p: run_detector(
            path, method, p, resolution="full", memory_mib=budget
        ))
        post, post_time = measure(lambda raw=raw: postprocess_result(raw, PostprocessParameters()))
        _, output = measure(lambda post=post: export_geojson(path.with_suffix(".geojson"), preview, post))
        result_rows.append({
            "method": method, "analysis_shape": raw.analysis_metadata["analysis_shape"],
            "window_px": raw.analysis_metadata["window_px"], "segments": len(raw.segments),
            "response_preview": asdict(response) if response else None,
            "detection": asdict(detection), "postprocessing": asdict(post_time),
            "export_geojson": asdict(output),
        })
    return result_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--large", action="store_true", help="Also profile 4096² and 6144² rasters")
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic(root / "accuracy", size=256, seed=7)
    _, _, _, report = run_experiment(
        dataset, dataset.dem_path, {key: {} for key in METHODS},
        MatchParameters(35), postprocessing=PostprocessParameters(), consensus=True,
        resolution="full",
    )
    export_report(root / "validation-report.json", report)
    sizes = [(1024, tuple(METHODS), 1024)]
    if args.large:
        sizes.extend([
            (4096, ("lsd", "canny-hough", "directional", "wavelet"), 512),
            (6144, ("lsd",), 512),
        ])
    measurements = {}
    for size, methods, budget in sizes:
        print(f"Profiling {size} × {size}", flush=True)
        path = root / f"profile-{size}.tif"
        write_profile_raster(path, size)
        measurements[size] = profile(path, methods, budget)
        (root / "profile.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
