"""Public, Qt-independent detector registry and raster execution API."""

from dataclasses import dataclass, replace
from threading import Event, Thread
from time import perf_counter
from types import MappingProxyType

import numpy as np
import psutil
import rasterio
from rasterio.windows import Window

from lineament_analyzer.algorithms.canny_hough import CannyHoughDetector, CannyHoughParameters
from lineament_analyzer.algorithms.edlines import EdLinesDetector, EdLinesParameters
from lineament_analyzer.algorithms.filter_banks import (
    DirectionalParameters,
    FilterDetector,
    LogGaborParameters,
    ShearletParameters,
    WaveletParameters,
    response_map,
)
from lineament_analyzer.algorithms.lsd import LsdDetector, LsdParameters
from lineament_analyzer.algorithms.ransac import RansacDetector, RansacParameters
from lineament_analyzer.algorithms.tracing import (
    AStarDetector,
    AStarParameters,
    SegmentTracingDetector,
    TraceParameters,
)
from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.data.dem import Dem, load_dem, load_preview


@dataclass(frozen=True)
class Method:
    name: str
    detector: object
    parameters: type
    family: str
    bytes_per_pixel: int
    windowed: bool = False


METHODS = MappingProxyType({
    "lsd": Method("LSD", LsdDetector(), LsdParameters, "lsd", 96, True),
    "canny-hough": Method("Canny + Hough", CannyHoughDetector(), CannyHoughParameters,
                          "canny-geometric", 144, True),
    "edlines": Method("EDLines", EdLinesDetector(), EdLinesParameters, "edlines", 144, True),
    "astar": Method("A*", AStarDetector(), AStarParameters, "gradient-path", 160),
    "segment-tracing": Method("Segment Tracing", SegmentTracingDetector(), TraceParameters,
                              "gradient-path", 128),
    "ransac": Method("RANSAC", RansacDetector(), RansacParameters, "canny-geometric", 160, True),
    "directional": Method("Directional Filters", FilterDetector("directional"),
                          DirectionalParameters, "filter-ridge", 256, True),
    "wavelet": Method("Wavelet", FilterDetector("wavelet"), WaveletParameters,
                      "filter-ridge", 256, True),
    "log-gabor": Method("Log-Gabor", FilterDetector("log-gabor"), LogGaborParameters,
                        "filter-ridge", 320),
    "shearlet": Method("Shearlet (experimental)", FilterDetector("shearlet"),
                       ShearletParameters, "filter-ridge", 384),
})

# Labels (RU / EN) and validated widget limits for the new parameter forms.
CONTROLS = MappingProxyType({
    "min_length_px": ("Мин. длина (px)", "Minimum length (px)", 2, 4096),
    "max_steps": ("Макс. число шагов", "Maximum path steps", 2, 4096),
    "max_turn_deg": ("Поворот за шаг (°)", "Maximum turn per step (°)", 0, 90),
    "support_threshold": ("Порог отклика пути", "Path response threshold", 0.01, 1),
    "max_gap_px": ("Макс. разрыв (px)", "Maximum gap (px)", 0, 16),
    "max_paths": ("Макс. число путей", "Maximum candidate paths", 1, 512),
    "turn_weight": ("Вес поворота", "Turn penalty weight", 0, 100),
    "orientation_weight": ("Вес ориентации", "Orientation penalty weight", 0, 100),
    "max_expansions": ("Лимит состояний", "Search expansion budget", 1, 1_000_000),
    "residual_px": ("Допуск остатка (px)", "Residual tolerance (px)", 0.1, 16),
    "iterations": ("Число гипотез", "Number of hypotheses", 1, 5000),
    "min_inliers": ("Мин. число inliers", "Minimum inlier count", 2, 4096),
    "max_lines": ("Макс. число линий", "Maximum candidate lines", 1, 512),
    "seed": ("Зерно генератора", "Random seed", 0, 2147483647),
    "levels": ("Число масштабов", "Scale levels", 1, 5),
    "threshold": ("Порог отклика фильтра", "Filter response threshold", 0.01, 1),
    "minimum_length_px": ("Мин. длина (px)", "Minimum length (px)", 2, 512),
    "angle_start_deg": ("Начальный угол растра (°)", "First raster angle (°)", 0, 179),
    "angle_end_deg": ("Конечный угол растра (°)", "Exclusive end raster angle (°)", 1, 180),
    "angle_step_deg": ("Угловой шаг (°)", "Angular step (°)", 5, 90),
    "first_scale_px": ("Начальный масштаб (px)", "Initial scale (px)", 0.6, 8),
    "first_frequency": ("Начальная частота (цикл/px)", "Initial frequency (cycles/px)", 0.03, 0.45),
    "bandwidth": ("Ширина полосы в log(f)", "Bandwidth in log(f)", 0.1, 1.5),
})

PROFILES = MappingProxyType({
    "quick": ("Быстрый анализ", "Quick analysis", {"lsd": {}, "directional": {"levels": 1}}),
    "straight": ("Прямолинейные", "Straight structures", {"ransac": {}, "edlines": {}}),
    "curved": ("Криволинейные", "Curved structures", {"astar": {}, "segment-tracing": {}}),
    "weak": ("Слабовыраженные", "Faint structures", {"directional": {"threshold": 0.18}}),
    "noisy": ("Зашумлённые DEM", "Noisy DEMs", {"ransac": {"min_inliers": 24}, "wavelet": {}}),
    "multi": ("Многомасштабный", "Multiscale", {"wavelet": {}, "log-gabor": {}, "shearlet": {}}),
})


@dataclass(frozen=True)
class Measurement:
    seconds: float
    peak_rss_mib: float


def measure(operation):
    """Measure wall time and sampled peak process RSS, including loaded libraries."""
    process = psutil.Process()
    stopped = Event()
    peak = process.memory_info().rss

    def sample():
        nonlocal peak
        while not stopped.wait(0.01):
            peak = max(peak, process.memory_info().rss)

    sampler = Thread(target=sample, daemon=True)
    start = perf_counter()
    sampler.start()
    try:
        value = operation()
        measurement = Measurement(perf_counter() - start,
                                  max(peak, process.memory_info().rss) / 1024**2)
        return value, measurement
    finally:
        stopped.set()
        sampler.join()


def _halo(algorithm, p):
    if algorithm == "directional":
        return 96 + int(np.ceil(4 * p.first_scale_px * 2**(p.levels - 1)))
    if algorithm == "wavelet":
        return 96 + 2 * (2**p.levels - 1)
    return 96


def dataset_grid(dataset):
    """Serializable grid fingerprint for results and project restore."""
    return (dataset.height, dataset.width), tuple(dataset.transform)[:6], (
        dataset.crs.to_wkt() if dataset.crs else None
    )


def _windowed(path, method, algorithm, p, limit, progress, cancel):
    halo = _halo(algorithm, p)
    tile = min(1024, int(np.sqrt(limit / method.bytes_per_pixel)) - 2 * halo)
    if tile < 128:
        raise ValueError(f"Insufficient memory budget for the {halo}-pixel analysis halo")
    all_lines = []
    with rasterio.Env(GDAL_CACHEMAX=16 * 1024**2), rasterio.open(path) as src:
        shape, transform, crs = dataset_grid(src)
        count = ((src.height + tile - 1) // tile) * ((src.width + tile - 1) // tile)
        index = 0
        for row in range(0, src.height, tile):
            for col in range(0, src.width, tile):
                check_cancel(cancel)
                x0, y0 = max(0, col - halo), max(0, row - halo)
                x1, y1 = min(src.width, col + tile + halo), min(src.height, row + tile + halo)
                window = Window(x0, y0, x1 - x0, y1 - y0)
                data = src.read(1, window=window, masked=True).astype(float)
                values = data.data * src.scales[0] + src.offsets[0]
                dem = Dem(values, ~np.ma.getmaskarray(data) & np.isfinite(values),
                          src.window_transform(window), src.crs, path, src.units[0])
                result = method.detector.run(dem, p, cancel=cancel)
                for line in result.segments:
                    vertices = tuple((x + x0, y + y0) for x, y in line.pixels)
                    mx, my = np.asarray(vertices).mean(axis=0)
                    if col <= mx < col + tile and row <= my < row + tile:
                        all_lines.append(replace(
                            line, pixel_start=vertices[0], pixel_end=vertices[-1],
                            pixel_path=vertices if line.pixel_path else (),
                        ))
                index += 1
                if progress:
                    progress(round(100 * index / count))
    return replace(result, segments=tuple(all_lines), source_shape=shape,
                   source_transform=transform, source_crs_wkt=crs), tile, halo


def run_detector(
    source, algorithm, parameters=None, *, resolution="preview", preview=None,
    memory_mib=512, working_path=None, endpoints=None, progress=None, cancel=None,
):
    """Run from a GeoTIFF path or Dem, recording input and actual processing grid.

    A supplied preview is accepted for an already displayed terrain derivative.
    Full-resolution paths may use windows and bounded overlap where supported.
    Interactive endpoints refer to the processing grid in the selected mode.
    """
    method = METHODS[algorithm]
    p = parameters if parameters is not None else method.parameters()
    if isinstance(p, dict):
        p = method.parameters(**p)
    if not isinstance(p, method.parameters):
        raise TypeError(f"{algorithm} requires {method.parameters.__name__}")
    if resolution not in {"preview", "full"} or not np.isfinite(memory_mib) or memory_mib < 16:
        raise ValueError("Choose preview or full resolution and a memory budget of at least 16 MiB")
    limit = min(memory_mib * 1024**2, psutil.virtual_memory().available * 0.8)
    extra = {}
    if endpoints is not None:
        if algorithm != "astar":
            raise ValueError("Interactive endpoints are only supported by A*")
        extra["endpoints"] = endpoints
    check_cancel(cancel)
    if isinstance(source, Dem):
        input_dem = source
        path = source.path
        working = None
        shape = source.values.shape
        unit = source.elevation_unit
    else:
        path = source
        working = working_path or path
        input_dem = preview if resolution == "preview" else None
        with rasterio.open(working) as dataset:
            shape, _, _ = dataset_grid(dataset)
            unit = dataset.units[0]
    used_shape = input_dem.values.shape if input_dem is not None else shape
    if input_dem is None and resolution == "preview":
        ratio = min(1.0, 1600 / max(shape))
        used_shape = tuple(max(1, round(side * ratio)) for side in shape)
    padding = 0
    if algorithm == "log-gabor":
        if not 0.03 <= p.first_frequency <= 0.45 or not 1 <= p.levels <= 5:
            raise ValueError("Invalid Log-Gabor frequency or levels")
        padding = min(max(used_shape), int(np.ceil(2**(p.levels - 1) / p.first_frequency)))
    elif algorithm == "shearlet":
        if not 1 <= p.levels <= 5:
            raise ValueError("Invalid Shearlet levels")
        padding = min(max(used_shape), 2**p.levels)
    required = np.prod(np.asarray(used_shape) + 2 * padding) * method.bytes_per_pixel
    if algorithm == "astar":
        required += p.max_expansions * 1024
    tile, halo = None, None
    if required > limit:
        if not (resolution == "full" and method.windowed and working is not None):
            raise MemoryError(
                f"{method.name} needs an estimated {required / 1024**2:.0f} MiB "
                f"for this grid; available operation budget is {limit / 1024**2:.0f} MiB. "
                "Reduce the analysis area or use Preview."
            )
        result, tile, halo = _windowed(working, method, algorithm, p, limit, progress, cancel)
    else:
        if input_dem is None:
            input_dem = load_preview(working) if resolution == "preview" else load_dem(working)
        result = method.detector.run(input_dem, p, progress=progress, cancel=cancel, **extra)
    check_cancel(cancel)
    a, b, _, d, e, _ = result.source_transform
    metadata = {
        "dem": str(path), "resolution_mode": resolution, "native_shape": list(shape),
        "analysis_shape": list(result.source_shape),
        "pixel_size_source_units": [float(np.hypot(a, d)), float(np.hypot(b, e))],
        "vertical_unit": unit, "window_px": tile, "halo_px": halo,
    }
    lines = tuple(replace(line, properties=MappingProxyType({
        **line.properties, "object_id": f"{algorithm}-{index:06d}",
    })) for index, line in enumerate(result.segments, start=1))
    return replace(result, segments=lines, analysis_metadata=MappingProxyType(metadata))


def run_methods(source, configurations, *, progress=None, cancel=None, **options):
    """Execute methods sequentially over the same source and return timings by ID."""
    if not configurations:
        raise ValueError("Select at least one algorithm")
    results, timings = {}, {}
    for index, (algorithm, p) in enumerate(configurations.items()):
        check_cancel(cancel)

        def update(value, index=index):
            if progress:
                progress(round((100 * index + value) / len(configurations)))

        results[algorithm], timings[algorithm] = measure(
            lambda algorithm=algorithm, p=p: run_detector(
                source, algorithm, p, progress=update, cancel=cancel, **options
            )
        )
    return results, timings


def detector_response(source, algorithm, parameters=None, *, progress=None, cancel=None):
    """Return a bounded-preview response map and its exact DEM grid."""
    method = METHODS[algorithm]
    dem = source if isinstance(source, Dem) else load_preview(source)
    p = parameters if parameters is not None else method.parameters()
    return dem, response_map(dem, p, algorithm, progress=progress, cancel=cancel)
