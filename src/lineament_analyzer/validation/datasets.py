"""Synthetic terrain truth, user reference layers and spatially separated datasets."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fiona
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from shapely import LineString, Polygon, distance, points, transform
from shapely.geometry import mapping, shape

from lineament_analyzer.core.windowed import check_cancel

FOLDS = ("calibration", "validation", "independent-test")


@dataclass(frozen=True)
class Dataset:
    dem_path: str
    reference_path: str
    region: str
    fold: str
    reference_layer: str | None = None
    confirmed_crs: str | None = None
    synthetic: bool = False

    def __post_init__(self):
        if self.fold not in FOLDS or not self.region.strip():
            raise ValueError("Specify a region and a calibration, validation or independent-test fold")

    def save(self, destination):
        Path(destination).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, source):
        return cls(**json.loads(Path(source).read_text(encoding="utf-8")))


def footprint(affine, width, height):
    return Polygon([affine * point for point in ((0, 0), (width, 0), (width, height), (0, height))])


def _lines(geometry):
    if geometry.geom_type == "LineString":
        if geometry.length > 0:
            yield geometry
    elif geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _lines(part)


def reference_layers(path):
    return tuple(fiona.listlayers(path))


def load_reference(path, dem_path, *, layer=None, confirmed_crs=None, cancel=None):
    """Load and clip a line layer to the DEM footprint. GeoJSON defaults to WGS84 per RFC 7946."""
    with rasterio.open(dem_path) as dem:
        if not dem.crs:
            raise ValueError("Assign the DEM CRS before importing reference data")
        destination = CRS.from_user_input(dem.crs)
        extent = footprint(dem.transform, dem.width, dem.height)
    with fiona.open(path, layer=layer) as collection:
        declared = collection.crs_wkt or confirmed_crs
        if not declared:
            raise ValueError("Reference CRS missing. Enter its confirmed EPSG code or WKT.")
        origin = CRS.from_user_input(declared)
        converter = Transformer.from_crs(origin, destination, always_xy=True)
        output = []
        for feature in collection:
            check_cancel(cancel)
            geometry = feature.geometry
            if geometry is None:
                raise ValueError(f"Reference feature {feature.id} is empty")
            line = shape(geometry)
            if line.geom_type not in {"LineString", "MultiLineString"} or line.is_empty or not line.is_valid:
                raise ValueError(f"Reference feature {feature.id} must be a valid nonempty line")

            def convert(x, y, *_):
                return converter.transform(x, y, errcheck=True)

            projected = transform(line, convert, interleaved=False)
            if not projected.is_valid or not np.isfinite(projected.bounds).all():
                raise ValueError(f"Reference feature {feature.id} cannot be transformed into DEM CRS")
            output.extend(_lines(projected.intersection(extent)))
        if not output:
            raise ValueError("No reference lines overlap the DEM footprint")
        return tuple(output)


def check_splits(datasets, minimum_separation_m=2000):
    """Reject regions shared across folds or spatial neighbours below the margin."""
    if not np.isfinite(minimum_separation_m) or minimum_separation_m < 0:
        raise ValueError("Minimum separation must be a finite nonnegative distance")
    areas = []
    for dataset in datasets:
        with rasterio.open(dataset.dem_path) as src:
            if src.crs is None:
                raise ValueError("All validation DEMs require a CRS")
            converter = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            area = transform(footprint(src.transform, src.width, src.height), converter.transform, interleaved=False)
        areas.append((dataset, area))
    for i, (first, a) in enumerate(areas):
        for second, b in areas[:i]:
            if first.fold == second.fold:
                continue
            if first.region.casefold() == second.region.casefold():
                raise ValueError(f"Region {first.region!r} appears in multiple folds")
            center = a.centroid
            local = CRS.from_proj4(
                f"+proj=aeqd +lon_0={center.x} +lat_0={center.y} +datum=WGS84 +units=m"
            )
            converter = Transformer.from_crs("EPSG:4326", local, always_xy=True)
            aa, bb = (transform(polygon, converter.transform, interleaved=False) for polygon in (a, b))
            if aa.intersects(bb) or aa.distance(bb) < minimum_separation_m:
                raise ValueError(f"Regions {first.region!r} and {second.region!r} are too close")


def generate_synthetic(
    directory, *, size=256, resolution_m=10, seed=7, noise_m=0.6, fold="validation",
    nodata=True, progress=None, cancel=None,
):
    """Write a synthetic DEM, exact planted vector centrelines and generator metadata."""
    if not 64 <= size <= 8192 or not 0 < resolution_m <= 100 or not 0 <= noise_m <= 100:
        raise ValueError("Invalid synthetic raster dimensions, resolution or noise")
    if fold not in FOLDS:
        raise ValueError("Invalid dataset fold")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tif, reference = root / "synthetic.tif", root / "reference.gpkg"
    region_index = FOLDS.index(fold)
    affine = Affine(resolution_m, 0, 420000 + region_index * 100000,
                    0, -resolution_m, 5100000)
    structures = [
        ("ridge-horizontal", ((.08, .19), (.78, .19)), 30, 2),
        ("valley-vertical", ((.75, .07), (.75, .72)), -26, 2),
        ("ridge-diagonal", ((.09, .48), (.55, .74)), 25, 2.6),
        ("parallel-ridge", ((.10, .53), (.55, .79)), 19, 1.7),
        ("curved-valley", ((.12, .84), (.24, .81), (.34, .82), (.45, .86), (.56, .94)), -34, 2),
        ("gap-west", ((.21, .36), (.40, .37)), 18, 1.5),
        ("gap-east", ((.45, .372), (.62, .38)), 18, 1.5),
        ("short-steep", ((.80, .85), (.84, .68)), 12, 1.6),
    ]
    y, x = np.ogrid[:size, :size]
    z = (0.018 * x + 0.014 * y
         + 15 * np.exp(-((x / size - .32)**2 + (y / size - .29)**2) / .02)
         - 12 * np.exp(-((x / size - .81)**2 + (y / size - .82)**2) / .012))
    records = []
    for index, (name, vertices, height, sigma) in enumerate(structures):
        check_cancel(cancel)
        # Compute exact raster-grid distance in a bounded patch around each centreline.
        vertices = np.asarray(vertices) * (size - 1)
        pad = int(np.ceil(6 * sigma))
        x0, y0 = np.maximum(np.floor(vertices.min(axis=0)).astype(int) - pad, 0)
        x1, y1 = np.minimum(np.ceil(vertices.max(axis=0)).astype(int) + pad + 1, size)
        py, px = np.mgrid[y0:y1, x0:x1]
        pixels = points(np.stack((px, py), axis=-1))
        offset = distance(pixels, LineString(vertices))
        z[y0:y1, x0:x1] += height * np.exp(-0.5 * (offset / sigma)**2)
        coords = tuple(affine * (float(xx) + .5, float(yy) + .5) for xx, yy in vertices)
        records.append({"name": name, "amplitude_m": height, "width_sigma_px": sigma,
                        "pixel_vertices": vertices.tolist(), "coordinates": coords})
        if progress:
            progress(round(80 * (index + 1) / len(structures)))
    z += np.random.default_rng(seed).normal(0, noise_m, z.shape)
    if nodata:
        z[round(.06 * size):round(.15 * size), round(.82 * size):round(.92 * size)] = np.nan
    check_cancel(cancel)
    with rasterio.open(tif, "w", driver="GTiff", width=size, height=size, count=1,
                       dtype="float32", crs="EPSG:32633", transform=affine, nodata=np.nan) as dst:
        dst.write(z.astype(np.float32), 1)
        dst.set_band_unit(1, "m")
    with fiona.open(reference, "w", driver="GPKG", layer="reference", crs="EPSG:32633",
                    schema={"geometry": "LineString", "properties": {"name": "str"}}) as dst:
        dst.writerecords({"geometry": mapping(LineString(record["coordinates"])),
                          "properties": {"name": record["name"]}} for record in records)
    metadata = {"seed": seed, "noise_sigma_m": noise_m, "noise_model": "independent Gaussian",
                "shape": [size, size], "resolution_m": resolution_m, "crs": "EPSG:32633",
                "transform": tuple(affine)[:6], "nodata": nodata, "structures": records,
                "fold": fold, "generator_version": 1}
    (root / "generation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dataset = Dataset(str(tif), str(reference), f"synthetic-{fold}", fold, "reference",
                      synthetic=True)
    dataset.save(root / "dataset.json")
    if progress:
        progress(100)
    return dataset
