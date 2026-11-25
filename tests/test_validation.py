import json
from dataclasses import replace

import fiona
import numpy as np
import pytest
from shapely import LineString

from lineament_analyzer.algorithms.results import build_path_result
from lineament_analyzer.data.dem import load_dem, load_preview
from lineament_analyzer.engine import run_detector, run_methods
from lineament_analyzer.export.geojson import export_geojson
from lineament_analyzer.export.geopackage import export_geopackage
from lineament_analyzer.export.shapefile import export_shapefile
from lineament_analyzer.gui.project import load_project, save_project
from lineament_analyzer.validation.comparison import compare_results, evaluate
from lineament_analyzer.validation.datasets import check_splits, generate_synthetic, load_reference
from lineament_analyzer.validation.geometry import MatchParameters
from lineament_analyzer.validation.report import export_report, make_report


def test_synthetic_roundtrip_and_experiment(tmp_path):
    dataset = generate_synthetic(tmp_path, size=128, seed=43)
    dem = load_dem(dataset.dem_path)
    references = load_reference(dataset.reference_path, dataset.dem_path, layer="reference")
    metadata = json.loads((tmp_path / "generation.json").read_text())
    assert len(references) == len(metadata["structures"]) == 8
    assert not dem.valid.all()
    assert next(iter(references[0].coords)) == tuple(metadata["structures"][0]["coordinates"][0])
    truth = build_path_result(dem, [row["pixel_vertices"] for row in metadata["structures"]],
                              algorithm_id="lsd", parameters={})
    evaluation = evaluate(truth, references, MatchParameters(10))
    assert evaluation.metrics["f1"] == 1
    assert evaluation.metrics["buffered_iou"] == pytest.approx(1)
    assert evaluation.metrics["mean_position_m"] < 0.01
    assert "precision" not in evaluate(truth).metrics
    results, timing = run_methods(dataset.dem_path, {"lsd": {}, "directional": {"levels": 1}})
    assert set(results) == set(timing) == {"lsd", "directional"}
    assert results["lsd"].analysis_metadata["resolution_mode"] == "preview"
    with pytest.raises(MemoryError, match="operation budget"):
        run_detector(dataset.dem_path, "astar", {"max_expansions": 100000}, memory_mib=16)
    report = make_report({"truth": truth}, {"truth": evaluation}, MatchParameters(10), dataset=dataset)
    for suffix in ("csv", "html", "json"):
        output = tmp_path / f"report.{suffix}"
        export_report(output, report)
        assert "precision" in output.read_text(encoding="utf-8")


def test_consensus_duplicates_and_fragmentation(tmp_path):
    dataset = generate_synthetic(tmp_path, size=64, noise_m=0)
    dem = load_dem(dataset.dem_path)
    paths = [[(5, 30), (53, 30)], [(5, 31), (53, 31)], [(30, 10), (30, 52)]]
    lsd = build_path_result(dem, paths, algorithm_id="lsd", parameters={})
    second = build_path_result(dem, [[(5, 30), (53, 30)]], algorithm_id="edlines", parameters={})
    p = MatchParameters(11)
    reference = [LineString(lsd.segments[0].coordinates)]
    metrics = evaluate(lsd, reference, p).metrics
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == 1
    assert metrics["duplicate_rate"] == pytest.approx(1 / 3)
    consensus, unique = compare_results([lsd, second], p)
    assert len(consensus.segments) == 1
    assert consensus.segments[0].properties["supporting_algorithms"] == ["edlines", "lsd"]
    assert consensus.segments[0].confidence is None
    assert set(unique["lsd"]) == {1, 2}
    fragments = build_path_result(dem, [[(5, 30), (28, 30)], [(30, 30), (53, 30)]],
                                  algorithm_id="lsd", parameters={})
    evaluation = evaluate(fragments, reference, p)
    assert evaluation.metrics["fragmented_references"] == 1


def test_polyline_exports_and_project(tmp_path):
    dataset = generate_synthetic(tmp_path, size=64)
    dem = load_dem(dataset.dem_path)
    preview = load_preview(dataset.dem_path, 32)
    result = build_path_result(dem, [[(1, 2), (6, 8), (30, 11)]],
                               algorithm_id="astar", parameters={})
    result = replace(result, analysis_metadata={"dem": dataset.dem_path, "resolution_mode": "full"})
    geojson = tmp_path / "result.geojson"
    export_geojson(geojson, preview, result)
    feature = json.loads(geojson.read_text(encoding="utf-8"))["features"][0]
    assert len(feature["geometry"]["coordinates"]) == 3
    assert feature["properties"]["analysis"]["resolution_mode"] == "full"
    for name, writer in (("output.shp", export_shapefile), ("output.gpkg", export_geopackage)):
        path = tmp_path / name
        writer(path, preview, result)
        with fiona.open(path) as src:
            assert len(next(iter(src)).geometry.coordinates) == 3
    project = tmp_path / "test.laproj"
    save_project(project, {"dem_path": dem.path, "detection_results": {"astar": result}})
    assert load_project(project)["detection_results"]["astar"] == result


def test_spatially_independent_folds(tmp_path):
    a = generate_synthetic(tmp_path / "a", size=64, fold="calibration")
    b = generate_synthetic(tmp_path / "b", size=64, fold="independent-test")
    check_splits([a, b])
    with pytest.raises(ValueError, match="too close"):
        check_splits([a, replace(b, dem_path=a.dem_path)])
    with pytest.raises(ValueError, match="multiple folds"):
        check_splits([a, replace(b, region=a.region)])
    assert np.isfinite(load_dem(a.dem_path).values).any()
