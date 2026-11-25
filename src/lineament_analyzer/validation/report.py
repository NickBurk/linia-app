"""Reproducible validation experiment reports in JSON, CSV and HTML."""

import csv
import html
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from lineament_analyzer import __version__

LIMITATIONS = (
    "Confidence is uncalibrated. The methods' support scores are not probabilities.",
    "Synthetic accuracy measures recovery of planted structures, not geological accuracy.",
    "Reference geometry is clipped to the DEM footprint, not to unlabelled areas.",
    "Width and raster resolution affect whether edge detections match reference centrelines.",
    "Position errors are symmetrically sampled on matched centreline geometries.",
)


def make_report(results, evaluations, parameters, *, timings=None, dataset=None):
    rows = []
    for label, result in results.items():
        evaluation = evaluations[label]
        rows.append({
            "label": label, "algorithm": result.algorithm_id,
            "parameters": dict(result.parameters),
            "postprocessing": dict(result.postprocessing_parameters),
            "raster": dict(result.analysis_metadata),
            "source_shape": result.source_shape,
            "source_transform": result.source_transform,
            "source_crs_wkt": result.source_crs_wkt,
            "metrics": evaluation.metrics,
            "measurement": asdict(timings[label]) if timings and label in timings else {},
            "matches": [asdict(pair) for pair in evaluation.matches],
            "orientation_errors_deg": evaluation.orientation_errors_deg,
            "detected_rose": evaluation.detection_rose,
            "reference_rose": evaluation.reference_rose,
            "detected_coordinates": evaluation.detected_coordinates,
            "reference_coordinates": evaluation.reference_coordinates,
        })
    return {"application_version": __version__, "generated_utc": datetime.now(UTC).isoformat(),
            "dataset": asdict(dataset) if dataset else None, "matching": asdict(parameters),
            "limitations": LIMITATIONS, "results": rows}


def export_report(path, report):
    target = Path(path)
    rows = report["results"]
    fields = list(dict.fromkeys(
        key for row in rows for key in (*row["metrics"], *row["measurement"])
    ))
    if target.suffix.lower() == ".csv":
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["algorithm", *fields])
            writer.writeheader()
            for row in rows:
                writer.writerow({"algorithm": row["label"], **row["metrics"], **row["measurement"]})
    elif target.suffix.lower() == ".html":
        def cell(value):
            return html.escape(f"{value:.5g}" if isinstance(value, float) else str(
                "" if value is None else value
            ))

        columns = "".join(f"<th>{cell(label)}</th>" for label in ("Algorithm", *fields))
        records = ""
        for row in rows:
            values = {**row["metrics"], **row["measurement"]}
            records += "<tr>" + "".join(
                f"<td>{cell(value)}</td>" for value in [row["label"], *(values.get(k) for k in fields)]
            ) + "</tr>"
        document = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Validation report</title>'
            '<style>body{font:16px system-ui;margin:2rem;color:#132638}'
            'td,th{padding:.5rem;border:1px solid #ccc}table{border-collapse:collapse}'
            'pre{overflow:auto}</style><h1>Validation report</h1>'
        )
        if not any("precision" in row["metrics"] for row in rows):
            document += "<p>Insufficient reference data to calculate accuracy metrics.</p>"
        document += f"<table><thead><tr>{columns}</tr></thead><tbody>{records}</tbody></table>"
        document += "<h2>Experiment metadata and spatial results</h2><pre>"
        document += html.escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        target.write_text(document + "</pre></html>", encoding="utf-8")
    elif target.suffix.lower() == ".json":
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")
    else:
        raise ValueError("Choose an HTML, JSON or CSV validation report")
