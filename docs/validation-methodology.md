# Validation methodology

Use the same reference layer, raster grid, detector settings and matching thresholds
throughout a comparison. Record the region, partition, DEM path, reference layer
and any confirmed CRS in a validation dataset descriptor.

GeoJSON uses its declared CRS or RFC 7946 WGS84. A Shapefile or GeoPackage layer
without CRS needs an operator-confirmed CRS. Import validates type, emptiness,
geometry and reprojection. It splits multipart lines and clips them to the full
affine footprint of the DEM. It rejects a layer with no overlapping lines. Internal
NoData holes stay inside the study footprint. Reference lines across holes can
raise fragmentation or false-negative counts.

Validation projects lines into a WGS84 azimuthal equidistant frame centred on the
DEM, with a 500 km corner-radius limit. Distances and buffer widths are in metres.
Orientation is the endpoint bearing in that local frame modulo 180°. Exported
bearings instead use true north. Detection lengths sum ellipsoidal lengths of each
vertex pair.

## Spatial matching

For detected line A and reference line B, query an STRtree for candidates at distance
≤ tolerance r and apply an axial angle limit. Each directed coverage is the length
of one line inside the other line's buffer of radius r, divided by the first
line's length. Require both coverages and the ratio of shorter to longer length
to clear their thresholds. Pair score equals minimum directed coverage × length
ratio × (1 − axial error / 180). A sparse bipartite assignment maximizes match
count first and sum of pair scores second. Every detected and reference line can be
matched once. Dense overlaps can still produce a large candidate graph.

| Metric | Definition |
| --- | --- |
| Precision | Matched detections / detections. Zero for an empty detection result with usable truth. |
| Recall | Matched references / references. |
| F1 | Harmonic mean of precision and recall. Zero when both are zero. |
| Buffered IoU | Intersection area divided by union area of detection and reference buffer unions. |
| Positional error | Mean and median of bidirectional point-to-line distances for matched pairs. Sampling interval targets r/2 and is capped at 4096 points per line. |
| Orientation error | Mean absolute axial difference along the shorter arc of the 180° circle. |
| Length error | Mean absolute length difference in metres for matched pairs. |
| Fragmentation | References with at least two compatible detections meeting directed coverage, after omitting duplicates. |
| Duplicate rate | Unmatched detections that clear full thresholds against an already matched reference, divided by all detections. |
| Runtime and memory | Wall time and sampled process RSS, including native libraries. Sampling interval is 10 ms and can miss transient peaks. |

Accuracy fields are omitted when no reference is selected. An empty or invalid
selected layer produces an input error. Position, orientation and length error fields
stay undefined when there are no matches. Rose plots use 15° axial bins.

## Independent datasets and confidence

Keep spatial regions intact across calibration, validation and independent-test
partitions. The Python function check_splits enforces region and footprint separation
across supplied descriptors. Its default buffer is 2000 m. Select a margin that fits
the study's spatial autocorrelation. The GUI processes one dataset per run. Freeze
algorithm choice and parameters before evaluating the independent test partition.
The application does not perform automatic fitting or model selection.

Synthetic folds occupy separate footprints and share a scenario generator. They
measure recovery of planted features. Reference definition matters. Detectors may
track the edge of a wide ridge whose truth line follows the centre. No documented
independent geological reference was available for this release. Confidence stays
a method-specific support or fit score. Brier score and reliability diagrams require
independent labels and a justified probability calibration.
