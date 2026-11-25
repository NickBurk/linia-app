# PHASE 06 — scientific validation

The Qt-independent validation layer imports GeoJSON, Shapefile and GeoPackage
reference lines, checks geometry and CRS, and runs one-to-one metre-based spatial
matching. The GUI presents comparable raw, postprocessed and consensus rows, a
match map, axial roses and orientation errors. Reports export metrics, parameters,
metadata, matched pairs and reference/detected coordinates. See the
[metric definitions and assumptions](validation-methodology.md).

The generator writes a DEM GeoTIFF, a GeoPackage of planted truth lines, a dataset
descriptor and generation metadata. Its scene includes horizontal, vertical and
oblique ridges, a curving valley, parallel lines, intersections and gaps, hills,
a depression, sensor noise and a NoData patch. The run below used the validation
fold with eight planted line objects, a 256 × 256 grid, 10 m pixels, seed 7 and
Gaussian noise with σ = 0.6 m. The CRS is EPSG:32633. The default methods ran at
native resolution with a matching tolerance of 35 m, 20° axial tolerance, 0.55
minimum two-way coverage and 0.45 minimum length ratio. Parameters were applied
uniformly without search over the validation results.

| Method | Detections | Precision | Recall | F1 | Buffered IoU | Wall time (s) | Sampled RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsd | 24 | 0.292 | 0.875 | 0.438 | 0.529 | 0.011 | 195.3 |
| canny-hough | 36 | 0.167 | 0.750 | 0.273 | 0.554 | 0.011 | 198.7 |
| edlines | 33 | 0.212 | 0.875 | 0.341 | 0.520 | 0.010 | 199.3 |
| astar | 32 | 0.031 | 0.125 | 0.050 | 0.162 | 3.250 | 205.6 |
| segment-tracing | 31 | 0.097 | 0.375 | 0.154 | 0.359 | 0.053 | 205.8 |
| ransac | 24 | 0.292 | 0.875 | 0.438 | 0.559 | 0.104 | 204.3 |
| directional | 48 | 0.104 | 0.625 | 0.179 | 0.429 | 0.102 | 208.9 |
| wavelet | 48 | 0.167 | 1.000 | 0.286 | 0.470 | 0.077 | 210.4 |
| log-gabor | 48 | 0.104 | 0.625 | 0.179 | 0.479 | 0.138 | 217.2 |
| shearlet | 48 | 0.125 | 0.750 | 0.214 | 0.411 | 0.143 | 218.9 |
| consensus | 49 | 0.143 | 0.875 | 0.246 | 0.530 | 0.526 | 208.1 |
| lsd · post | 20 | 0.350 | 0.875 | 0.500 | 0.522 | 0.003 | 208.4 |
| canny-hough · post | 20 | 0.300 | 0.750 | 0.429 | 0.550 | 0.003 | 208.4 |
| edlines · post | 24 | 0.208 | 0.625 | 0.312 | 0.498 | 0.003 | 208.5 |
| astar · post | 32 | 0.031 | 0.125 | 0.050 | 0.162 | 0.008 | 208.5 |
| segment-tracing · post | 31 | 0.097 | 0.375 | 0.154 | 0.359 | 0.017 | 208.5 |
| ransac · post | 16 | 0.375 | 0.750 | 0.500 | 0.543 | 0.003 | 208.5 |
| directional · post | 48 | 0.104 | 0.625 | 0.179 | 0.429 | 0.027 | 208.6 |
| wavelet · post | 48 | 0.167 | 1.000 | 0.286 | 0.470 | 0.031 | 208.6 |
| log-gabor · post | 48 | 0.104 | 0.625 | 0.179 | 0.479 | 0.027 | 208.6 |
| shearlet · post | 48 | 0.125 | 0.750 | 0.214 | 0.411 | 0.028 | 208.7 |
| consensus · post | 37 | 0.189 | 0.875 | 0.311 | 0.539 | 0.012 | 208.7 |

The synthetic reference records centrelines, so edge-pair outputs and short fragments
often appear as additional detections. The score spread supports further scenario-based
testing. In this run postprocessing improved F1 for some algorithms and reduced it
for EDLines. Consensus retained useful agreement while leaving substantial false
positives under the fixed tolerances.

No independently documented real geological truth was available. The validation
infrastructure can register a real layer and check cross-partition spatial separation.
Geological accuracy remains unestablished. Confidence remains an uncalibrated
method-specific score. A field-labelled study should fix the meaning and completeness
of the reference layer before calibration or independent testing.
