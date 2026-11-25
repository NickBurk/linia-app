# Line detector reference

All methods operate on the current bounded DEM preview and return the common
`DetectionResult` / `LineSegment` format. Pixel endpoints are zero-based pixel
centres; source coordinates use the preview's complete affine transform and
CRS. WGS84 GeoJSON converts those same pixel centres with the half-pixel offset
documented for PHASE 02.

The common record contains pixel and source-CRS start/end coordinates, pixel
length, ellipsoidal metric length, forward azimuth, axial orientation,
confidence, algorithm ID and detector-specific properties. Each result also
records the source CRS, affine transform and raster shape to prevent applying
pixel endpoints to another raster. Forward azimuth is clockwise from true
north in `[0, 360)`; axial orientation is in `[0, 180)`. Confidence currently
records the fraction of sampled segment support that coincides with the
detector's edge image for Canny–Hough and EDLines. It is a diagnostic score,
not a probability, geological certainty or cross-algorithm calibration. LSD
leaves confidence undefined because its current adapter does not expose a
score.

| Method | Controls | Notes |
| --- | --- | --- |
| LSD | Minimum preview-pixel length; detector scale is 0.8 | Existing OpenCV LSD primitive. Retains the phase 02 Nx4 `detect` API. |
| Canny + Hough | Canny low/high threshold (normalized 8-bit intensity), aperture, L2 gradient, Hough vote threshold, minimum length (m), maximum gap (m) | Canny edge detection (`canny.detect_edges`) and Probabilistic Hough (`hough.detect_lines`) are standalone components with independent parameter dataclasses. The GUI selects their line-extraction pipeline. Map-unit lengths are converted using the geometric-mean local ground sample distance and output length is filtered geodesically. |
| EDLines | Sigma, gradient/anchor threshold, minimum path length (px), minimum segment length (m), fit/max error, maximum line separation (px), NFA validation, PF mode, edge operator, scan interval | Independent OpenCV contrib Edge Drawing `detectLines` backend. It has its own normalization and parameter set and never falls back to Canny–Hough. |

NoData pixels are extended from their nearest valid sample only to prevent an
artificial intensity cliff inside the native edge operators. Candidates whose
sampled support touches a dilated NoData boundary are discarded. Real image
borders remain valid. All metric length thresholds use source georeferencing;
in geographic CRSs the local pixel scale varies with latitude, and the preview
is not resampled to a globally uniform ground grid. The local metric scale is
an approximation for Hough gap/minimum-pixel settings; exact result lengths
still use the CRS ellipsoid.

New detectors such as A*, RANSAC, shearlet, wavelet, Log-Gabor and directional
filters should implement the shared `LineDetector` contract and keep their
parameters in their own modules. They remain unavailable until implemented and
tested.

## Postprocessing

The Qt-independent `algorithms.postprocessing` module accepts any
`DetectionResult`. Its `PostprocessParameters` independently configures merge
angle, endpoint distance, collinearity tolerance, along-line maximum gap,
crossing preservation, minimum/maximum preview-pixel length, optional axial
orientation limits, optional minimum confidence, duplicate tolerance,
direction-bin count, length-class thresholds and density-cluster radius.
Preview-pixel distance thresholds are deliberate: they work without a CRS and
match the pixel-coordinate geometry used by the bounded detector preview.

Merging is conservative: only near-collinear segment endpoints can connect;
proper interior crossings remain separate by default. When a merge combines
segments with defined confidence scores, their mean score is retained. Missing
scores remain missing and are not treated as zero by confidence filtering.
Classification records direction, length, orientation-derived geometry and
centroid-density cluster labels on each feature. `class_id` combines these
categories for coloring and per-class summaries.

`statistics.summarize_results` bins axial orientation in `[0, 180)` and reports
length distribution, direction counts, object count, known metric-length
summary, footprint density per square kilometre when CRS metadata allows it,
and class counts. When any segment has unknown metric length, the total metric
length is `null` instead of silently summing only a subset.

GeoJSON export retains source and postprocessing parameters, class and object
IDs, source/pixel endpoints and geometry metrics. Shapefile output keeps
geometry in the source CRS and writes `.prj`/`.cpg` files for QGIS plus a
`.metadata.json` companion with complete attributes and parameter objects.

## A*, Segment Tracing and RANSAC

A* and Segment Tracing use a smoothed DEM gradient, its perpendicular tangent and a
conservative NoData margin. A* searches eight-connected raster cells with heading,
unsupported-run length and step count as state. Its cost is the step length times
`1 + 2(1 − strength) + orientation_weight(1 − |cos(heading − tangent)|)` plus the
squared turn penalty. The Euclidean distance to the endpoint is an admissible heuristic.
Interactive endpoints must lie on supported pixels. Automatic endpoints come from
bounded bidirectional tracing. The expansion budget limits time and memory per path.

Segment Tracing is a local greedy search starting at gradient maxima. Its score combines
strength, tangent alignment and a turn penalty. Visited and occupied pixels, turning
limits, maximum gap and route length stop the search. Both methods require at least 65%
per-pixel support and preserve ordered vertices. Their output length is the sum of
ellipsoidal segment lengths. The endpoint bearing describes the overall axial direction.

RANSAC samples pairs of Canny edge pixels within spatial groups. Inliers fall within
the perpendicular residual tolerance and the hypothesis extent. The best hypothesis is
refitted by total least squares, split at large projected gaps and checked against the
source mask. RMSE is reported in pixels. Its support score is derived from fit residuals.
The seed, iteration and candidate budgets make the search bounded and reproducible.

## Directional, wavelet, Log-Gabor and experimental Shearlet

Directional filtering projects a scale-normalized Gaussian Hessian onto the normal
of each configured tangent. Scales double at every level. The wavelet implementation
uses an undecimated separable B-spline filter with taps `[1, 4, 6, 4, 1]/16`, dyadic
tap spacing and reflective boundaries. Horizontal, vertical and diagonal details plus
the final approximation sum to the input. The spline is symmetric and supports
shift-equivariant detail maps on the interior without decimation.

Log-Gabor uses zero-DC radial windows `exp(−log(f/fc)^2/(2b^2))` and axial Gaussian
angular weights. Central frequency halves at each level. The sampled experimental
Shearlet frame uses two overlapping frequency cones, dyadic radial centres and
triangular shear windows in slope, with angular spacing proportional to `2^(−j/2)`.
Windows are normalized pointwise by their root sum of squares, so analysis and
synthesis recover the input through a partition of unity. This is a sampled discrete
cone frame with periodic Fourier boundary conditions. Its coefficients are not
interchangeable with a reference discrete shearlet library's coefficients.

The detectors select maximum absolute band response, suppress transverse non-maxima
and link ridge pixels along the preferred tangent. Filter radius controls NoData
erosion. Fourier filters reflect-pad the raster and have global support, so NoData
fill and edge effects can extend beyond that exclusion margin. Iterating bands retains
a constant number of raster-sized arrays. Frequency methods take an FFT per band.
