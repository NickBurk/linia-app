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
