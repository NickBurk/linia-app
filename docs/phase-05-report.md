# PHASE 05 — extended detection engine

The release branch starts at 87e4cf7 on fix/dem-loading-ui-ux. GitHub Actions verified
the same source code at b8d53d8 through Linux tests, Windows tests and frozen EXE
execution, with an uploaded ZIP. The intermediate change only updated documentation.
[Baseline CI](https://github.com/NickBurk/linia-app/actions/runs/45717568468)

## Implemented

| Method | Model and controls | Computational cost and scope |
| --- | --- | --- |
| A* | Bounded eight-neighbour search over position, heading, gap and step count. Minimum length, expansion and step limits, turn, orientation and gap penalties, and support threshold. Map endpoints or automatic candidates. | Gradient field linear in pixels. Priority queue O(E log E) per bounded search, where E is the expansion budget and fan-out is bounded. Global grid. |
| Segment Tracing | Bidirectional gradient-tangent continuation, configurable length, curvature, support, gap and path count. Rejects loops and conservatively joins aligned tips over valid gaps. | Linear gradient computation, seed ordering and bounded local traces. Fragment joining is quadratic in the bounded returned path count. |
| RANSAC | Canny spatial groups, two-point hypotheses and total least-squares refinement. Residual tolerance, hypothesis count, inlier minimum, gap, output limit and seed. | O(SI) per pass for S sampled points and I hypotheses, plus sorted projected inliers. |
| Directional Filters | Scale-normalized Gaussian Hessian projected onto band normals. Angular range and step, initial scale, levels, response threshold, length and line count. | Separable Gaussian convolutions plus linear work per pixel per angle and scale. |
| Wavelet | Undecimated B-spline analysis with three detail components per level, final approximation and exact additive reconstruction. Levels and ridge extraction thresholds. | Separable dilated convolution. Smoothing cost grows with tap spacing in the current implementation. |
| Log-Gabor | Log-Gaussian radial response and axial angular weighting with zero DC. Central frequency, bandwidth, angular range/step, levels and extraction parameters. | FFT per band, O(N log N) for N raster cells. Global grid with reflection margins. |
| Shearlet | Experimental sampled frame with two frequency cones, dyadic radial scales, parabolic angular subdivision and Fourier reconstruction. Levels and extraction parameters. | FFT per band and a second window pass for normalization. Global grid. |

The unified engine and GUI retain LSD, Canny + Hough and EDLines. Canny and Hough
remain separately callable building blocks. Polyline candidates retain full vertices,
CRS, actual analysis affine and dimensions through projects and all vector exporters.
Postprocessing merges straight segments, keeps traced geometry and filters and
classifies both. Confidence retains each method's meaning. See
[formulas and limits](algorithms.md) and [method agreement](algorithm-comparison.md).

Preview uses the bounded displayed grid. Full-resolution runs check RAM before
allocating a full raster. LSD, Canny + Hough, EDLines, RANSAC, Directional Filters and
Wavelet can use overlapping windows. Global methods report the memory requirement
if the work cannot fit. Overlap limits tile edge effects, although tile-local
normalization and ownership can change results near seams. Terrain derivatives
are computed in worker threads and cached within the session.

## Profiling

Python 3.11 on Linux, default native dependencies and one OpenBLAS thread. The
profile raster contains three ridges and a base slope. Filter timings use two levels.
Tracing is limited to eight paths in the medium workload. The response column uses
the display grid, while detection uses the native grid. RSS is sampled process
memory and includes loaded libraries and caches. Retained allocations can affect
later rows. Exact timings depend on the machine.

    OPENBLAS_NUM_THREADS=1 python scripts/profile_detectors.py /tmp/lineament-profile --large

| Raster width | Method | Tile core (px) | Candidates | Response (s) | Detection (s) | Peak detection RSS (MiB) | Postprocess (s) | GeoJSON (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | lsd | — | 20 | — | 0.052 | 269.1 | 0.002 | 0.003 |
| 1024 | canny-hough | — | 78 | — | 0.035 | 261.0 | 0.006 | 0.007 |
| 1024 | edlines | — | 18 | — | 0.037 | 261.0 | 0.003 | 0.003 |
| 1024 | astar | — | 8 | — | 0.111 | 285.0 | 0.003 | 0.004 |
| 1024 | segment-tracing | — | 8 | — | 0.096 | 285.0 | 0.002 | 0.003 |
| 1024 | ransac | — | 36 | — | 0.184 | 261.0 | 0.004 | 0.004 |
| 1024 | directional | — | 48 | 0.317 | 0.505 | 352.4 | 0.039 | 0.026 |
| 1024 | wavelet | — | 48 | 0.158 | 0.269 | 369.2 | 0.045 | 0.022 |
| 1024 | log-gabor | — | 48 | 0.685 | 0.765 | 428.9 | 0.073 | 0.025 |
| 1024 | shearlet | — | 48 | 0.805 | 0.874 | 459.7 | 0.048 | 0.022 |
| 4096 | lsd | 1024 | 36 | — | 0.899 | 331.9 | 0.003 | 0.007 |
| 4096 | canny-hough | 1024 | 100 | — | 0.416 | 326.6 | 0.008 | 0.013 |
| 4096 | directional | 1024 | 247 | 0.636 | 7.078 | 502.7 | 1.584 | 0.146 |
| 4096 | wavelet | 1024 | 250 | 0.263 | 3.989 | 536.9 | 1.414 | 0.140 |
| 6144 | lsd | 1024 | 46 | — | 1.695 | 421.5 | 0.004 | 0.009 |

Profiling caught a subpixel LSD endpoint outside the raster-centre range. The
adapter constrains native endpoints before shared result validation. The desktop
acceptance flow also reproduced a reference-count crash in PySide6 6.12.0 during
repeated path rendering. Packaging stays below 6.12 pending a verified upstream
fix. Polyline overlays batch each geometry into a Qt polygon path.

The release workflow installs Python 3.11 and executes pytest, Ruff, dependency
checks and the application self-test on Linux and Windows. Windows packages the
application with PyInstaller, launches the bundled EXE using qwindows, reopens
the saved project in a second process and publishes a ZIP. The automated scenario
covers every detector, Hillshade, comparison, postprocessing, synthetic validation,
three GIS formats and project creation, save and reopen.
[Release branch CI](https://github.com/NickBurk/linia-app/actions?query=branch%3Afeature%2Fphase-05-06-final-engine)
