# Lineament Analyzer 1.0 release candidate

- Added A*, Segment Tracing, RANSAC, Directional Filters, Wavelet, Log-Gabor and experimental Shearlet.
- Added resource checks, native detection with halo windows for compatible methods and explicit analysis grid provenance.
- Added presets, map-picked A* endpoints, filter-component views and Algorithm Consensus.
- Added synthetic terrain, reference imports, spatial validation, charts and HTML/JSON/CSV reports.
- Added GeoPackage export and full polyline geometry across vector formats.

## Windows installation

Download **LineamentAnalyzer-Windows** from the successful release branch run under
GitHub Actions Artifacts. Extract the GitHub delivery archive and its application
ZIP. Run LineamentAnalyzer.exe and keep the companion _internal directory alongside
it. The Windows job uses the native Qt Windows platform for the frozen acceptance
flow, including a new process that reopens the saved project.
[Release branch Actions](https://github.com/NickBurk/linia-app/actions?query=branch%3Afeature%2Fphase-05-06-final-engine)

Global FFT and path solvers need an in-memory grid. Tile results may differ at seams.
Shearlet is an experimental sampled Fourier cone frame. Synthetic metrics measure
planted geometries. Confidence remains uncalibrated.
[PHASE 05](phase-05-report.md) · [PHASE 06](phase-06-report.md) · [Methodology](validation-methodology.md)
