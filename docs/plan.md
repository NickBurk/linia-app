# Development plan

1. Establish packaging, CI, documentation, configuration and tests.
2. Build a localized Qt Widgets shell with raster viewport, menus and parameter controls.
3. Load GeoTIFF DEM with CRS, affine transform, NoData and bounded previews.
4. Calculate hillshade, slope, aspect and TPI with explicit horizontal and vertical units.
5. Implement and validate LSD with georeferenced output.
6. Add Canny–Hough, EDLines, A*, tracing, RANSAC, shearlet, wavelet/Log-Gabor and directional filtering as separately tested modules. Only expose completed methods in the UI.
7. Add segment merging, filters, classification and quantitative summaries.
8. Export GeoJSON and Shapefile with CRS and validation.
9. Validate against synthetic and documented real DEM datasets, report sensitivity and false detections.
10. Build a Windows executable in CI and test it on Windows.
11. Add a thin QGIS adapter around the Qt-independent core.

Acceptance requires meaningful numerical, geospatial and GUI tests at each stage. Large rasters should use overview or windowed display and tiled processing in later stages.

## PHASE 02 audit

The PHASE 02 audit covers CRS and units, conservative NoData support, windowed terrain processing, LSD/GeoJSON geometry and asynchronous GUI execution. Large DEM terrain processing now uses source-resolution windows and bounded previews. See [phase-02-audit.md](phase-02-audit.md) for verification and remaining limits. PHASE 03 detectors are implemented and validated on `feature/phase-03-lineament-algorithms`.

## PHASE 04 — postprocessing (active)

Working branch: `feature/phase-04-postprocessing`, based on `575e8cf` from `feature/phase-03-lineament-algorithms`. Keep `main` and the previous feature branches unchanged.

1. Audit the shared result models, affine/CRS conversions, detectors and exports.
2. Add Qt-independent segment merging, filters, duplicate removal, classification and axial statistics.
3. Add a localized GUI panel with cancellable background processing and per-class visibility.
4. Export provenance and derived attributes to GeoJSON and source-CRS Shapefile.
5. Add synthetic numeric/geospatial/NoData tests, documentation and report.
6. Complete Python 3.11 pytest/Ruff, Linux and Windows CI, PyInstaller, Windows frozen EXE self-test and ZIP artifact publication.

PHASE 05 must not begin until every PHASE 04 check has a recorded successful result or a clearly documented blocker. See [phase-04-report.md](phase-04-report.md) for the current implementation and validation record.

## PHASE 04.1 — DEM loading and GUI

Dedicated branch `fix/dem-loading-ui-ux` from PHASE 04. The work adds transactional
repeatable DEM loading, opt-in last-DEM restore, persistent Qt window/language settings,
versioned JSON projects, a docked map workspace, and lifecycle/project regression tests.
Windows frozen runtime verification and CI results are tracked in `docs/dem-loading-fix.md`
and `docs/gui-redesign.md`; do not begin PHASE 05 until the branch validation is recorded.

## PHASE 05 and PHASE 06

The release branch feature/phase-05-06-final-engine starts from PHASE 04.1 commit
87e4cf7. The code and workflow match runtime-tested b8d53d8.
[PHASE 04.1 CI](https://github.com/NickBurk/linia-app/actions/runs/45717568468)

- Add and test detectors, native/preview execution, parameter profiles and method agreement.
- Implement reference import, synthetic DEMs, spatial evaluation and reports.
- Integrate the Qt workspaces, GIS exports and frozen Windows acceptance flow.
- Record performance and scientific limits in the phase reports.
