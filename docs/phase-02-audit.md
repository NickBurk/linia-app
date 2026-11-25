# PHASE 02 technical audit

The baseline is commit `2c3f93b` on `feature/phase-02-dem-audit` in `NickBurk/linia-app`. The audit continues on that branch with ordinary commits.

## Numerical and geospatial checks

| Area | Implementation | Evidence |
| --- | --- | --- |
| CRS and units | Full affine Jacobian, projected linear units, ellipsoidal geographic scale by latitude, band scale/offset, elevation units and user Z-factor | Synthetic planes, rotated grids, metre/foot conversions, four cardinal aspects, latitudes 0/45/75 degrees, explicit unit override and invalid Z-factor tests |
| NoData and borders | Finite valid derivative stencils, one-sided true borders, complete TPI support and no edge extension | Plane with holes and nonfinite samples, TPI support mask, constant derivatives through valid edges |
| Large DEM | Window reads, halo/crop/write, atomic output, memory estimate and GDAL cache limit | Whole-array equivalence for four derivatives and three tile sizes, explicit bounded-read assertions, preview transform, cancellation and injected partial failure |
| LSD and GeoJSON | Length filtering, mask-support rejection, sparse-line normalization, pixel-centre transform, EPSG:4326 export and attributes | Horizontal, vertical and diagonal known lines, mask boundary, three source CRSs, finite JSON and preservation on invalid export |
| GUI | QThread task, progress, cancellation event, queued results and errors, deferred close | Event-loop heartbeat during a blocked worker, GUI-thread receiver assertion, error/cancel tests, close during task, real load/terrain/LSD flow |
| Windows | Python 3.11 on windows-latest, PyInstaller, Qt plugin check, GDAL/PROJ collection, frozen self-test, ZIP upload | The workflow must complete on the audit commit before the current build can be declared successful |

Local verification uses Python 3.11.18. `QT_QPA_PLATFORM=offscreen pytest -q` passed 52 tests. `ruff check .`, import checks, compileall, dependency compatibility checks and the installed application self-test passed. Three warnings in baseline terrain tests exercise inputs with missing units/CRS. Dedicated tests explicitly assert these warnings.

A source-resolution geographic DEM of 4096×4096 was generated and processed using 512-pixel requested blocks and a 32 MiB operation budget. The budget reduced the actual tile size to fit halo intermediates. Slope took 1.84 seconds on this Linux environment. Peak process RSS was 151.82 MiB, growing 40.86 MiB above the loaded-library baseline. RSS includes interpreter, native libraries, GDAL cache and allocators. These measurements are observations from this environment.

## Conventions and limits

- Elevation units support metres, international feet, US survey feet, centimetres and millimetres. Unknown units produce a warning and assume metres. GUI offers an override. Missing horizontal CRS warns and assumes metre map coordinates. GeoJSON export requires an assigned source CRS.
- Geographic scale uses the local differential metric of the CRS ellipsoid. Finite differences approximate the sampled surface. Polar pixels and geographic angular units other than degrees raise a descriptive error. Projected aspect is relative to grid north. TPI is in metres and uses an odd square pixel window.
- Gradients require at least two rows and columns. TPI leaves a half-window NoData border and expands holes by that radius. Flat aspect is undefined.
- Memory limits bound estimated processing intermediates. They do not cap total process RSS. Output disk space scales with source dimensions. Preview reads use nearest-neighbour sampling, preserving the sampled mask and spatial footprint.
- GUI preview and LSD are bounded to 1600 pixels on the longest side. Segment lengths in pixels refer to that preview. The preview shape, source endpoints and georeferencing remain in the exported result. Full-resolution array LSD is available through the core detector. Full-resolution tiled LSD merging is outside this terrain audit.
- Cancellation is cooperative between raster windows and before/after native calls. Closing waits for a current native call. Indeterminate progress is shown for operations without measurable substeps.
- Synthetic tests establish numerical and coordinate conventions. Geological interpretation and validation against real surveyed terrain require separate datasets.

## GitHub verification

The inherited run `45691861915` completed with a successful Windows packaging job and failed Linux tests. That result belongs to the baseline commit. CLI `gh auth status` reported an invalid GH_TOKEN. The connected GitHub plugin successfully read Actions run and job metadata, providing an authenticated alternative for checking results. Current audit workflow verification is recorded after pushing the implementation.

## Changed files

- `.github/workflows/checks.yml`
- `README.md`
- `docs/architecture.md`
- `docs/phase-02-audit.md`
- `docs/plan.md`
- `pyproject.toml`
- `src/lineament_analyzer/algorithms/lsd.py`
- `src/lineament_analyzer/app.py`
- `src/lineament_analyzer/core/terrain.py`
- `src/lineament_analyzer/core/windowed.py`
- `src/lineament_analyzer/data/dem.py`
- `src/lineament_analyzer/export/geojson.py`
- `src/lineament_analyzer/gui/tasks.py`
- `src/lineament_analyzer/gui/window.py`
- `src/lineament_analyzer/selftest.py`
- `tests/test_crs.py`
- `tests/test_gui_tasks.py`
- `tests/test_lsd_geojson.py`
- `tests/test_nodata.py`
- `tests/test_prototype.py`
- `tests/test_windowed.py`
