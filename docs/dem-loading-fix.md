# DEM loading diagnosis and fix

## Diagnosis

The reported second-launch failure could not be reproduced from the repository state or
without the user's affected raster and launch sequence. Source audit found that both full
DEM reads and bounded previews already use `with rasterio.open(...)`, so `DatasetReader`
handles are closed when each read completes. Background workers deliver results through
queued Qt signals; failures and cooperative cancellation do not call the GUI result
receiver. The previous GUI had no persisted last-DEM/session state, so a later launch
always started empty and required another manual file selection. That explains the
missing automatic restore, but does not prove it was the user's reported display failure.

## Changes

- Normalize selected paths to absolute paths and reject missing paths before worker start.
- Build the preview/image completely in the worker before `accept_layer` swaps the active
  DEM and map. Failed and cancelled reads preserve the current valid layer.
- Reopening the same path is supported; opening another valid path clears results tied to
  the former raster. Rasterio resources stay scoped to their context managers.
- Store last DEM, language, window geometry/dock layout, active algorithm and an opt-in
  automatic restore flag with `QSettings`. A missing last path produces a status message;
  it is not opened automatically.
- Add sequence tests for first and repeated opens, two different DEMs, bad input retaining
  the previous DEM, and reopening after the failure. The general worker tests cover cancel,
  task completion and close-during-task behavior.

## Verification and limits

The offscreen synthetic GeoTIFF suite passed: **130 tests**, with three expected
`UnitWarning`s from terrain tests where the test DEM has no declared units or CRS. Ruff
reported clean. The Linux job passed dependency checks, pytest, Ruff and the installed
application self-test. The Windows job passed those checks, built the PyInstaller
application, verified `qwindows.dll`, ran the frozen EXE self-test and uploaded the ZIP.
The self-test opens the same DEM twice, switches to another DEM, and exercises detectors
and GeoJSON export in the frozen environment.

GitHub Actions run: [Checks #14](https://github.com/NickBurk/linia-app/actions/runs/45716420297)
(commit `ccc8eb4`). Windows artifact: [LineamentAnalyzer-Windows.zip](https://github.com/NickBurk/linia-app/actions/runs/45716420297/artifacts/15070042431)
(189,412,916 bytes; SHA-256 `012ebc789fb4a7d4e738be777cae9cd2243bafa82ad97611aed30f8b467cb95f`).

The exact original user failure remains unconfirmed because no affected DEM, prior EXE,
or reproduction trace was provided. The new EXE's automated lifecycle test passed, but
the exact user file and interactive two-session Windows workflow were not available for
testing. Cancellation is cooperative: a single Rasterio read or OpenCV call must return
before the worker can stop. Session restore persists the DEM and window preferences,
not in-flight work.
