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

The GUI lifecycle tests run offscreen with synthetic GeoTIFFs. Full test, Ruff, Linux CI,
Windows CI, and frozen EXE results are recorded after the branch workflow completes.
The exact original user failure remains unconfirmed because no original DEM, Windows EXE,
or reproduction trace was provided. Cancellation is cooperative: a single Rasterio read
or OpenCV call must return before the worker can stop. Session restore persists the DEM
and window preferences, not in-flight work.
