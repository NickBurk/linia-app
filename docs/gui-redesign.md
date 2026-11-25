# GUI redesign

The central `MapCanvas` is the main workspace. It preserves raster aspect ratio, zooms
with the wheel around the pointer, supports hand panning, and can fit the current DEM to
view. Resizable left and right docks contain the available layer list and four property
tabs: Data, Detection, Postprocessing and Statistics. Existing detector, terrain,
postprocessing, class visibility and GeoJSON/Shapefile export functions remain connected.
The toolbar exposes open, save project, zoom, detection, postprocessing and export actions.
The status area reports the active file, CRS, raster dimensions and task progress.

Russian remains the default interface; English is available through the language selector.
The selected language, window size/position, dock state and algorithm are stored with
Qt Settings. Users can opt in to automatic restore of the last DEM.

`.laproj` is versioned JSON. It contains the DEM's absolute path, stored results and
postprocessing results, class visibility, and display selections. Result geometry is
restored only when raster shape, affine transform and CRS match. Project files do not copy
or embed source DEMs, so moving a project to another computer requires the original raster
at its saved path. The current Data tab shows preview dimensions, CRS, resolution and
valid elevation range; band count and explicit NoData metadata are not yet displayed.

The layer list contains the active DEM/terrain view and available detector result entries.
It is not yet a full GIS layer manager: per-layer checkboxes, context actions, cursor
coordinates and a numeric scale readout remain future improvements. Help text is currently
brief, and some dialog titles are still fixed strings. Linux and Windows jobs passed,
including the frozen Windows self-test; links and the artifact are recorded in
[`dem-loading-fix.md`](dem-loading-fix.md).
