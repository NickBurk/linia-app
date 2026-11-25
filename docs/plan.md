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
