# Lineament Analyzer 1.0 user guide

## Data and projects

1. Open a single-band GeoTIFF in **Data**. Check CRS, resolution and elevation units.
2. In **Terrain analysis**, select elevation, hillshade, slope, aspect or TPI. Set elevation units and Z-factor when needed.
3. Scroll to zoom, drag to pan, or use **Zoom to data** to fit the DEM.
4. Save a .laproj file from **File**. Reopening a project uses the original GeoTIFF at its saved path. Turn on automatic restore if you want the last DEM at launch.

## Detection and comparison

1. Select an algorithm and profile in **Detection**. Preview uses the displayed grid. Full Resolution uses the native grid and checks the memory budget. Global methods may require a smaller analysis area.
2. For interactive A*, use **Pick two A* endpoints** and select start and end on the map. Both need raster support. **Detect** runs the automatic endpoint search.
3. Use **Show response** to inspect filters. Component 0 shows aggregate strength or preferred orientation. Positive component numbers display signed bands. Reapply the terrain layer to return to its image.
4. Select methods in **Compare results** and run a batch. Select individual methods, unique candidates or Algorithm Consensus in the results list. The consensus table identifies contributing methods per object.
5. Configure length and joining thresholds in pixels of the actual analysis grid. Review derived results, statistics and class visibility.

## Validation and export

1. In **Validation**, choose a DEM and reference GeoJSON, Shapefile or GeoPackage line layer. When prompted for CRS, provide its confirmed EPSG identifier or WKT. Generate a synthetic dataset to exercise the workflow.
2. Select methods, matching tolerance in metres and angular tolerance. Add postprocessing and consensus if needed. View metrics, the match map, orientation errors and axial rose plots.
3. Export HTML or JSON for a full report and CSV for the metric table. Save the dataset descriptor to repeat the study.
4. Select result layers and choose GeoJSON, Shapefile or GeoPackage in **Export**. GeoJSON uses WGS84. Shapefile and GeoPackage use the DEM CRS. Keep the Shapefile sidecars with the geometry file.

Extracted lines are candidates for geological interpretation. Shearlet is experimental.
Confidence reflects method-specific support. Synthetic accuracy quantifies recovery of planted geometry.
