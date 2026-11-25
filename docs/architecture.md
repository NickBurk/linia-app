# Architecture

`data` owns DEM values, masks, band scale/offset, elevation units, CRS and affine transforms. `core` implements metric terrain math and windowed file processing independently of Qt. `algorithms` contains independent detectors and the shared `LineDetector` contract. Detectors return `DetectionResult` with frozen `LineSegment` records. Pixel endpoints retain the zero-based pixel-centre convention; source endpoints are transformed with the DEM affine and retain the DEM CRS. When a CRS is assigned, length and bearings use the CRS ellipsoid's geodesic between WGS84-transformed endpoints. Forward azimuth is clockwise from true north; axial orientation is in `[0, 180)`. Confidence is optional and must only be populated when a detector provides a meaningful score. `export` writes WGS84 GeoJSON with source metadata. `gui` owns Qt widgets, bounded previews and background task lifecycle.

`CannyHoughDetector` composes the independently reusable `canny.detect_edges` and `hough.detect_lines` operations. Canny normalizes valid DEM values to an 8-bit contrast range; the Hough component runs the probabilistic line transform. Canny and Hough use distinct parameter dataclasses. Minimum segment length and maximum gap are specified in metres, mapped to preview pixels using the local geometric-mean ground sample distance, and output lengths are checked again geodesically. Hough outputs crossing NoData neighborhoods are discarded. The edge support fraction is a diagnostic confidence score, not a calibrated probability.

`EdLinesDetector` is a separate implementation backed by OpenCV contrib's Edge Drawing `detectLines` API. It has independent sigma, gradient/anchor, path length, fit error, NFA validation, operator, scan interval and line separation settings. It performs its own normalization, passes a metric-derived minimum line length to EDLines and rechecks the geodesic output length. It shares only common result/georeferencing and generic NoData support helpers with other detectors; it never falls back to Canny–Hough when its backend is absent.

The GUI's detector controls, worker inputs and result storage are method-specific. Multiple method results can be overlaid or selected individually. Structured exports carry detector ID, parameters, confidence and support properties alongside source/pixel endpoints. Future A*, RANSAC, shearlet, wavelet, Log-Gabor and directional-filter implementations can register through `LineDetector` without adding Qt dependencies to the processing core. See [the algorithm reference](algorithms.md) for parameter units and limits.

Terrain gradients first differentiate along pixel columns and rows, then solve the transpose pixel-to-metric Jacobian. Projected units are converted through CRS axis conversion factors. Geographic rasters use ellipsoid radii of curvature evaluated at each pixel latitude. Elevation unit conversion precedes the user Z-factor. Aspect is downhill, clockwise from north, with flat terrain undefined. Projected derivatives use grid north and projection distances. Geographic derivatives use local ellipsoidal east/north distances. Projection distortion and grid convergence are properties of the selected projected CRS.

Gradient stencils require finite valid samples and use one-sided differences only at actual raster borders. TPI subtracts the complete square neighbourhood mean, including the centre. Unsupported windows and NoData remain NaN. No extrapolation or filling is applied.

`core.windowed.process_dem` reads clipped halo windows, computes local derivatives, crops the halo and writes disjoint destination windows. Halo is one pixel for gradients and half the TPI window size. CRS, transform and shape are preserved. A temporary TIFF in the destination directory is atomically replaced after success. Failure and cancellation remove the temporary output. The budget estimates 256 bytes per halo pixel and separately bounds GDAL cache to 16 MiB. Library baseline, decoder allocations and GUI memory contribute to process RSS.

GUI terrain calculations run at source resolution on disk. Only bounded nearest-neighbour previews enter the viewport and LSD. The preview transform includes its resolution change. A QThread task receives immutable operation inputs, emits progress, warnings, errors and results through queued signals, and checks a threading event for cooperative cancellation. Widgets and QPixmap are touched only in the GUI thread. Closing requests cancellation and defers destruction until thread completion. OpenCV and single Rasterio calls complete before cancellation is observed.

LSD endpoints use zero-based image pixel-centre coordinates. Display adds half a pixel to match QGraphicsPixmapItem edges. Export applies the same half-pixel shift and full affine transform, then transforms source coordinates to EPSG:4326. GeoJSON retains source WKT as a foreign metadata member and source/pixel endpoints, method parameters, pixel length, WGS84 geodesic length and axial orientation as feature properties. The GeoJSON geometry uses longitude/latitude per RFC 7946.

PyInstaller hooks supply Qt platform plugins. Explicit Rasterio and pyproj collection supplies dynamically imported modules and GDAL/PROJ resources. CI exercises dependencies, numerical and GUI tests, installed self-test, frozen EXE self-test and ZIP artifact upload on Windows with Python 3.11.

## Postprocessing

`algorithms.postprocessing` is independent of Qt and consumes the shared immutable `DetectionResult`. It performs greedy iterative endpoint merges with separate axial-angle, endpoint-distance, along-line maximum-gap and collinearity tolerances. Proper interior crossings are preserved by default. Pixel-space tolerances are explicit because detection runs on a bounded preview and results may have no CRS. Rebuilt segments retain the source affine/CRS, detector ID and detector parameters; merged confidence is the mean of available scores and remains undefined when none were supplied.

Filters use preview-pixel length limits, an optional axial azimuth interval and an optional confidence threshold. A missing confidence value is left unfiltered. Duplicate removal compares forward and reversed endpoints within a configurable tolerance. Classification attaches a direction bin, length class, orientation-derived geometry class and density cluster label to each feature. Spatial clusters use centroid distance and a DBSCAN-style minimum-neighbour count.

Statistics count orientations in `[0, 180)` using the segment's axial orientation, so reversing endpoints does not create a second directional bin. They include a length histogram, direction bins, count, complete metric total/mean/median length when all lengths are known, footprint density per km² when a valid CRS footprint is available, and counts by combined class. No-CRS results still expose pixel-based classification and orientation, while metric lengths and density remain unknown.

GUI postprocessing and statistics use the existing cancellable QThread task. Class overlays are colored by stable class ID and each class can be hidden independently. GeoJSON stays RFC 7946 WGS84 and retains source CRS metadata. Shapefile geometry is written in the source CRS with `.prj` and UTF-8 `.cpg` sidecars; a `.metadata.json` companion preserves complete parameters and provenance beyond the DBF width limit.

## GUI state and projects

`gui.project` serializes versioned `.laproj` JSON documents. Detection and postprocessed
results retain pixel/source endpoints, CRS WKT, transform, shape, optional confidence,
parameters and class properties. Project opening reads and validates JSON before starting
a DEM worker; results are attached only after the raster is prepared and checked against
their stored shape, transform and CRS. Writes use a sibling temporary file followed by
an atomic replace. `QSettings` stores last DEM, language, window geometry/state, selected
algorithm and the opt-in automatic DEM restore preference. Raster previews are loaded
inside `rasterio.open` context managers; failed/cancelled workers never deliver a partial
preview to the GUI, so the currently displayed DEM remains intact.

The main window puts `MapCanvas` in the centre and uses dock widgets for layers and
properties. The map keeps aspect ratio, zooms around the mouse wheel and supports drag
panning. Layer and processing controls remain in Qt widgets, while heavy raster and
algorithm operations retain the existing cancellable `Task` lifecycle.
