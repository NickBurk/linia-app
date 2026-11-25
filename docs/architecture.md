# Architecture

`data` owns raster loading, metadata and bounded previews. `core` owns terrain math. `algorithms` exposes independent detection modules through typed parameters. `gui` owns localization and Qt widgets. Future `postprocessing`, `export` and `qgis_adapter` use core data types without introducing Qt dependencies.

Pixel geometry uses Rasterio affine transforms. Numeric terrain operations require projected horizontal units or an explicit conversion; vertical elevation units are supplied by the user. NoData pixels stay masked throughout derivatives and detection. Detection results retain source coordinates, CRS and method parameters. Geologic interpretation remains the analyst's responsibility.

PySide6 is isolated at the application boundary so command-line processing, tests and a later QGIS adapter can reuse the same algorithms. The first detector uses OpenCV's image-processing LSD primitive and does not claim a custom geological detector.
