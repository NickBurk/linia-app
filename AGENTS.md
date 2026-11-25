# Lineament Analyzer

Use Python 3.11, PySide6, NumPy, SciPy, OpenCV, Rasterio, Pytest and Ruff. Keep the processing core independent of Qt. Preserve DEM masks, transforms, CRS, and units. Every displayed algorithm must be implemented and tested. Run `pytest` and `ruff check .` after changes. Use `QT_QPA_PLATFORM=offscreen` for headless GUI checks. Do not add QGIS or geological detection engines. Do not rewrite Git history or user data. Develop large features on dedicated branches.

PHASE 05/06 uses the engine module for detector registration, native/preview
execution and parameter profiles. Preserve full polyline vertices and analysis grid
metadata. Keep validation, vector I/O and reporting independent of Qt. Document
score semantics, resource limits and actual build/runtime evidence. Use the
installed and frozen --self-test for the desktop acceptance flow.
