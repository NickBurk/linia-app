# Lineament Analyzer

Use Python 3.11, PySide6, NumPy, SciPy, OpenCV, Rasterio, Pytest and Ruff. Keep the processing core independent of Qt. Preserve DEM masks, transforms, CRS, and units. Every displayed algorithm must be implemented and tested. Run `pytest` and `ruff check .` after changes. Use `QT_QPA_PLATFORM=offscreen` for headless GUI checks. Do not add QGIS or geological detection engines. Do not rewrite Git history or user data. Develop large features on dedicated branches.
