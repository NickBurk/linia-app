# Lineament Analyzer

Настольное приложение для просмотра GeoTIFF DEM и исследования потенциальных линеаментов. Русский интерфейс включён по умолчанию.

## Разработка

Python 3.11 required. Create a virtual environment and run `pip install -e '.[dev]'`. Start with `lineament-analyzer`. Run `pytest` and `ruff check .`.

The initial prototype loads a single-band DEM, displays it, calculates hillshade, slope, aspect and TPI, detects line segments with OpenCV LSD, and exports those segments to WGS84 GeoJSON. Detected lines are image candidates requiring geological interpretation.

## Status

See [architecture](docs/architecture.md) and [development plan](docs/plan.md). Windows builds run in GitHub Actions and publish an executable artifact. Windows runtime validation depends on that workflow.
