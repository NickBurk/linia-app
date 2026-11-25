# Lineament Analyzer

Настольное приложение для просмотра GeoTIFF DEM и исследования потенциальных линеаментов. Русский интерфейс включён по умолчанию.

## Разработка

Python 3.11 required. Create a virtual environment and run `pip install -e '.[dev]'`. Start with `lineament-analyzer`. Run `pytest` and `ruff check .`.

The initial prototype loads a single-band DEM, displays it, calculates hillshade, slope, aspect and TPI, detects line segments with OpenCV LSD, and exports those segments to WGS84 GeoJSON. Detected lines are image candidates requiring geological interpretation.

## Status

See [architecture](docs/architecture.md) and [development plan](docs/plan.md). Windows builds run in GitHub Actions and publish an executable artifact. Windows runtime validation depends on that workflow.

## PHASE 02 — технический аудит

Расчёт Slope, Aspect и Hillshade учитывает Affine, единицы проекции и высот, Z-factor и широту географического DEM. Единицы высот читаются из GeoTIFF, доступны ручной выбор и Z-factor. При неизвестных единицах появляется предупреждение с указанием принятого допущения о метрах.

Производные больших DEM рассчитываются по Rasterio windows с halo в исходном разрешении. GUI показывает предпросмотр до 1600 пикселей по длинной стороне. LSD работает на отображаемом предпросмотре, поэтому минимальная длина задаётся в его пикселях. Экспорт использует transform предпросмотра и сохраняет параметры анализа. Для анализа массива в исходном разрешении доступен `algorithms.lsd.detect`.

Загрузка, производные, LSD и экспорт выполняются в QThread. Кнопка отмены останавливает оконный расчёт между блоками. Вызов OpenCV завершается до обработки отмены, его результат при отмене отбрасывается. Закрытие окна дожидается безопасного завершения задачи.

Проверка установленного приложения, GDAL/PROJ и Qt доступна через `lineament-analyzer --self-test`. В Windows workflow эта же проверка запускается на собранном EXE до создания ZIP. Скачать ZIP можно на странице запуска Actions в разделе Artifacts после успешной задачи `windows-exe`.

Подробные формулы, проверки, ограничения и перечень изменённых файлов находятся в [отчёте PHASE 02](docs/phase-02-audit.md).
