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

## PHASE 03 — обнаружение линеаментов

Доступны LSD, Canny + Probabilistic Hough и отдельный EDLines через OpenCV contrib. Алгоритм выбирается в интерфейсе; настройки Canny–Hough и EDLines независимы. Вычисления выполняются в фоновом потоке, запуск можно отменить. Результаты можно накладывать для сравнения и экспортировать вместе в GeoJSON с ID алгоритма, параметрами, координатами, длиной, азимутом, ориентацией и оценкой edge support.

Порог длины задаётся в метрах для Canny–Hough и EDLines и в пикселях предпросмотра для LSD. Вычисление остаётся ограниченным предпросмотром до 1600 пикселей по длинной стороне. Оценка edge support не является вероятностью или утверждением геологической точности. Синтетические тесты проверяют геометрию, CRS, NoData и чувствительность к параметрам; оценка по полевым данным ещё не выполнялась. См. [справочник алгоритмов](docs/algorithms.md).

## PHASE 04 — постобработка

Добавлены объединение близких коллинеарных концов с независимыми допусками по углу, расстоянию, коллинеарности и максимальному разрыву; фильтры по длине, осевому азимуту и confidence; дедупликация; классификация по направлению, длине, форме и пространственной плотности; и статистика с розой азимутов `[0, 180°)`. Confidence, которого нет у LSD, остаётся неизвестным и не приравнивается к нулю.

Панель постобработки и статистика доступны на русском и английском. Обработка выполняется в отменяемом фоновом потоке. Классы отображаются разными цветами и могут скрываться по отдельности. GeoJSON сохраняет исходную CRS и параметры, Shapefile записывает геометрию в исходной CRS и создаёт `.prj`, `.cpg` и `.metadata.json` для QGIS и полной provenance.

Расстояния и пороги длины постобработки задаются в пикселях предпросмотра; геодезическая длина и азимут остаются в метрах/градусах результата. См. [отчёт PHASE 04](docs/phase-04-report.md) для аудита, формул, результатов проверок и ограничений.

## GUI и проекты

Интерфейс PHASE 04.1 использует центральное поле карты с пропорциональным масштабом,
колесом мыши и перетаскиванием. Слои и свойства размещены в изменяемых боковых панелях;
настройки распределены по вкладкам данных, обнаружения, постобработки и статистики.
Файл проекта `.laproj` сохраняет путь к DEM, результаты алгоритмов и постобработки,
видимость классов и параметры отображения. Путь к последнему DEM и состояние окна
запоминаются через Qt Settings; автоматическое восстановление включается в меню «Файл».
См. [исправление загрузки](docs/dem-loading-fix.md) и [обновление GUI](docs/gui-redesign.md).

## Lineament Analyzer 1.0 RC

PHASE 05 adds A*, Segment Tracing, RANSAC, Directional Filters, Wavelet, Log-Gabor
and experimental Shearlet. Run individual detectors, compare methods, inspect unique
candidates and form Algorithm Consensus. Choose Preview or Full Resolution.
Vector export supports GeoJSON, Shapefile and GeoPackage.

PHASE 06 adds synthetic truth, user reference layers and a **Валидация / Validation**
workspace. Review matches, metrics and axial direction plots. Export experiment
reports in HTML, JSON or CSV.
[Русский справочник](docs/user-manual-ru.md) · [English guide](docs/user-manual-en.md)

[Core API](docs/architecture.md#independent-python-api) · [Windows release](docs/release-notes.md) · [Validation methodology](docs/validation-methodology.md)
