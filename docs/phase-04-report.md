# PHASE 04 — постобработка линеаментов

## Область и исходное состояние

- Репозиторий: `NickBurk/linia-app`.
- Рабочая ветка: `feature/phase-04-postprocessing`.
- База: `575e8cf083ce2ad6235637f35c8cb2df3cf178a6` от `feature/phase-03-lineament-algorithms`.
- `main` и предыдущая feature-ветка не изменялись.
- До разработки изучены `AGENTS.md`, `docs/architecture.md`, `docs/plan.md`, `docs/phase-03-plan.md` и `docs/algorithms.md`.

PHASE 03 предоставляет неизменяемые `DetectionResult`/`LineSegment`, zero-based pixel-centre endpoints, affine/CRS метаданные, geodesic длины и ориентации, optional confidence, GeoJSON и отменяемый `QThread` worker. Исходные результаты детекторов остаются доступными отдельно от постобработанных.

## Реализовано

### Алгоритмический модуль

`lineament_analyzer.algorithms.postprocessing` не зависит от Qt и предоставляет:

- независимые лимиты угла, расстояния между концами, коллинеарности и продольного разрыва;
- консервативное соединение близких сегментов с сохранением proper interior intersections;
- фильтры минимальной/максимальной пиксельной длины, осевой ориентации и optional confidence;
- дедупликацию совпадающих и обращённых концами сегментов;
- сохранение `None` confidence без приравнивания к нулю; при слиянии известных оценок вычисляется их среднее;
- классификацию по направлению, длине, ориентации/геометрии и плотностному кластеру;
- count, суммарную/среднюю/медианную geodesic длину, footprint density per km², distribution длин, axial rose и статистику классов;
- cooperative cancellation в тяжёлых циклах.

Distance/length limits заданы в пикселях bounded preview, чтобы они работали без CRS. Геодезические поля результата остаются в метрах и градусах. Rose histogram охватывает `[0, 180°)`.

### GUI

Русско-английская панель даёт настройки merge/filter, классификацию, статистику, цветные классовые overlays и чекбоксы видимости групп. Постобработка и статистика выполняются существующим background task; отмена и закрытие окна используют прежний жизненный цикл задач. Raw и postprocessed результаты выбираются раздельно.

### Экспорт

- GeoJSON продолжает записывать геометрию WGS84/RFC 7946 и добавляет source algorithm, object ID, class ID, параметры детектора и постобработки.
- Shapefile записывает source-CRS geometry, `.prj`, UTF-8 `.cpg`, метрики, confidence и class ID.
- Companion `.metadata.json` сохраняет полный CRS, affine, dimensions, detector/postprocessing параметры и все feature properties сверх лимита DBF-полей.

## Тестирование

Синтетическое покрытие включает слияние разрывов, независимые расстояние/разрыв/угол/коллинеарность, пересечения и разнонаправленные линии, шумовые короткие сегменты, длины, confidence missing/known, дубликаты/reversed duplicates, axial statistics, density/classes, повёрнутые аффинные трансформы, projected/geographic/no-CRS результаты, GeoJSON/Shapefile round-trip и GUI worker/visibility. Существующие PHASE 02/03 тесты сохраняют проверки CRS, NoData, detector geometry и совместимости.

Фактические результаты локальных проверок будут записаны после итогового прогона:

- Python: 3.11.18.
- `QT_QPA_PLATFORM=offscreen pytest -q`: 127 passed, 3 existing unit/CRS assumption warnings.
- `ruff check .`: passed.
- `uv pip check`: all installed packages compatible.
- `QT_QPA_PLATFORM=offscreen python -m lineament_analyzer.app --self-test`: exit code 0.
- Linux PyInstaller `--onedir` build: completed; frozen Linux `--self-test`: exit code 0.
- Linux GitHub Actions: ссылка и job conclusion — добавить после push.
- Windows GitHub Actions, PyInstaller, frozen EXE self-test и ZIP artifact: ссылка и artifact — добавить после завершения workflow.

Проверки, которые не запускались или не завершились успешно, не считаются пройденными.

## Изменённые файлы

Итоговый перечень и SHA исходного commit будут зафиксированы после проверки ветки.

## Ограничения

- Обработка и классификация основаны на ограниченном DEM preview; пространственные merge/cluster thresholds задаются в preview pixels.
- Confidence разных алгоритмов не откалиброваны между собой. Отсутствующие значения остаются неизвестными.
- Плотностная классификация использует расстояния между центроидами отрезков; это операционная группа, не геологическая интерпретация.
- Footprint density per km² использует геодезическую площадь четырёх углов растра; для очень больших/высокоширотных растров это оценка.
- Для Shapefile сверх стандартных sidecar файлов необходим `.metadata.json`, чтобы сохранить полные параметры без обрезания ограниченными DBF-полями.
- Проверки синтетические; геологическая точность на полевых/эталонных данных и качество ручной интерпретации не оценивались.
- PHASE 05 не начинается до завершения всех перечисленных PHASE 04 проверок.
