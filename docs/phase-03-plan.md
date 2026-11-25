# PHASE 03 implementation plan

## Starting point and constraints

- Repository: `NickBurk/linia-app`.
- Source branch: `feature/phase-02-dem-audit`.
- Required source commit: `64e37e0` (`Record verified Windows executable and CI audit evidence`).
- Working branch: `feature/phase-03-lineament-algorithms`, created directly at the required commit.
- PHASE 02 already provides `Dem` with validity mask, affine transform and CRS; a bounded nearest-neighbour preview; LSD; pixel-centre coordinate export to WGS84 GeoJSON; Qt worker tasks with cooperative cancellation; and Linux/Windows CI including the Windows frozen self-test and ZIP artifact.
- Preserve existing imports, Nx4 LSD calls, export output fields and GUI workflow where practical. Do not edit `main` or the PHASE 02 branch.
- Repository instructions require Python 3.11. The current execution runtime is Python 3.12 and does not have OpenCV installed, so establish a compatible validation environment before making numerical claims.

## Phase 0 — baseline and architecture review

1. Confirm branch, clean worktree, exact base SHA and remote.
2. Trace DEM loading, preview affine scaling, NoData masking, LSD detector, export coordinate convention, Qt task lifecycle, current synthetic tests and CI packaging.
3. Record compatibility constraints and detector backend options for EDLines, including whether the chosen maintained OpenCV distribution exposes `ximgproc.EdgeDrawing` / line detection in both Linux and Windows wheels.
4. Keep this document as the reviewed execution checklist. No PHASE 03 detector work starts until the shared coordinate and result contract is fixed.

**Review gate:** clean branch at `64e37e0`; each existing behavior has a mapped owner and regression test.

## Phase 1 — shared detector and geospatial result contract

1. Introduce a Qt-independent detector protocol/base contract with stable algorithm ID, parameter validation, cancellation checkpoints and a common `detect(dem_or_image, parameters, progress, cancel)` entry point.
2. Add an immutable segment/result model. Pixel endpoints use the established zero-based pixel-centre convention; each segment also records source-coordinate endpoints, metric length, forward azimuth, axial orientation, optional normalized confidence, algorithm ID and JSON-safe detector-specific properties.
3. Derive spatial coordinates from the exact `Dem.transform` and source CRS. Use a documented geodesic metric convention for length and azimuth; reject missing/invalid spatial references when geographic output is requested. Preserve CRS metadata and serialize non-finite values safely.
4. Keep adapters for the legacy LSD Nx4 API and existing GeoJSON tests/consumers. Add tests for north-up, rotated/sheared, anisotropic, projected and geographic transforms, pixel-centre offsets, axis order, NoData, and compatibility.

**Review gate:** one shared, tested result contract serves LSD without changing existing results unexpectedly; processing package remains independent of Qt.

## Phase 2 — Canny and probabilistic Hough detector

1. Add a self-contained Canny–Hough detector module and immutable parameters for contrast normalization, Canny low/high thresholds, aperture/L2 mode, Hough vote threshold, minimum physical segment length, maximum physical gap, and minimum output length.
2. Normalize only finite valid DEM samples. Prevent NoData boundaries/fill from becoming detections and reject candidates whose support crosses invalid pixels. Check cancellation before and after non-interruptible OpenCV calls.
3. Convert map-unit length/gap settings to preview pixel units using the local affine/CRS metric scale; measure and filter returned lines using transformed coordinates. Document that preview resolution bounds computational detail while georeferencing remains exact.
4. Build deterministic synthetic grayscale/DEM fixtures covering horizontal, vertical, diagonal, crossing, parallel, noisy, NoData-gapped and short/long lines. Assert angular error, endpoint/length tolerance, false detections at mask edges, and geospatial coordinates.

**Review gate:** detector has stable, parameterized behavior, no mask-edge artifacts in fixtures, and tests at more than one pixel size.

## Phase 3 — independent EDLines detector

1. Confirm and pin a cross-platform backend that provides the actual EDLines / EdgeDrawing line implementation. Do not implement EDLines by reusing Canny–Hough internals, and do not silently fall back to another detector.
2. Add a distinct module, parameter type, algorithm ID, output metadata and cancellation handling. Keep preprocessing and mask policy explicit and testable.
3. Reuse the same synthetic geospatial fixtures where meaningful, with independent expected behavior/tolerances and explicit tests that the selected backend is invoked.

**Review gate:** EDLines is independently selectable and tested on Linux and Windows; missing backend produces a clear installation/runtime error, never fabricated results.

## Phase 4 — result export and GUI integration

1. Extend export to the shared result model while retaining the existing LSD GeoJSON schema compatibility. Include algorithm ID, parameters, source/pixel endpoints, length, azimuth, axial orientation and confidence where available.
2. Add a localized algorithm selector, algorithm-specific validated controls, run/cancel state, progress/status reporting, and safe empty/error handling.
3. Run all detector work in the existing background-task framework; freeze input arrays/parameters for each task and honor cooperative cancellation before publishing results.
4. Store results by algorithm/configuration and allow switching overlays and comparing at least two result sets without replacing the DEM or confusing their provenance. Add GUI event-loop, cancellation, parameter-change, rendering and comparison tests.

**Review gate:** Russian and English UI exposes only implemented algorithms; GUI remains responsive and results are attributable to their detector/configuration.

## Phase 5 — compatibility, docs and release validation

1. Document detector assumptions, parameter units, orientation conventions, NoData policy, resolution behavior, confidence meaning, output properties and known limitations. Update architecture and README.
2. Run full pytest and Ruff in Python 3.11; fix regressions. Add CI assertions for EDLines backend availability and the new tests on Linux and Windows.
3. Verify the Windows PyInstaller build includes native OpenCV resources, run the frozen self-test, and upload the Windows ZIP. Confirm Linux and Windows Actions results for the feature branch.
4. Record actual commits, test totals, Actions URLs, ZIP artifact link, and any unsupported validation. Do not claim geological accuracy beyond synthetic measurements; report false-positive/real-data limits.

**Review gate:** all stated checks have evidence, artifacts are tied to the reviewed branch commit, and no validation is described as passing without a successful result.

## Extensibility rules

- Future A*, RANSAC, shearlet, wavelet, Log-Gabor and directional-filter methods register through the same detector contract and return the same segment/result type.
- Detector-specific tuning stays in dedicated parameter dataclasses/modules. A future method must not become selectable until implemented and tested.
- Keep raster access, coordinate conversion, export and Qt orchestration outside detector-specific algorithms where possible.

## Current status

- Phase 0 review: complete. The installed base `opencv-python-headless` wheel exposes LSD but not `ximgproc`; Phase 3 will therefore evaluate the contrib wheel before EDLines implementation.
- Phase 1 shared contract: complete. Existing and new tests pass (59 total); Ruff passes. Commit: `094ef32`.
- Phase 2 Canny and Probabilistic Hough: implemented as standalone components and exposed as a selectable Canny–Hough line pipeline. Synthetic coverage includes direct component tests and geospatial end-to-end cases. Initial full suite: 76 passed, Ruff passed. Component separation was added during final review and is being rechecked.
- Phase 3 EDLines: complete. Independent detector, contrib dependency, real-backend synthetic tests and explicit missing-backend behavior are implemented. Full suite: 99 passed; Ruff and installed self-test passed. Commit: pending.
- Phase 4 GUI/export: implemented and covered by localized workflow, comparison, cancellation, geospatial export, provenance and legacy compatibility tests. Full suite currently: 103 passed; Ruff passed. Final stage commit is pending.
- Phase 5 release validation: not started.
