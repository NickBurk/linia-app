"""End-to-end reproducible detector experiments without Qt."""

from lineament_analyzer.algorithms.postprocessing import postprocess_result
from lineament_analyzer.core.windowed import check_cancel
from lineament_analyzer.engine import measure, run_methods
from lineament_analyzer.validation.comparison import compare_results, evaluate
from lineament_analyzer.validation.datasets import load_reference
from lineament_analyzer.validation.report import make_report


def run_experiment(dataset, source, configurations, matching, *, postprocessing=None,
                   consensus=False, progress=None, cancel=None, **options):
    """Evaluate selected algorithms, optionally including postprocessed and consensus results."""
    if dataset is not None and dataset.dem_path != str(source):
        raise ValueError("The validation dataset DEM differs from the analysis source")
    reference = None
    if dataset is not None:
        reference = load_reference(dataset.reference_path, source, layer=dataset.reference_layer,
                                   confirmed_crs=dataset.confirmed_crs, cancel=cancel)

    def detection_progress(value):
        if progress:
            progress(round(0.75 * value))

    raw, timings = run_methods(source, configurations, progress=detection_progress,
                               cancel=cancel, **options)
    unique = {}
    if consensus and len(raw) >= 2:
        (combined, unique), timings["consensus"] = measure(
            lambda: compare_results(list(raw.values()), matching, cancel=cancel)
        )
        raw["consensus"] = combined
    post = {}
    if postprocessing is not None:
        for label, result in raw.items():
            check_cancel(cancel)
            post[f"{label} · post"], timings[f"{label} · post"] = measure(
                lambda result=result: postprocess_result(result, postprocessing, cancel=cancel)
            )
    results = raw | post
    evaluations = {}
    for index, (label, result) in enumerate(results.items(), start=1):
        check_cancel(cancel)
        evaluations[label] = evaluate(result, reference, matching, cancel=cancel)
        if progress:
            progress(75 + round(25 * index / len(results)))
    return raw, post, unique, make_report(results, evaluations, matching,
                                          timings=timings, dataset=dataset)
