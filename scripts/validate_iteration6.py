"""Independent integrity checks for Iteration 6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from tuac.cascade import CascadeProbabilityArtifact, uncertainty_scores
from tuac.data import load_jsonl
from tuac.metrics import binary_auroc, binary_average_precision, classification_metrics
from tuac.scoring import kl_divergence
from tuac.structured_sparsity import (
    _probability_features,
    js_divergence,
    load_probability_artifact,
    predict_logistic,
)


ROOT = Path("runs/iter6_arc")


def close(left, right) -> bool:
    return bool(np.isclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def top_k(scores: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, min(len(scores), int(round(rate * len(scores)))))
    order = np.argsort(-np.asarray(scores), kind="stable")
    result = np.zeros(len(scores), dtype=bool)
    result[order[:count]] = True
    return result


def main() -> int:
    calibration_data = load_jsonl("runs/iter1_arc/calibration_pool.jsonl")
    evaluation_data = load_jsonl("runs/iter1_arc/evaluation.jsonl")
    cascade = CascadeProbabilityArtifact.load("runs/iter3_arc/probabilities.npz")
    calibration = load_probability_artifact(ROOT / "calibration.npz")
    evaluation = load_probability_artifact(ROOT / "evaluation.npz")
    analysis = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "report/artifact.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "masks.npz", allow_pickle=False) as saved:
        masks = {key: saved[key] for key in saved.files}
    with np.load(ROOT / "importance.npz", allow_pickle=False) as saved:
        importance = {key: saved[key] for key in saved.files}
    with np.load(ROOT / "probe.npz", allow_pickle=False) as saved:
        probe = {key: saved[key] for key in saved.files}

    medium_masks, aggressive_masks = masks["medium"], masks["aggressive"]
    checks = {
        "calibration_count_750": len(calibration_data) == 750,
        "evaluation_count_299": len(evaluation_data) == 299,
        "splits_disjoint": {row.id for row in calibration_data}.isdisjoint({row.id for row in evaluation_data}),
        "calibration_ids_match": calibration["example_ids"] == tuple(row.id for row in calibration_data),
        "evaluation_ids_match": evaluation["example_ids"] == tuple(row.id for row in evaluation_data),
        "cascade_calibration_ids_match": cascade.calibration_ids == calibration["example_ids"],
        "cascade_evaluation_ids_match": cascade.evaluation_ids == evaluation["example_ids"],
        "twenty_eight_layers": importance["importance"].shape[0] == 28,
        "dense_ffn_width_8960": importance["importance"].shape[1] == 8960,
        "medium_width_6720": medium_masks.shape == (28, 6720),
        "aggressive_width_4480": aggressive_masks.shape == (28, 4480),
        "physical_channel_pruning": calibration["metadata"]["physical_channel_pruning"] is True,
        "calibration_dense_reference_accepted": calibration["metadata"]["dense_reference_check"]["mean_absolute_probability_difference"] <= 0.01 and calibration["metadata"]["dense_reference_check"]["prediction_agreement"] >= 0.98,
        "evaluation_dense_reference_accepted": evaluation["metadata"]["dense_reference_check"]["mean_absolute_probability_difference"] <= 0.01 and evaluation["metadata"]["dense_reference_check"]["prediction_agreement"] >= 0.98,
        "static_gate_passed": analysis["static_gate"]["passed"] is True,
        "calibration_gate_passed": analysis["gate"]["passed"] is True,
        "uplift_ablation_failed": analysis["gate"]["uplift_ablation_passed"] is False,
        "heldout_gate_failed": benchmark["gate"]["passed"] is False,
        "summary_no_go": summary["decision"]["label"] == "NO-GO",
        "three_level_not_run": summary["decision"]["three_level_run"] is False,
        "mmlu_not_run": summary["decision"]["mmlu_pro_run"] is False and not Path("runs/iter6_mmlu_pro").exists(),
        "runtime_not_run": summary["decision"]["runtime_run"] is False and not (ROOT / "runtime").exists(),
        "report_html_exists": (ROOT / "report/report.html").is_file(),
        "report_html_nontrivial": (ROOT / "report/report.html").stat().st_size > 50_000,
        "report_three_charts": len(report["manifest"]["charts"]) == 3,
        "report_two_tables": len(report["manifest"]["tables"]) == 2,
    }

    for layer, (medium, aggressive) in enumerate(zip(medium_masks, aggressive_masks)):
        checks[f"medium_unique_{layer}"] = len(np.unique(medium)) == len(medium)
        checks[f"aggressive_unique_{layer}"] = len(np.unique(aggressive)) == len(aggressive)
        checks[f"nested_{layer}"] = np.isin(aggressive, medium).all()
        checks[f"medium_sorted_{layer}"] = bool(np.all(np.diff(medium) > 0))
        checks[f"aggressive_sorted_{layer}"] = bool(np.all(np.diff(aggressive) > 0))

    for split_name, artifact in (("calibration", calibration), ("evaluation", evaluation)):
        for variant in ("dense", "medium", "aggressive"):
            probabilities = artifact[variant]
            checks[f"{split_name}_{variant}_finite"] = np.isfinite(probabilities).all()
            checks[f"{split_name}_{variant}_nonnegative"] = (probabilities >= 0).all()
            checks[f"{split_name}_{variant}_rows_sum"] = np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    compute = analysis["metadata"]["compute"]
    fixed = compute["dense"]["fixed_transformer_matrix_parameters"]
    mlp = compute["dense"]["dense_mlp_matrix_parameters"]
    checks["dense_compute_one"] = close(compute["dense"]["normalized_linear_flops"], 1.0)
    checks["medium_compute_recomputed"] = close(compute["medium"]["normalized_linear_flops"], (fixed + 0.75 * mlp) / (fixed + mlp))
    checks["aggressive_compute_recomputed"] = close(compute["aggressive"]["normalized_linear_flops"], (fixed + 0.50 * mlp) / (fixed + mlp))

    labels = calibration["labels"]
    dense_correct = calibration["dense"].argmax(axis=1) == labels
    medium_correct = calibration["medium"].argmax(axis=1) == labels
    aggressive_correct = calibration["aggressive"].argmax(axis=1) == labels
    events = {
        "aggressive_to_medium": ~aggressive_correct & medium_correct,
        "aggressive_to_dense": ~aggressive_correct & dense_correct,
        "medium_to_dense": ~medium_correct & dense_correct,
    }
    static_lookup = {row["variant"]: row for row in analysis["static_rows"]}
    for variant in ("dense", "medium", "aggressive"):
        observed = classification_metrics(calibration[variant], labels)
        for metric, value in observed.items():
            checks[f"static_{variant}_{metric}"] = close(value, static_lookup[variant][metric])
        checks[f"static_{variant}_kl"] = close(
            np.mean(kl_divergence(calibration["dense"], calibration[variant])),
            static_lookup[variant]["output_kl_from_dense"],
        )
    for name, values in events.items():
        checks[f"event_count_{name}"] = analysis["event_counts"][name]["count"] == int(values.sum())
        checks[f"event_prevalence_{name}"] = close(analysis["event_counts"][name]["prevalence"], values.mean())

    sparse_uncertainty = uncertainty_scores(calibration["aggressive"])
    estimator_scores = {
        "margin": sparse_uncertainty["margin"],
        "entropy": sparse_uncertainty["entropy"],
        "max_probability": sparse_uncertainty["max_probability"],
        "error_probe": probe["error_oof"],
        "uplift_probe": probe["uplift_oof"],
        "cross_sparsity_js": js_divergence(calibration["aggressive"], calibration["medium"]),
        "prediction_disagreement": (calibration["aggressive"].argmax(axis=1) != calibration["medium"].argmax(axis=1)).astype(float),
    }
    estimator_lookup = {row["estimator"]: row for row in analysis["estimator_rows"]}
    uplift_labels = events["aggressive_to_dense"]
    for name, scores in estimator_scores.items():
        checks[f"estimator_{name}_auroc"] = close(binary_auroc(scores, uplift_labels), estimator_lookup[name]["auroc"])
        checks[f"estimator_{name}_auprc"] = close(binary_average_precision(scores, uplift_labels), estimator_lookup[name]["auprc"])

    eval_labels = evaluation["labels"]
    eval_aggressive_correct = evaluation["aggressive"].argmax(axis=1) == eval_labels
    eval_dense_correct = evaluation["dense"].argmax(axis=1) == eval_labels
    features = _probability_features(evaluation["aggressive"])
    uplift_model = {"mean": probe["feature_mean"], "scale": probe["feature_scale"], "weights": probe["uplift_weights"]}
    error_model = {"mean": probe["error_feature_mean"], "scale": probe["error_feature_scale"], "weights": probe["error_weights"]}
    eval_uncertainty = uncertainty_scores(evaluation["aggressive"])
    heldout_scores = {
        "margin": eval_uncertainty["margin"], "entropy": eval_uncertainty["entropy"],
        "max_probability": eval_uncertainty["max_probability"],
        "error_probe": predict_logistic(error_model, features),
        "uplift_probe": predict_logistic(uplift_model, features),
        "error_oracle": (~eval_aggressive_correct).astype(float),
        "uplift_oracle": (~eval_aggressive_correct & eval_dense_correct).astype(float),
    }
    correctness_vectors = {}
    for row in benchmark["records"]:
        method, rate = row["method"], float(row["target_rate"])
        if method.endswith("oracle"):
            decisions = top_k(heldout_scores[method], rate)
        else:
            threshold = analysis["thresholds"][method][str(rate)]
            decisions = heldout_scores[method] > threshold
        correct = np.where(decisions, eval_dense_correct, eval_aggressive_correct)
        correctness_vectors[(method, rate)] = correct
        checks[f"policy_{method}_{rate}_rate"] = close(decisions.mean(), row["dense_execution_rate"])
        checks[f"policy_{method}_{rate}_accuracy"] = close(correct.mean(), row["accuracy"])
        checks[f"policy_{method}_{rate}_compute"] = close(compute["aggressive"]["normalized_linear_flops"] + decisions.mean(), row["expected_sequential_compute"])
    for row in benchmark["comparisons"]:
        rate = float(row["target_rate"])
        baseline = row["comparison"].removeprefix("uplift_probe_minus_")
        difference = np.mean(
            correctness_vectors[("uplift_probe", rate)].astype(float)
            - correctness_vectors[(baseline, rate)].astype(float)
        )
        checks[f"comparison_{baseline}_{rate}"] = close(difference, row["difference"])

    source_lookup = {source["id"]: source for source in report["manifest"]["sources"]}
    checks["all_charts_have_sql_sources"] = all(source_lookup[chart["sourceId"]].get("query", {}).get("sql") for chart in report["manifest"]["charts"])
    checks["all_tables_have_sql_sources"] = all(source_lookup[table["sourceId"]].get("query", {}).get("sql") for table in report["manifest"]["tables"])
    block_index = {block["id"]: index for index, block in enumerate(report["manifest"]["blocks"])}
    for chart in report["manifest"]["charts"]:
        chart_blocks = [block for block in report["manifest"]["blocks"] if block.get("chartId") == chart["id"]]
        checks[f"chart_{chart['id']}_adjacent_narrative"] = len(chart_blocks) == 1 and block_index[chart_blocks[0]["id"]] > 0 and report["manifest"]["blocks"][block_index[chart_blocks[0]["id"]] - 1]["type"] == "markdown"
    checks["required_report_sections"] = all(
        any(label in block.get("body", "") for block in report["manifest"]["blocks"])
        for label in ("Technical summary", "Scope, data, and metric definitions", "Methodology and experimental design", "Limitations, uncertainty, and robustness", "Recommended next steps", "Further questions")
    )

    checks = {key: bool(value) for key, value in checks.items()}
    key_files = [
        ROOT / "calibration.npz", ROOT / "masks.npz", ROOT / "importance.npz",
        ROOT / "analysis.json", ROOT / "evaluation.npz", ROOT / "benchmark/results.json",
        ROOT / "summary.json", ROOT / "report/artifact.json", ROOT / "report/report.html",
    ]
    output = {
        "assessment": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "all_checks_passed": all(checks.values()), "check_count": len(checks), "checks": checks,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "report_package_verification": "structural_only",
        "sha256": {str(path): digest(path) for path in key_files},
        "material_caveats": [
            "The fresh calibration dense pass has 98.53% prediction agreement with its cached BF16 reference; all within-run comparisons use fresh probabilities.",
            "The learned uplift probe did not beat generic uncertainty in calibration and did not dominate entropy on held-out ARC.",
            "Cross-sparsity prediction disagreement requires an extra model execution and was not evaluated as a held-out three-level policy after the two-level gate failed.",
            "The compute axis is an analytical transformer matrix-parameter proxy, not latency, memory, energy, or throughput.",
            "Portable report packaging passed structurally; browser interaction and viewport QA were unavailable.",
        ],
    }
    (ROOT / "validation.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"assessment": output["assessment"], "passed": sum(checks.values()), "total": len(checks)}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
