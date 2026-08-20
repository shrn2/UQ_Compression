"""Independent integrity and reproducibility checks for Iteration 4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from tuac.artifacts import MultiBitCalibrationArtifact
from tuac.cascade import CascadeProbabilityArtifact, uncertainty_scores
from tuac.cascade_compression import routing_outcomes
from tuac.data import load_jsonl
from tuac.metrics import classification_metrics, spearman_correlation
from tuac.scoring import kl_divergence


ROOT = Path("runs/iter4_arc")


def close(left, right) -> bool:
    return bool(np.isclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True))


def valid_probabilities(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(values >= 0) and np.allclose(values.sum(axis=-1), 1.0, atol=1e-5))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    calibration = load_jsonl("runs/iter1_arc/calibration_pool.jsonl")
    evaluation = load_jsonl("runs/iter1_arc/evaluation.jsonl")
    cascade = CascadeProbabilityArtifact.load("runs/iter3_arc/probabilities.npz")
    probes = MultiBitCalibrationArtifact.load(ROOT / "layer_probes.npz")
    policies = json.loads((ROOT / "policies.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "report/artifact.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "benchmark/probabilities.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}

    checks: dict[str, bool] = {
        "calibration_count_750": len(calibration) == 750,
        "evaluation_count_299": len(evaluation) == 299,
        "split_ids_disjoint": {item.id for item in calibration}.isdisjoint({item.id for item in evaluation}),
        "probe_ids_match_calibration": probes.example_ids == tuple(item.id for item in calibration),
        "cascade_calibration_ids_match": cascade.calibration_ids == probes.example_ids,
        "cascade_evaluation_ids_match": cascade.evaluation_ids == tuple(item.id for item in evaluation),
        "probe_shape_3_28_750_5": probes.quantized_probabilities.shape == (3, 28, 750, 5),
        "probe_bits_2_4_8": probes.probe_bits == (2, 4, 8),
        "probe_model_1_5b": probes.metadata["student_model"] == "Qwen/Qwen2.5-1.5B-Instruct",
        "fallback_model_7b": probes.metadata["fallback_model"] == "Qwen/Qwen2.5-7B-Instruct",
        "probe_fp_reuses_cascade": np.array_equal(probes.student_probabilities, cascade.slm_fp_calibration),
        "probe_fallback_reuses_cascade": np.array_equal(probes.teacher_probabilities, cascade.fallback_calibration),
        "fp_probabilities_valid": valid_probabilities(probes.student_probabilities),
        "fallback_probabilities_valid": valid_probabilities(probes.teacher_probabilities),
        "probe_probabilities_valid": valid_probabilities(probes.quantized_probabilities),
        "all_84_probe_checkpoints_exist": len(list((ROOT / "layer_probes_checkpoints").glob("*.npy"))) == 84,
        "policy_scope_750": policies["scope"]["examples"] == 750,
        "policy_four_rates": policies["scope"]["target_rates"] == [0.1, 0.2, 0.3, 0.5],
        "policy_five_costs": policies["scope"]["cost_coefficients"] == [0.0, 0.01, 0.025, 0.05, 0.1],
        "policy_two_budgets": policies["scope"]["budgets"] == [3.5, 4.0],
        "policy_ten_random_seeds": policies["scope"]["random_seeds"] == list(range(10)),
        "router_signal_margin": policies["scope"]["router_signal"] == "margin",
        "thresholds_frozen": policies["scope"]["threshold_policy"] == "FP calibration thresholds frozen across compression methods",
        "benchmark_299": benchmark["scope"]["evaluation_examples"] == 299,
        "benchmark_26_configurations": len(benchmark["configurations"]) == 26,
        "benchmark_520_rows": len(benchmark["records"]) == 520,
        "bootstrap_2000": benchmark["scope"]["bootstrap_replicates"] == 2000,
        "bootstrap_seed_fixed": benchmark["scope"]["bootstrap_seed"] == 20260819,
        "stage1_gate_failed": benchmark["gate"]["passed"] is False,
        "next_stage_stop": benchmark["gate"]["next_stage"] == "STOP",
        "summary_no_go": summary["decision"]["label"] == "NO-GO",
        "dbaq_proxy_only": summary["decision"]["dbaq_proxy_computed"] is True and summary["decision"]["dbaq_policy_evaluated"] is False,
        "later_stages_declared_not_run": not any(summary["decision"][key] for key in ("carq_search_run", "mmlu_pro_run", "real_quantization_run")),
        "no_dbaq_benchmark": not (ROOT / "benchmark_dbaq").exists(),
        "no_carq_benchmark": not (ROOT / "benchmark_carq").exists(),
        "no_iter4_mmlu": not Path("runs/iter4_mmlu_pro").exists(),
        "no_real_quantization_output": not (ROOT / "real_quantization").exists(),
        "report_html_exists": (ROOT / "report/report.html").is_file(),
        "report_html_nontrivial": (ROOT / "report/report.html").stat().st_size > 50_000,
        "report_has_5_charts": len(report["manifest"]["charts"]) == 5,
        "report_has_2_tables": len(report["manifest"]["tables"]) == 2,
        "report_required_sections": all(any(label in block.get("body", "") for block in report["manifest"]["blocks"]) for label in ("Technical summary", "Scope and metric definitions", "Experimental design", "Limitations and robustness", "Recommended next step", "Further questions")),
    }

    # Every method and random policy must spend the same attainable weighted budget.
    for budget in ("3.5", "4.0"):
        rows = policies["policies"][budget]
        expected = rows["kl"]["total_bits"]
        for method in ("kl", "accuracy", "uq", "cuaq", "dbaq"):
            checks[f"matched_total_bits_{budget}_{method}"] = rows[method]["total_bits"] == expected
            checks[f"allocation_length_{budget}_{method}"] = len(rows[method]["bits"]) == 28
        checks[f"ten_random_policies_{budget}"] = len(rows["random"]) == 10
        checks[f"ten_unique_random_policies_{budget}"] = len({tuple(row["bits"]) for row in rows["random"].values()}) == 10
        for seed, row in rows["random"].items():
            checks[f"matched_random_bits_{budget}_{seed}"] = row["total_bits"] == expected

    # Recompute all local KL and CUAQ sensitivities from source probabilities.
    row_lookup = {(row["bit"], row["layer_index"]): row for row in policies["layer_rows"]}
    signal = policies["scope"]["router_signal"]
    thresholds = {float(key): value for key, value in policies["thresholds"].items()}
    costs = policies["scope"]["cost_coefficients"]
    fp_scores = uncertainty_scores(probes.student_probabilities)[signal]
    fp_vectors = []
    for rate, threshold in thresholds.items():
        _, correct, decisions = routing_outcomes(probes.student_probabilities, probes.teacher_probabilities, probes.labels, fp_scores, threshold)
        for cost in costs:
            fp_vectors.append(correct.astype(float) - cost * decisions.astype(float))
    fp_mean = float(np.mean(fp_vectors))
    for bit_index, bit in enumerate(probes.probe_bits):
        for layer_index in range(len(probes.layer_names)):
            row = row_lookup[(bit, layer_index)]
            probabilities = probes.quantized_probabilities[bit_index, layer_index]
            observed_kl = float(np.mean(kl_divergence(probes.student_probabilities, probabilities)))
            scores = uncertainty_scores(probabilities)[signal]
            vectors = []
            for rate, threshold in thresholds.items():
                _, correct, decisions = routing_outcomes(probabilities, probes.teacher_probabilities, probes.labels, scores, threshold)
                for cost in costs:
                    vectors.append(correct.astype(float) - cost * decisions.astype(float))
            observed_cuaq = fp_mean - float(np.mean(vectors))
            checks[f"kl_recomputed_bit{bit}_layer{layer_index}"] = close(observed_kl, row["kl_sensitivity"])
            checks[f"cuaq_recomputed_bit{bit}_layer{layer_index}"] = close(observed_cuaq, row["cuaq_sensitivity"])

    # Recompute every held-out record from the saved probability arrays.
    config_lookup = {row["name"]: row for row in benchmark["configurations"]}
    fp_eval = arrays["full_precision"]
    fallback_eval = arrays["fallback"]
    labels = arrays["labels"]
    fp_eval_scores = uncertainty_scores(fp_eval)[signal]
    for index, row in enumerate(benchmark["records"]):
        config = config_lookup[row["name"]]
        probabilities = arrays[config["probability_key"]]
        scores = uncertainty_scores(probabilities)[signal]
        metrics = classification_metrics(probabilities, labels)
        outcome, correct, decisions = routing_outcomes(probabilities, fallback_eval, labels, scores, thresholds[row["target_rate"]])
        utility = float(np.mean(correct.astype(float) - row["cost_coefficient"] * decisions.astype(float)))
        checks[f"record_{index}_slm_accuracy"] = close(row["standalone_accuracy"], metrics["accuracy"])
        checks[f"record_{index}_cascade_accuracy"] = close(row["cascade_accuracy"], outcome["cascade_accuracy"])
        checks[f"record_{index}_delegation_rate"] = close(row["delegation_rate"], outcome["delegation_rate"])
        checks[f"record_{index}_utility"] = close(row["cascade_utility"], utility)
        checks[f"record_{index}_prediction_agreement"] = close(row["prediction_agreement_with_fp"], np.mean(probabilities.argmax(axis=1) == fp_eval.argmax(axis=1)))
        checks[f"record_{index}_rank_rho"] = close(row["uncertainty_spearman_with_fp"], spearman_correlation(fp_eval_scores, scores))

    for budget, result in benchmark["comparisons"].items():
        checks[f"cuaq_not_above_kl_{budget}"] = result["cuaq_vs_kl"]["difference"] <= 0
        checks[f"random_mean_ci_contains_zero_{budget}"] = result["cuaq_vs_random_mean"]["bootstrap_95_ci"][0] <= 0 <= result["cuaq_vs_random_mean"]["bootstrap_95_ci"][1]
    checks["4bit_cuaq_significantly_worse_than_kl"] = benchmark["comparisons"]["4.0"]["cuaq_vs_kl"]["bootstrap_95_ci"][1] < 0
    gate_expected = any(
        result["cuaq_vs_kl"]["difference"] > 0 and result["cuaq_vs_random_mean"]["difference"] > 0
        for result in benchmark["comparisons"].values()
    )
    checks["gate_logic_recomputed"] = benchmark["gate"]["passed"] == gate_expected

    source_lookup = {source["id"]: source for source in report["manifest"]["sources"]}
    checks["all_charts_have_sql_sources"] = all(source_lookup[chart["sourceId"]].get("query", {}).get("sql") for chart in report["manifest"]["charts"])
    checks["all_tables_have_sql_sources"] = all(source_lookup[table["sourceId"]].get("query", {}).get("sql") for table in report["manifest"]["tables"])
    block_index = {block["id"]: i for i, block in enumerate(report["manifest"]["blocks"])}
    for chart in report["manifest"]["charts"]:
        chart_blocks = [block for block in report["manifest"]["blocks"] if block.get("chartId") == chart["id"]]
        checks[f"chart_{chart['id']}_adjacent_narrative"] = len(chart_blocks) == 1 and block_index[chart_blocks[0]["id"]] > 0 and report["manifest"]["blocks"][block_index[chart_blocks[0]["id"]] - 1]["type"] == "markdown"

    checks = {key: bool(value) for key, value in checks.items()}
    key_files = [ROOT / "layer_probes.npz", ROOT / "policies.json", ROOT / "benchmark/probabilities.npz", ROOT / "benchmark/results.json", ROOT / "summary.json", ROOT / "report/artifact.json", ROOT / "report/report.html"]
    output = {
        "assessment": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "report_package_verification": "structural_only",
        "sha256": {str(path): digest(path) for path in key_files},
        "material_caveats": [
            "The no-go applies to one model pair, ARC-Challenge, layer-local additive CUAQ, and controlled fake quantization.",
            "The 3.5-bit CUAQ-versus-KL interval crosses zero, while the 4-bit loss is statistically clear.",
            "Fake quantization does not establish packed size, latency, memory, or throughput.",
            "Later stages were intentionally skipped under the preregistered Stage-1 gate.",
            "Portable report packaging passed structurally; browser interaction and viewport QA were unavailable.",
        ],
    }
    (ROOT / "validation.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"assessment": output["assessment"], "checks": len(checks), "passed": sum(checks.values())}, indent=2))
    if not output["all_checks_passed"]:
        print("failed:", [key for key, value in checks.items() if not value])
    return 0 if output["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
