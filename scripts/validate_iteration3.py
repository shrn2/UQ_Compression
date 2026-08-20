"""Independent integrity and reproducibility checks for Iteration 3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from tuac.cascade import CascadeProbabilityArtifact, top_fraction_decisions, uncertainty_scores
from tuac.data import load_jsonl
from tuac.metrics import classification_metrics, pairwise_ordering_flip_rate, spearman_correlation


ARC = Path("runs/iter3_arc")
MMLU = Path("runs/iter3_mmlu_pro")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def close(left, right) -> bool:
    return bool(np.isclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True))


def valid_probabilities(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(values >= 0) and np.allclose(values.sum(axis=-1), 1, atol=1e-5))


def main() -> int:
    arc_data_cal = load_jsonl("runs/iter1_arc/calibration_pool.jsonl")
    arc_data_eval = load_jsonl("runs/iter1_arc/evaluation.jsonl")
    mmlu_data_cal = load_jsonl(MMLU / "calibration.jsonl")
    mmlu_data_eval = load_jsonl(MMLU / "evaluation.jsonl")
    arc_artifact = CascadeProbabilityArtifact.load(ARC / "probabilities.npz")
    arc = json.loads((ARC / "analysis.json").read_text(encoding="utf-8"))
    mmlu_artifact = CascadeProbabilityArtifact.load(MMLU / "probabilities.npz")
    mmlu = json.loads((MMLU / "analysis.json").read_text(encoding="utf-8"))
    mmlu_drift_checkpoint = json.loads((MMLU / "drift_analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((ARC / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((ARC / "report/artifact.json").read_text(encoding="utf-8"))

    checks: dict[str, bool] = {
        "arc_calibration_count_750": len(arc_data_cal) == 750,
        "arc_evaluation_count_299": len(arc_data_eval) == 299,
        "arc_split_ids_disjoint": {x.id for x in arc_data_cal}.isdisjoint({x.id for x in arc_data_eval}),
        "arc_artifact_ids_match": set(arc_artifact.calibration_ids) == {x.id for x in arc_data_cal} and set(arc_artifact.evaluation_ids) == {x.id for x in arc_data_eval},
        "arc_bits_8_4_2": arc_artifact.bits == (8, 4, 2),
        "arc_slm_model_1_5b": arc_artifact.metadata["slm_model"] == "Qwen/Qwen2.5-1.5B-Instruct",
        "arc_fallback_model_7b": arc_artifact.metadata["fallback_model"] == "Qwen/Qwen2.5-7B-Instruct",
        "arc_fallback_better": arc["decision"]["criteria"]["fallback_accuracy_gap_positive"],
        "arc_router_gate_passed": arc["router"]["gate_passed"],
        "arc_router_margin_selected": arc["router"]["selected_signal"] == "margin",
        "arc_bootstrap_2000": arc["bootstrap_replicates"] == 2000,
        "arc_random_10_seeds": len(arc["scope"]["random_seeds"]) == 10,
        "arc_four_rates": arc["scope"]["target_rates"] == [0.1, 0.2, 0.3, 0.5],
        "arc_decision_moderate": arc["decision"]["label"] == "MODERATE_POSITIVE",
        "arc_real_quant_not_warranted": arc["decision"]["real_quantization_warranted"] is False,
        "mmlu_calibration_count_70": len(mmlu_data_cal) == 70,
        "mmlu_evaluation_count_1000": len(mmlu_data_eval) == 1000,
        "mmlu_split_ids_disjoint": {x.id for x in mmlu_data_cal}.isdisjoint({x.id for x in mmlu_data_eval}),
        "mmlu_artifact_ids_match": set(mmlu_artifact.calibration_ids) == {x.id for x in mmlu_data_cal} and set(mmlu_artifact.evaluation_ids) == {x.id for x in mmlu_data_eval},
        "mmlu_bits_8_4_2": mmlu_artifact.bits == (8, 4, 2),
        "mmlu_slm_model_1_5b": mmlu_artifact.metadata["slm_model"] == "Qwen/Qwen2.5-1.5B-Instruct",
        "mmlu_fallback_model_3b": mmlu_artifact.metadata["fallback_model"] == "Qwen/Qwen2.5-3B-Instruct",
        "mmlu_router_gate_passed": mmlu["router"]["gate_passed"],
        "mmlu_router_entropy_selected": mmlu["router"]["selected_signal"] == "entropy",
        "mmlu_bootstrap_2000": mmlu["bootstrap_replicates"] == 2000,
        "mmlu_fallback_better": mmlu["decision"]["criteria"]["fallback_accuracy_gap_positive"],
        "mmlu_complete_routing_rows": len(mmlu["routing_rows"]) == 249,
        "mmlu_complete_flip_rows": len(mmlu["flip_rows"]) == 12,
        "summary_claim_boundary": summary["claim_boundary"]["routing_only_system_harm_established"] is False and summary["claim_boundary"]["deployment_claims_supported"] is False,
        "summary_mmlu_complete": summary["claim_boundary"]["mmlu_cascade_completed"] is True and summary["mmlu_pro"]["status"] == "complete",
        "report_html_exists": (ARC / "report/report.html").is_file(),
        "report_html_nontrivial": (ARC / "report/report.html").stat().st_size > 50_000,
        "report_has_8_charts": len(report["manifest"]["charts"]) == 8,
        "report_has_3_tables": len(report["manifest"]["tables"]) == 3,
        "report_has_required_sections": all(any(label in block.get("body", "") for block in report["manifest"]["blocks"]) for label in ("Technical summary", "Scope and metric definitions", "Methods and experimental design", "Limitations and robustness", "Recommended next step", "Further questions")),
    }

    for name, values in (
        ("arc_fp_cal", arc_artifact.slm_fp_calibration), ("arc_fp_eval", arc_artifact.slm_fp_evaluation),
        ("arc_fallback_cal", arc_artifact.fallback_calibration), ("arc_fallback_eval", arc_artifact.fallback_evaluation),
        ("arc_q_cal", arc_artifact.quantized_calibration), ("arc_q_eval", arc_artifact.quantized_evaluation),
    ):
        checks[f"{name}_valid_probabilities"] = valid_probabilities(values)
    for name, values in (
        ("mmlu_fp_cal", mmlu_artifact.slm_fp_calibration), ("mmlu_fp_eval", mmlu_artifact.slm_fp_evaluation),
        ("mmlu_fallback_cal", mmlu_artifact.fallback_calibration), ("mmlu_fallback_eval", mmlu_artifact.fallback_evaluation),
        ("mmlu_q_cal", mmlu_artifact.quantized_calibration), ("mmlu_q_eval", mmlu_artifact.quantized_evaluation),
    ):
        checks[f"{name}_valid_probabilities"] = valid_probabilities(values)

    # Recompute precision-level task and rank metrics from primary probability arrays.
    arc_signal = arc["router"]["selected_signal"]
    arc_fp_u = uncertainty_scores(arc_artifact.slm_fp_evaluation)[arc_signal]
    arc_probs = {"fp": arc_artifact.slm_fp_evaluation, **{f"q{bit}": arc_artifact.quantized_evaluation[i] for i, bit in enumerate(arc_artifact.bits)}}
    arc_small = {row["precision"]: row for row in arc["small_model_rows"]}
    arc_drift = {row["precision"]: row for row in arc["drift_rows"]}
    for precision, probabilities in arc_probs.items():
        metrics = classification_metrics(probabilities, arc_artifact.evaluation_labels)
        scores = uncertainty_scores(probabilities)[arc_signal]
        checks[f"arc_{precision}_accuracy_recomputed"] = close(metrics["accuracy"], arc_small[precision]["accuracy"])
        checks[f"arc_{precision}_auroc_recomputed"] = close(metrics["error_auroc"], arc_small[precision]["error_auroc"])
        checks[f"arc_{precision}_rank_rho_recomputed"] = close(spearman_correlation(arc_fp_u, scores), arc_drift[precision]["uncertainty_spearman"])
        checks[f"arc_{precision}_ior_recomputed"] = close(pairwise_ordering_flip_rate(arc_fp_u, scores), arc_drift[precision]["pairwise_ordering_inversion"])

    mmlu_signal = mmlu["router"]["selected_signal"]
    mmlu_fp_u = uncertainty_scores(mmlu_artifact.slm_fp_evaluation)[mmlu_signal]
    mmlu_small = {row["precision"]: row for row in mmlu["small_model_rows"]}
    mmlu_drift = {row["precision"]: row for row in mmlu["drift_rows"]}
    mmlu_probs = {"fp": mmlu_artifact.slm_fp_evaluation, **{f"q{bit}": mmlu_artifact.quantized_evaluation[i] for i, bit in enumerate(mmlu_artifact.bits)}}
    for precision, probabilities in mmlu_probs.items():
        metrics = classification_metrics(probabilities, mmlu_artifact.evaluation_labels)
        scores = uncertainty_scores(probabilities)[mmlu_signal]
        checks[f"mmlu_{precision}_accuracy_recomputed"] = close(metrics["accuracy"], mmlu_small[precision]["accuracy"])
        checks[f"mmlu_{precision}_auroc_recomputed"] = close(metrics["error_auroc"], mmlu_small[precision]["error_auroc"])
        checks[f"mmlu_{precision}_rank_rho_recomputed"] = close(spearman_correlation(mmlu_fp_u, scores), mmlu_drift[precision]["uncertainty_spearman"])
        checks[f"mmlu_{precision}_ior_recomputed"] = close(pairwise_ordering_flip_rate(mmlu_fp_u, scores), mmlu_drift[precision]["pairwise_ordering_inversion"])

    # Recompute every MMLU exact matched-rate DFR directly from evaluation scores.
    decision_lookup = {(row["precision"], row["target_rate"], row["method"]): row for row in mmlu_drift_checkpoint["decision_rows"]}
    for precision in ("q8", "q4", "q2"):
        scores = uncertainty_scores(mmlu_probs[precision])[mmlu_signal]
        for rate in (0.1, 0.2, 0.3, 0.5):
            fp_decision = top_fraction_decisions(mmlu_fp_u, rate)
            q_decision = top_fraction_decisions(scores, rate)
            observed = float(np.mean(fp_decision != q_decision))
            expected = decision_lookup[(precision, rate, "matched_rate")]["delegation_flip_rate"]
            checks[f"mmlu_{precision}_{rate}_matched_dfr_recomputed"] = close(observed, expected)

    # The decision is deliberately guarded by the routing-only causal comparison.
    for key, value in arc["routing_only_comparisons"].items():
        checks[f"routing_only_ci_order_{key}"] = value["bootstrap_95_ci"][0] <= value["bootstrap_95_ci"][1]
        checks[f"arc_routing_only_ci_contains_zero_{key}"] = value["bootstrap_95_ci"][0] <= 0 <= value["bootstrap_95_ci"][1]
    for key, value in mmlu["routing_only_comparisons"].items():
        low, high = value["bootstrap_95_ci"]
        checks[f"mmlu_routing_only_ci_order_{key}"] = low <= high
        checks[f"mmlu_no_significant_harm_{key}"] = low <= 0
    checks["decision_rule_no_significant_routing_only_loss"] = arc["decision"]["criteria"]["significant_routing_only_loss"] is False

    source_lookup = {item["id"]: item for item in report["manifest"]["sources"]}
    checks["all_charts_have_sql_sources"] = all(source_lookup[chart["sourceId"]].get("query", {}).get("sql") for chart in report["manifest"]["charts"])
    checks["all_tables_have_sql_sources"] = all(source_lookup[table["sourceId"]].get("query", {}).get("sql") for table in report["manifest"]["tables"])
    block_index = {block["id"]: index for index, block in enumerate(report["manifest"]["blocks"])}
    for chart in report["manifest"]["charts"]:
        chart_blocks = [block for block in report["manifest"]["blocks"] if block.get("chartId") == chart["id"]]
        checks[f"chart_{chart['id']}_has_adjacent_narrative"] = len(chart_blocks) == 1 and block_index[chart_blocks[0]["id"]] > 0 and report["manifest"]["blocks"][block_index[chart_blocks[0]["id"]] - 1]["type"] == "markdown"

    checks = {key: bool(value) for key, value in checks.items()}
    key_files = [
        ARC / "probabilities.npz", ARC / "analysis.json", ARC / "summary.json",
        MMLU / "calibration.jsonl", MMLU / "evaluation.jsonl", MMLU / "probabilities.npz", MMLU / "analysis.json", MMLU / "drift_analysis.json",
        ARC / "report/artifact.json", ARC / "report/report.html",
    ]
    output = {
        "assessment": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "report_package_verification": "structural_only",
        "sha256": {str(path): digest(path) for path in key_files},
        "material_caveats": [
            "Controlled fake quantization does not establish packed storage, latency, memory, throughput, or kernel behavior.",
            "ARC routing-only paired intervals all cross zero at n=299; system-level routing harm is not established.",
            "MMLU-Pro uses a deterministic category-stratified 1,000-example test sample rather than all 12,032 test rows.",
            "ARC and MMLU-Pro use different fallbacks (7B and 3B), so absolute frontiers are not directly comparable.",
            "Portable report packaging passed structurally; browser interaction and viewport QA were unavailable.",
        ],
    }
    (ARC / "validation.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"assessment": output["assessment"], "checks": len(checks), "passed": sum(checks.values())}, indent=2))
    if not output["all_checks_passed"]:
        print("failed:", [key for key, value in checks.items() if not value])
    return 0 if output["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
