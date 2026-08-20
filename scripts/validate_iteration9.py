"""Independently validate Iteration 9 data, metrics, gates, and report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tuac.capacity_contrast import load_contrast_artifact, scalar_output_features
from tuac.capacity_contrast_routing import linear_ccr_scores, load_ccr_evaluation, load_frozen_ccr
from tuac.counterfactual_routing import _paired_policy_interval, _policy_row
from tuac.metrics import binary_average_precision, binary_auroc


ROOT = Path("runs/iter9_mmlu_pro")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    router = load_frozen_ccr(ROOT / "frozen_ccr.npz")
    freeze = router["metadata"]
    check("router frozen before outcomes", freeze["weights_frozen_before_evaluation"] is True)
    check("freeze did not read outcomes", freeze["outcome_labels_read"] is False)
    check("frozen training count", freeze["training_examples"] == 2400)
    check("frozen validation count", freeze["validation_examples"] == 600)
    check("frozen held-out AUROC", abs(freeze["heldout_disagreement_auroc"] - 0.677678878999011) < 1e-12)
    check("freeze score reproduction", freeze["heldout_score_max_absolute_difference"] < 1e-6)
    check("router source hash", freeze["source_checkpoint_sha256"] == sha256(Path("runs/iter8_arc/routers.pt")))
    paired = load_contrast_artifact("runs/iter8_arc/paired_capacity.npz")
    with np.load("runs/iter8_arc/scores.npz", allow_pickle=False) as prior_scores:
        names = [str(value) for value in prior_scores["validation_score_names"]]
        validation = prior_scores["validation_indices"]
        reference = prior_scores["validation_scores"][names.index("linear")]
    reproduced = linear_ccr_scores(
        router, paired["hidden_states"][validation, 1].astype(np.float32),
        paired["scalar_features"][validation],
    )
    check("frozen scores independently reproduce", np.max(np.abs(reproduced - reference)) < 1e-6)

    cohort = json.loads((ROOT / "evaluation_manifest.json").read_text(encoding="utf-8"))
    check("cohort outcome blind", cohort["correctness_not_used"] is True)
    check("cohort checked 1000", cohort["evaluation_questions_checked"] == 1000)
    check("no exact duplicates", cohort["exact_duplicates_removed"] == 0)
    check("no near duplicates", cohort["near_duplicates_removed"] == 0)
    check("final cohort 1000", cohort["final_evaluation_size"] == 1000)
    check("cohort output hash", cohort["output_sha256"] == sha256(ROOT / "evaluation.jsonl"))
    check("cohort IDs unique", len(set(cohort["final_ids"])) == 1000)

    evaluation = load_ccr_evaluation(ROOT / "evaluation.npz")
    check("evaluation IDs match manifest", list(evaluation["example_ids"]) == cohort["final_ids"])
    check("evaluation data hash", evaluation["metadata"]["data_sha256"] == cohort["output_sha256"])
    check("evaluation router hash", evaluation["metadata"]["frozen_router_sha256"] == sha256(ROOT / "frozen_ccr.npz"))
    check("evaluation examples", len(evaluation["labels"]) == 1000)
    check("ten-choice matrices", all(values.shape == (1000, 10) for values in (evaluation["s50"], evaluation["dense"], evaluation["s25"])))
    check("h14 shape", evaluation["hidden_l14"].shape == (1000, 1536))
    check("probabilities finite", all(np.isfinite(values).all() for values in (evaluation["s50"], evaluation["dense"], evaluation["s25"])))
    check("probabilities normalized", all(np.allclose(values.sum(axis=1), 1.0, atol=1e-5) for values in (evaluation["s50"], evaluation["dense"], evaluation["s25"])))
    recomputed_scalar = scalar_output_features(evaluation["s50"])
    check("scalar features recompute", np.allclose(recomputed_scalar, evaluation["scalar_features"], atol=1e-7))
    recomputed_ccr = linear_ccr_scores(router, evaluation["hidden_l14"], recomputed_scalar)
    check("evaluation CCR scores recompute", np.allclose(recomputed_ccr, evaluation["ccr_scores"], atol=1e-6))

    analysis = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
    labels = evaluation["labels"]
    sparse_correct = evaluation["s50"].argmax(axis=1) == labels
    dense_correct = evaluation["dense"].argmax(axis=1) == labels
    advantage = dense_correct.astype(np.int8) - sparse_correct.astype(np.int8)
    positive = advantage == 1
    check("S50 accuracy recomputes", abs(sparse_correct.mean() - analysis["static"]["s50_accuracy"]) < 1e-12)
    check("dense accuracy recomputes", abs(dense_correct.mean() - analysis["static"]["dense_accuracy"]) < 1e-12)
    check("positive count recomputes", positive.sum() == analysis["static"]["positive_advantage_count"])
    check("harmful count recomputes", np.sum(advantage == -1) == analysis["static"]["harmful_advantage_count"])
    check("advantage categories total", sum(analysis["static"][key] for key in ("positive_advantage_count", "harmful_advantage_count", "neutral_count")) == 1000)

    with np.load(ROOT / "scores.npz", allow_pickle=False) as saved:
        score_names = [str(value) for value in saved["score_names"]]
        score_matrix = saved["scores"].astype(np.float64)
        check("saved advantage target", np.array_equal(saved["signed_advantage"], advantage))
        check("saved sparse correctness", np.array_equal(saved["sparse_correct"], sparse_correct))
        check("saved dense correctness", np.array_equal(saved["dense_correct"], dense_correct))
    score_lookup = {name: score_matrix[index] for index, name in enumerate(score_names)}
    metric_lookup = {row["method"]: row for row in analysis["advantage_metrics"]}
    for name in score_names:
        check(f"{name} AUROC recomputes", abs(binary_auroc(score_lookup[name], positive) - metric_lookup[name]["auroc"]) < 1e-7)
        check(f"{name} AUPRC recomputes", abs(binary_average_precision(score_lookup[name], positive) - metric_lookup[name]["auprc"]) < 1e-7)

    policy_lookup = {(row["method"], row["target_rate"]): row for row in analysis["policy_rows"]}
    decision_lookup = {}
    for name in ("ccr", "entropy", "max_probability", "actual_capacity_disagreement", "signed_advantage_oracle"):
        for rate in (0.1, 0.2, 0.3, 0.5):
            row, decision, _ = _policy_row(name, score_lookup[name], rate, advantage, sparse_correct, analysis["static"]["s50_compute_proxy"])
            saved_row = policy_lookup[(name, rate)]
            check(f"{name} accuracy @{rate} recomputes", abs(row["accuracy"] - saved_row["accuracy"]) < 1e-12)
            check(f"{name} NAC @{rate} recomputes", abs(row["net_advantage_capture"] - saved_row["net_advantage_capture"]) < 1e-12)
            decision_lookup[(name, rate)] = decision
    check("20 random seeds", len({row["seed"] for row in analysis["random_rows"]}) == 20)
    check("four random operating points", len(analysis["random_summary"]) == 4)

    comparison_lookup = {(row["comparison"], row["target_rate"]): row for row in analysis["comparisons"]}
    recomputed_comparison = _paired_policy_interval(
        decision_lookup[("ccr", 0.3)], decision_lookup[("max_probability", 0.3)],
        advantage, sparse_correct, 0.3, seed=20260826 + 300 + len("max_probability"), replicates=5000,
    )
    saved_comparison = comparison_lookup[("ccr_minus_max_probability", 0.3)]
    check("primary accuracy difference recomputes", abs(recomputed_comparison["accuracy_difference"] - saved_comparison["accuracy_difference"]) < 1e-12)
    check("primary bootstrap CI recomputes", np.allclose(recomputed_comparison["accuracy_difference_ci"], saved_comparison["accuracy_difference_ci"], atol=1e-12))
    for method in ("ccr", "entropy", "max_probability"):
        rows = [row for row in analysis["routing_outcomes_30"] if row["method"] == method]
        check(f"{method} outcome composition totals", abs(sum(row["selected_rate"] for row in rows) - 1.0) < 1e-12)
        check(f"{method} outcome counts total", sum(row["selected_count"] for row in rows) == 300)
    check("threshold rows present", [row["target_calibration_rate"] for row in analysis["threshold_transfer"]] == [0.2, 0.3])
    check("threshold transfer under-routes", all(row["actual_dense_rate"] < row["target_calibration_rate"] for row in analysis["threshold_transfer"]))

    gate = analysis["gate"]
    alignment = metric_lookup["ccr"]["auroc"] > max(metric_lookup["entropy"]["auroc"], metric_lookup["max_probability"]["auroc"])
    system = any(row["passed"] for row in gate["meaningful_operating_points"])
    check("alignment gate recomputes", gate["alignment_passed"] == alignment)
    check("system gate recomputes", gate["system_passed"] == system)
    check("overall gate recomputes", gate["passed"] == (alignment and system))
    check("NO-GO stop enforced", gate["decision"] == "NO-GO" and gate["next_stage"] == "STOP")
    check("conditional work blocked", all(value == "not_run_gate_failed" for value in analysis["conditional_followups"].values()))

    artifact = json.loads((ROOT / "report" / "artifact.json").read_text(encoding="utf-8"))
    html = (ROOT / "report" / "report.html").read_text(encoding="utf-8")
    check("report surface", artifact["surface"] == "report")
    check("report snapshot ready", artifact["snapshot"]["status"] == "ready")
    check("four charts", len(artifact["manifest"]["charts"]) == 4)
    check("one threshold table", len(artifact["manifest"]["tables"]) == 1)
    check("report technical sections", all(value in html for value in (
        "Technical summary", "Scope, data, and metric definitions", "Frozen model specification and experimental design",
        "Limitations, uncertainty, and robustness checks", "Recommended next steps", "Further questions",
    )))
    check("semantic fallback embedded", "portable-fallback" in html)
    result = {"status": "passed" if all(row["passed"] for row in checks) else "failed",
              "passed": sum(row["passed"] for row in checks), "total": len(checks), "checks": checks}
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "passed", "total")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
