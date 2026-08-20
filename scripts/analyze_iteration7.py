"""Consolidate Iteration 7 oracle and ARC-development evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("runs/iter7_arc")


def main() -> int:
    oracle = json.loads((ROOT / "oracle_analysis.json").read_text(encoding="utf-8"))
    development = json.loads((ROOT / "development_analysis.json").read_text(encoding="utf-8"))
    estimator_lookup = {row["estimator"]: row for row in development["estimator_rows"]}
    policy_lookup = {
        (row["method"], row["target_rate"]): row for row in development["policy_rows"]
    }
    selected_method = development["gate_b"]["selected_method"]
    display_methods = [
        "margin", "entropy", "max_probability", "error_probe",
        "binary_uplift_probe", "output_signed_ridge", selected_method,
        "predicted_instability_hidden_l14", "actual_capacity_disagreement",
        "positive_uplift_oracle", "signed_advantage_oracle",
    ]
    estimator_rows = [estimator_lookup[name] for name in display_methods]
    policy_rows = [
        row for row in development["policy_rows"] if row["method"] in display_methods
    ]
    frontier_rows = [dict(row, series=row["method"]) for row in policy_rows]
    for row in development["random_summary"]:
        frontier_rows.append({
            "method": "random_mean_10_seeds", "series": "random_mean_10_seeds",
            "target_rate": row["target_rate"], "dense_execution_rate": row["target_rate"],
            "accuracy": row["accuracy_mean"],
            "expected_sequential_compute": development["sparse_compute"] + row["target_rate"],
            "net_advantage_capture": row["nac_mean"],
            "oracle_recovery": row["oracle_recovery_mean"],
        })
    oracle_rows = [dict(row, series=row["variant"]) for row in oracle["oracle_rows"]]
    layer_rows = []
    for layer in ("l7", "l14", "l21", "l28", "concat", "concat_output"):
        for target in ("signed_advantage", "loss_advantage", "predicted_instability"):
            name = f"{target}_hidden_{layer}"
            row = estimator_lookup[name]
            layer_rows.append({
                "layer": layer.upper().replace("CONCAT_OUTPUT", "Concat + output").replace("CONCAT", "Concat"),
                "target": target, "estimator": name,
                "auroc": row["auroc_positive_advantage"],
                "auprc": row["auprc_positive_advantage"],
                "loss_spearman": row["loss_advantage_spearman"],
                "positive_folds": row["positive_fold_spearman_count"],
                "oracle_recovery_30": policy_lookup[(name, 0.3)]["oracle_recovery"],
            })
    outcome_rows = []
    for method in ("entropy", selected_method, "actual_capacity_disagreement", "signed_advantage_oracle"):
        row = policy_lookup[(method, 0.3)]
        for outcome, field in (
            ("Beneficial", "useful_densification_rate"),
            ("Harmful", "harmful_densification_rate"),
            ("Neutral", "wasted_densification_rate"),
            ("Missed positive", "missed_positive_advantage_rate"),
        ):
            outcome_rows.append({"method": method, "outcome": outcome, "rate": row[field]})
    selected_comparisons = [
        row for row in development["comparisons"]
        if row["comparison"].endswith("_minus_entropy")
    ]
    with np.load(ROOT / "representations.npz", allow_pickle=False) as saved:
        representation_metadata = json.loads(str(saved["metadata"]))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": "Counterfactual Capacity Routing: Iteration 7",
        "decision": {
            "label": "NO-GO",
            "reason": "Large sparse-to-dense oracle headroom is not predictably encoded in the tested sparse hidden-state probes.",
            "oracle_gate_passed": oracle["gate_a"]["passed"],
            "representation_gate_passed": development["gate_b"]["passed"],
            "mmlu_pro_run": False, "runtime_run": False,
        },
        "scope": {
            "dataset": "ARC-Challenge train", "data_role": "development only",
            "examples": development["scope"]["examples"],
            "folds": development["scope"]["folds"],
            "model": oracle["metadata"]["slm_model"],
            "sparse_variants": ["s25", "s40", "s50"],
            "selected_sparse_variant": development["selected_sparse_variant"],
            "target_rates": development["scope"]["target_rates"],
            "bootstrap_replicates": development["scope"]["bootstrap_replicates"],
            "arc_validation_reused": False,
        },
        "static_rows": oracle["static_rows"],
        "oracle_rows": oracle_rows,
        "estimator_rows": estimator_rows,
        "all_estimator_rows": development["estimator_rows"],
        "policy_rows": policy_rows,
        "frontier_rows": frontier_rows,
        "layer_rows": layer_rows,
        "outcome_rows": outcome_rows,
        "random_summary": development["random_summary"],
        "selected_comparisons": selected_comparisons,
        "targets": development["targets"],
        "oracle_selection": oracle["selection"],
        "gate_a": oracle["gate_a"],
        "gate_b": development["gate_b"],
        "representation_metadata": representation_metadata,
        "omissions": {
            "ARC_validation": "Already consumed by Iteration 6; not reused for a new generalization claim.",
            "MMLU-Pro": "The ARC-development representation gate failed.",
            "runtime": "Runtime measurement was conditional on successful fresh-benchmark replication.",
            "combined_router": "Independent hidden advantage and predicted-instability signals did not both clear their development comparisons.",
            "recovery_adaptation": "Optional and not needed to answer the primary zero-shot representation question.",
        },
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(ROOT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
