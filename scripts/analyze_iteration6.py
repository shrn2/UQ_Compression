"""Consolidate Iteration 6 calibration and held-out U-Dense results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter6_arc")


def main() -> int:
    analysis = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    estimator_rows = analysis["estimator_rows"]
    static_rows = analysis["static_rows"]
    event_rows = [
        {"event": name, **values} for name, values in analysis["event_counts"].items()
    ]
    policy_rows = [
        row for row in benchmark["records"]
        if row["method"] not in {"error_oracle", "uplift_oracle"}
    ]
    oracle_rows = [
        row for row in benchmark["records"]
        if row["method"] in {"error_oracle", "uplift_oracle"}
    ]
    frontier_rows = [
        {
            "method": row["method"],
            "target_rate": row["target_rate"],
            "dense_execution_rate": row["dense_execution_rate"],
            "accuracy": row["accuracy"],
            "expected_sequential_compute": row["expected_sequential_compute"],
            "kind": "learned/uncertainty",
        }
        for row in policy_rows
    ]
    frontier_rows.extend(
        {
            "method": row["method"],
            "target_rate": row["target_rate"],
            "dense_execution_rate": row["dense_execution_rate"],
            "accuracy": row["accuracy"],
            "expected_sequential_compute": row["expected_sequential_compute"],
            "kind": "oracle diagnostic",
        }
        for row in oracle_rows
    )
    frontier_rows.extend([
        {"method": "aggressive_static", "target_rate": 0.0, "dense_execution_rate": 0.0,
         "accuracy": benchmark["static"]["aggressive_accuracy"], "expected_sequential_compute": benchmark["static"]["aggressive_compute"], "kind": "static"},
        {"method": "dense_static", "target_rate": 1.0, "dense_execution_rate": 1.0,
         "accuracy": benchmark["static"]["dense_accuracy"], "expected_sequential_compute": 1.0, "kind": "static capacity"},
    ])
    uplift_entropy = [
        row for row in benchmark["comparisons"]
        if row["comparison"] == "uplift_probe_minus_entropy"
    ]
    best_heldout = max(policy_rows, key=lambda row: row["accuracy"])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": "Uncertainty-Guided Progressive Densification: Iteration 6",
        "decision": {
            "label": "NO-GO",
            "reason": "The learned uplift probe did not beat generic uncertainty on calibration and did not statistically dominate entropy on held-out ARC.",
            "calibration_gate_passed": analysis["gate"]["passed"],
            "uplift_ablation_passed": analysis["gate"]["uplift_ablation_passed"],
            "heldout_gate_passed": benchmark["gate"]["passed"],
            "three_level_run": False,
            "mmlu_pro_run": False,
            "runtime_run": False,
        },
        "scope": {
            "calibration_examples": analysis["scope"]["examples"],
            "evaluation_examples": benchmark["scope"]["examples"],
            "folds": analysis["scope"]["folds"],
            "bootstrap_replicates": analysis["scope"]["bootstrap_replicates"],
            "target_rates": analysis["scope"]["target_rates"],
            "model": analysis["metadata"]["slm_model"],
            "dataset": "ARC-Challenge",
            "sparsity_levels": [0.0, analysis["metadata"]["medium_sparsity"], analysis["metadata"]["aggressive_sparsity"]],
            "physical_channel_pruning": analysis["metadata"]["physical_channel_pruning"],
        },
        "static_rows": static_rows,
        "event_rows": event_rows,
        "estimator_rows": estimator_rows,
        "policy_rows": policy_rows,
        "oracle_rows": oracle_rows,
        "frontier_rows": frontier_rows,
        "random_summary": benchmark["random_summary"],
        "uplift_entropy_comparisons": uplift_entropy,
        "calibration_gate": analysis["gate"],
        "static_gate": analysis["static_gate"],
        "heldout_gate": benchmark["gate"],
        "error_detection": analysis["error_detection"],
        "best_heldout_policy_point": best_heldout,
        "dense_reference_check": analysis["metadata"]["dense_reference_check"],
        "omissions": {
            "three_level_dynamic_policy": "The two-level held-out gate failed.",
            "MMLU-Pro": "Replication was conditional on a positive held-out ARC result.",
            "runtime": "Latency measurement was conditional on replicated policy benefit; compute values are analytical matrix-parameter proxies only.",
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
