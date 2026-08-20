"""Consolidate Iteration 4 Stage-1 results for reporting."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tuac.metrics import spearman_correlation


ROOT = Path("runs/iter4_arc")


def main() -> int:
    policies = json.loads((ROOT / "policies.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in benchmark["records"]:
        grouped[row["name"]].append(row)
    configuration_rows = []
    for name, rows in grouped.items():
        rates = sorted({row["target_rate"] for row in rows})
        rate_rows = [next(row for row in rows if row["target_rate"] == rate and row["cost_coefficient"] == 0.0) for rate in rates]
        configuration_rows.append({
            "name": name,
            "family": rows[0]["family"],
            "budget": rows[0]["budget"],
            "actual_average_bits": rows[0]["actual_average_bits"],
            "standalone_accuracy": rows[0]["standalone_accuracy"],
            "mean_cascade_accuracy": float(np.mean([row["cascade_accuracy"] for row in rate_rows])),
            "mean_delegation_rate": float(np.mean([row["delegation_rate"] for row in rate_rows])),
            "mean_cascade_utility": float(np.mean([row["cascade_utility"] for row in rows])),
            "prediction_agreement_with_fp": rows[0]["prediction_agreement_with_fp"],
            "output_kl_from_fp": rows[0]["output_kl_from_fp"],
            "uncertainty_spearman_with_fp": rows[0]["uncertainty_spearman_with_fp"],
            "compression_ratio_vs_fp16": rows[0]["compression_ratio_vs_fp16"],
        })
    non_fp = [row for row in configuration_rows if row["family"] != "fp"]
    standalone_utility_rho = spearman_correlation(
        np.asarray([row["standalone_accuracy"] for row in non_fp]),
        np.asarray([row["mean_cascade_utility"] for row in non_fp]),
    )

    comparison_rows = []
    for budget, result in benchmark["comparisons"].items():
        for comparison, label in (
            ("cuaq_vs_kl", "CUAQ minus KL"),
            ("cuaq_vs_random_mean", "CUAQ minus random mean"),
            ("cuaq_vs_best_random", "CUAQ minus best random"),
        ):
            row = result[comparison]
            comparison_rows.append({
                "budget": float(budget), "comparison": label, "difference": row["difference"],
                "ci_low": row["bootstrap_95_ci"][0], "ci_high": row["bootstrap_95_ci"][1],
            })

    allocation_rows = []
    for budget, methods in policies["policies"].items():
        for method in ("kl", "cuaq"):
            for index, bit in enumerate(methods[method]["bits"]):
                allocation_rows.append({
                    "budget": float(budget), "method": method.upper(), "layer": index,
                    "layer_name": policies["metadata"].get("student_model", "SLM") + f" L{index}", "bits": bit,
                })

    outcome_rows = []
    keep_names = {"full_precision", "uniform_4.0", "kl_3.5", "cuaq_3.5", "kl_4.0", "cuaq_4.0"}
    for row in benchmark["records"]:
        if row["name"] in keep_names and row["cost_coefficient"] == 0.025:
            outcome_rows.append({
                "name": row["name"], "target_rate": row["target_rate"],
                "cascade_accuracy": row["cascade_accuracy"], "delegation_rate": row["delegation_rate"],
                "cascade_utility": row["cascade_utility"],
                "fallback_correction_rate": row["fallback_correction_rate"],
                "harmful_escalation_rate": row["harmful_escalation_rate"],
                "missed_beneficial_delegation_rate": row["missed_beneficial_delegation_rate"],
            })

    sensitivity_rows = []
    for row in policies["layer_rows"]:
        sensitivity_rows.append({
            "bit": row["bit"], "layer": row["layer_index"],
            "kl_sensitivity": row["kl_sensitivity"], "cuaq_sensitivity": row["cuaq_sensitivity"],
            "uq_sensitivity": row["uq_sensitivity"], "dbaq_sensitivity": row["dbaq_sensitivity"],
        })

    for budget in (3.5, 4.0):
        left = policies["policies"][str(budget)]["kl"]["bits"]
        right = policies["policies"][str(budget)]["cuaq"]["bits"]
        for row in comparison_rows:
            if row["budget"] == budget:
                row["allocation_hamming_kl_vs_cuaq"] = sum(a != b for a, b in zip(left, right))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": "Cascade-Aware Compression: Iteration 4",
        "decision": {
            "label": "NO-GO",
            "stage1_gate_passed": benchmark["gate"]["passed"],
            "next_stage": benchmark["gate"]["next_stage"],
            "reason": "Local CUAQ did not beat KL-sensitive allocation at either held-out ARC budget.",
            "dbaq_proxy_computed": True,
            "dbaq_policy_evaluated": False,
            "carq_search_run": False,
            "mmlu_pro_run": False,
            "real_quantization_run": False,
        },
        "scope": {
            **policies["scope"],
            **benchmark["scope"],
            "calibration_examples": 750,
            "evaluation_examples": 299,
            "layers": 28,
            "probe_cells": 84,
            "unique_mixed_allocations_evaluated": len({row["probability_key"] for row in benchmark["configurations"] if row["family"] not in {"fp", "uniform"}}),
        },
        "models": benchmark["models"],
        "standalone_utility_spearman": standalone_utility_rho,
        "allocation_rank_correlations": policies["allocation_rank_correlations"],
        "delegation_value_counts": policies["delegation_value_counts"],
        "configuration_rows": sorted(configuration_rows, key=lambda row: (row["budget"], row["family"], row["name"])),
        "comparison_rows": comparison_rows,
        "allocation_rows": allocation_rows,
        "outcome_rows": outcome_rows,
        "sensitivity_rows": sensitivity_rows,
        "gate": benchmark["gate"],
        "omissions": {
            "DBAQ held-out evaluation/CARQ": "Stage 1 CUAQ gate failed on held-out ARC; the zero-inference DBAQ proxy diagnostic was retained.",
            "MMLU-Pro": "Plan permits replication only for methods that survive ARC.",
            "real_quantization": "Plan permits packed quantization only after a positive MMLU-Pro result.",
            "external_MPQ": "Not required by the minimum comparison and Stage 1 stopped the direction.",
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(ROOT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
