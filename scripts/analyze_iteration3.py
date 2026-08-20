"""Consolidate the ARC and MMLU-Pro Iteration 3 analyses for reporting."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ARC = Path("runs/iter3_arc")
MMLU = Path("runs/iter3_mmlu_pro")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    arc = json.loads((ARC / "analysis.json").read_text(encoding="utf-8"))
    mmlu = json.loads((MMLU / "analysis.json").read_text(encoding="utf-8"))
    mmlu_drift_only = json.loads((MMLU / "drift_analysis.json").read_text(encoding="utf-8"))

    precision_rows: list[dict] = []
    for dataset, analysis in (("ARC-Challenge", arc), ("MMLU-Pro", mmlu)):
        metrics = {row["precision"]: row for row in analysis["small_model_rows"]}
        for row in analysis["drift_rows"]:
            precision_rows.append(
                {
                    "dataset": dataset,
                    "precision": row["precision"],
                    "bit_width": {"fp": 16, "q8": 8, "q4": 4, "q2": 2}[row["precision"]],
                    "accuracy": metrics[row["precision"]]["accuracy"],
                    "error_auroc": metrics[row["precision"]]["error_auroc"],
                    "uncertainty_spearman": row["uncertainty_spearman"],
                    "ordering_inversion_rate": row["pairwise_ordering_inversion"],
                    "prediction_agreement": row["prediction_agreement_with_fp"],
                    "top20_overlap": row["top20_overlap"],
                }
            )

    # Exact matched-rate ARC frontiers. Random is summarized over its ten seeds.
    frontier_rows: list[dict] = []
    random_groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in arc["routing_rows"]:
        if row["method"] == "random":
            random_groups[(row["precision"], row["target_rate"])].append(row["cascade_accuracy"])
        elif row["method"] in {"matched_rate", "cascade_oracle"}:
            frontier_rows.append(
                {
                    "precision": row["precision"],
                    "method": "uncertainty" if row["method"] == "matched_rate" else "oracle",
                    "target_rate": row["target_rate"],
                    "delegation_rate": row["delegation_rate"],
                    "cascade_accuracy": row["cascade_accuracy"],
                }
            )
    for (precision, rate), values in sorted(random_groups.items()):
        frontier_rows.append(
            {
                "precision": precision,
                "method": "random_mean_10_seeds",
                "target_rate": rate,
                "delegation_rate": rate,
                "cascade_accuracy": mean(values),
            }
        )

    mmlu_frontier_rows: list[dict] = []
    mmlu_random_groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in mmlu["routing_rows"]:
        if row["method"] == "random":
            mmlu_random_groups[(row["precision"], row["target_rate"])].append(row["cascade_accuracy"])
        elif row["method"] in {"matched_rate", "cascade_oracle"}:
            mmlu_frontier_rows.append(
                {"precision": row["precision"], "method": "uncertainty" if row["method"] == "matched_rate" else "oracle", "target_rate": row["target_rate"], "delegation_rate": row["delegation_rate"], "cascade_accuracy": row["cascade_accuracy"]}
            )
    for (precision, rate), values in sorted(mmlu_random_groups.items()):
        mmlu_frontier_rows.append({"precision": precision, "method": "random_mean_10_seeds", "target_rate": rate, "delegation_rate": rate, "cascade_accuracy": mean(values)})

    recalibration_rows = [
        {
            "precision": row["precision"],
            "target_rate": row["target_rate"],
            "method": row["method"],
            "delegation_rate": row["delegation_rate"],
            "cascade_accuracy": row["cascade_accuracy"],
        }
        for row in arc["routing_rows"]
        if row["precision"] != "fp"
        and row["method"] in {
            "frozen_fp_threshold", "retuned_threshold", "temperature", "isotonic", "matched_rate"
        }
    ]

    dfr_rows: list[dict] = []
    for row in arc["flip_rows"]:
        dfr_rows.append(
            {
                "dataset": "ARC-Challenge",
                "method": "frozen_fp_threshold",
                "precision": row["precision"],
                "target_rate": row["target_rate"],
                "delegation_flip_rate": row["delegation_flip_rate"],
            }
        )
    for row in mmlu_drift_only["decision_rows"]:
        dfr_rows.append(
            {
                "dataset": "MMLU-Pro",
                "method": row["method"],
                "precision": row["precision"],
                "target_rate": row["target_rate"],
                "delegation_flip_rate": row["delegation_flip_rate"],
            }
        )

    flip_taxonomy_rows: list[dict] = []
    for row in arc["flip_rows"]:
        for category, field in (
            ("Harmful false accept", "harmful_false_accept_rate"),
            ("Wasteful false escalation", "wasteful_false_escalation_rate"),
            ("Beneficial escalation", "beneficial_escalation_rate"),
            ("Benign/other", "benign_or_other_flip_rate"),
        ):
            flip_taxonomy_rows.append(
                {
                    "precision": row["precision"],
                    "target_rate": row["target_rate"],
                    "category": category,
                    "rate": row[field],
                }
            )

    routing_only_rows = []
    for dataset, analysis in (("ARC-Challenge", arc), ("MMLU-Pro", mmlu)):
        for key, value in analysis["routing_only_comparisons"].items():
            precision, rate = key.split(":")
            routing_only_rows.append({
                "dataset": dataset,
                "precision": precision,
                "target_rate": float(rate),
                "fp_ranked_minus_q_ranked_accuracy": value[
                    "fp_ranked_minus_quantized_ranked_accuracy"
                ],
                "ci_low": value["bootstrap_95_ci"][0],
                "ci_high": value["bootstrap_95_ci"][1],
                "mcnemar_p": value["mcnemar"]["exact_p"],
            })

    accuracy_preserved_rows = []
    for row in arc["accuracy_preserved_rows"]:
        accuracy_preserved_rows.append({"dataset": "ARC-Challenge", **row})
    for row in mmlu["accuracy_preserved_rows"]:
        accuracy_preserved_rows.append({"dataset": "MMLU-Pro", **row})

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": "When Compression Changes Who Gets Help: Iteration 3",
        "decision": arc["decision"],
        "arc": {
            "status": "complete",
            "scope": arc["scope"],
            "metadata": arc["metadata"],
            "router_signal": arc["router"]["selected_signal"],
            "router_aurocs": arc["router"]["signal_aurocs"],
            "reference_metrics": arc["reference_metrics"],
        },
        "mmlu_pro": {
            "status": "complete",
            "scope": mmlu["scope"],
            "metadata": mmlu["metadata"],
            "router_signal": mmlu["router"]["selected_signal"],
            "router_aurocs": mmlu["router"]["signal_aurocs"],
            "reference_metrics": mmlu["reference_metrics"],
            "decision": mmlu["decision"],
        },
        "precision_rows": precision_rows,
        "dfr_rows": dfr_rows,
        "frontier_rows": frontier_rows,
        "mmlu_frontier_rows": mmlu_frontier_rows,
        "recalibration_rows": recalibration_rows,
        "flip_taxonomy_rows": flip_taxonomy_rows,
        "routing_only_rows": sorted(routing_only_rows, key=lambda r: (r["dataset"], r["precision"], r["target_rate"])),
        "accuracy_preserved_rows": accuracy_preserved_rows,
        "proximity_rows": arc["proximity_rows"],
        "claim_boundary": {
            "routing_drift_replicated": True,
            "routing_only_system_harm_established": False,
            "real_quantization_run": False,
            "deployment_claims_supported": False,
            "mmlu_cascade_completed": True,
        },
    }
    (ARC / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(ARC / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
