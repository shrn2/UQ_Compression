"""Create the reviewed, report-ready summary for Iteration 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.metrics import spearman_correlation

ROOT = Path("runs/iter2_arc")


def minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high == low else (values - low) / (high - low)


def main() -> int:
    analysis = json.loads((ROOT / "layer_analysis.json").read_text(encoding="utf-8"))
    rows = analysis["layer_rows"]
    scatter_rows = []
    depth_rows = []
    taxonomy_rows = []
    robustness_rows = []
    for bit in analysis["probe_bits"]:
        selected = [row for row in rows if row["bits"] == bit]
        capability = np.asarray([row["capability_kl"] for row in selected])
        trust = np.asarray([row["trust_entropy_rank_disruption"] for row in selected])
        cap_normalized, trust_normalized = minmax(capability), minmax(trust)
        for index, row in enumerate(selected):
            if bit in (2, 4):
                scatter_rows.append(
                    {
                        "bits": bit,
                        "layer": f"L{row['layer_index']}",
                        "layer_index": row["layer_index"],
                        "region": row["region"],
                        "capability_kl": row["capability_kl"],
                        "trust_rank_disruption": row["trust_entropy_rank_disruption"],
                        "taxonomy": row["taxonomy"],
                        "rank_displacement": row["rank_displacement"],
                    }
                )
                depth_rows.extend(
                    [
                        {
                            "bits": bit,
                            "layer": f"L{row['layer_index']}",
                            "layer_index": row["layer_index"],
                            "sensitivity": "Capability (KL)",
                            "normalized_score": float(cap_normalized[index]),
                        },
                        {
                            "bits": bit,
                            "layer": f"L{row['layer_index']}",
                            "layer_index": row["layer_index"],
                            "sensitivity": "Trust (entropy rank)",
                            "normalized_score": float(trust_normalized[index]),
                        },
                    ]
                )
            taxonomy_rows.append(
                {
                    "bits": bit,
                    "layer": f"L{row['layer_index']}",
                    "region": row["region"],
                    "taxonomy": row["taxonomy"].replace("_", " ").title(),
                    "capability_rank": row["capability_rank"],
                    "trust_rank": row["trust_rank"],
                    "rank_displacement": row["rank_displacement"],
                    "capability_top_frequency": row["capability_top_quartile_frequency"],
                    "trust_top_frequency": row["trust_top_quartile_frequency"],
                    "stable_trust_only": row["stable_trust_only"],
                }
            )
        diagnostic = analysis["ranking_diagnostics"][str(bit)]
        correct = np.asarray([row["trust_correct_rank_disruption"] for row in selected])
        incorrect = np.asarray([row["trust_incorrect_rank_disruption"] for row in selected])
        robustness_rows.append(
            {
                "bits": bit,
                "capability_trust_rho": diagnostic["capability_trust_spearman"],
                "rho_ci_low": diagnostic["capability_trust_spearman_bootstrap_95_ci"][0],
                "rho_ci_high": diagnostic["capability_trust_spearman_bootstrap_95_ci"][1],
                "capability_margin_rho": diagnostic["capability_margin_spearman"],
                "entropy_margin_map_rho": diagnostic["entropy_margin_trust_ranking_spearman"],
                "correct_incorrect_map_rho": spearman_correlation(correct, incorrect),
                "top3_overlap": diagnostic["top_k_overlap"]["3"],
                "top5_overlap": diagnostic["top_k_overlap"]["5"],
                "top8_overlap": diagnostic["top_k_overlap"]["8"],
                "stable_trust_only_layers": sum(
                    bool(row["stable_trust_only"]) for row in selected
                ),
            }
        )

    trust_by_bit = {
        bit: np.asarray(
            [row["trust_entropy_rank_disruption"] for row in rows if row["bits"] == bit]
        )
        for bit in analysis["probe_bits"]
    }
    cross_precision = [
        {
            "comparison": "2-bit vs 4-bit trust map",
            "spearman": spearman_correlation(trust_by_bit[2], trust_by_bit[4]),
        },
        {
            "comparison": "4-bit vs 8-bit trust map",
            "spearman": spearman_correlation(trust_by_bit[4], trust_by_bit[8]),
        },
    ]
    output = {
        "scope": {
            "dataset": "ARC-Challenge",
            "probing_examples": analysis["example_count"],
            "evaluation_examples_reserved": 299,
            "student_model": analysis["metadata"]["student_model"],
            "teacher_role": "exploratory agreement metric only; never used for allocation",
            "probe_bits": analysis["probe_bits"],
            "layers": len(analysis["layer_names"]),
            "group_size": analysis["metadata"]["group_size"],
            "quantization": analysis["metadata"]["quantization"],
            "bootstrap_replicates": analysis["bootstrap_replicates"],
        },
        "decision": {
            "label": "NO-GO",
            "gate_passed": analysis["gate"]["passed"],
            "rule": analysis["gate"]["rule"],
            "policy_benchmark_run": False,
            "mmlu_pro_run": False,
            "rationale": (
                "Capability and trust maps are nearly identical at meaningful precisions; "
                "the preregistered ranking-difference gate failed, so downstream policy and MMLU-Pro runs were skipped."
            ),
            "recommended_pivot": "Quantization-induced whole-model uncertainty and delegation drift.",
        },
        "ranking_diagnostics": analysis["ranking_diagnostics"],
        "robustness_rows": robustness_rows,
        "cross_precision_rows": cross_precision,
        "region_rows": analysis["region_rows"],
        "scatter_rows": scatter_rows,
        "depth_rows": depth_rows,
        "taxonomy_rows": taxonomy_rows,
        "layer_rows": rows,
    }
    (ROOT / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": output["decision"]["label"],
                "gate_passed": output["decision"]["gate_passed"],
                "rho_2bit": analysis["ranking_diagnostics"]["2"]["capability_trust_spearman"],
                "rho_4bit": analysis["ranking_diagnostics"]["4"]["capability_trust_spearman"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
