"""Consolidate Iteration 5 calibration search and held-out ARC results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter5_arc")


def main() -> int:
    search = json.loads((ROOT / "search/results.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    candidate_rows = []
    exact_rows = []
    allocation_rows = []
    for budget, result in search["budgets"].items():
        selected = result["selection"]
        for row in result["candidates"]:
            screen = row.get("screening_metrics")
            status = (
                "C3Q" if row["candidate_id"] == selected["c3q"] else
                "KL" if row["candidate_id"] == selected["kl"] else
                "Exact safe finalist" if bool(row.get("safe_joint")) else
                "Exact rejected finalist" if "metrics" in row else
                "Surrogate safe" if bool(row.get("screening_safe_joint")) else
                "Surrogate rejected"
            )
            candidate_rows.append({
                "name": row["candidate_id"], "budget": float(budget), "family": row["family"],
                "changed_layers": row["changed_layers"], "generation_seed": row["generation_seed"],
                "screening_safe": bool(row.get("screening_safe_joint")),
                "screening_accuracy": screen["standalone_accuracy"] if screen else None,
                "screening_kl": screen["output_kl_from_fp"] if screen else None,
                "screening_utility": screen["mean_utility"] if screen else None,
                "exact_finalist": "metrics" in row,
                "exact_safe": bool(row.get("safe_joint")),
                "status": status,
                "selected": "C3Q" if row["candidate_id"] == selected["c3q"] else "KL" if row["candidate_id"] == selected["kl"] else "finalist" if "metrics" in row else "screened",
            })
            if "metrics" in row:
                metrics = row["metrics"]
                exact_rows.append({
                    "name": row["candidate_id"], "budget": float(budget), "family": row["family"],
                    "selected": "C3Q" if row["candidate_id"] == selected["c3q"] else "KL" if row["candidate_id"] == selected["kl"] else "Finalist",
                    "safe": bool(row["safe_joint"]), "standalone_accuracy": metrics["standalone_accuracy"],
                    "output_kl": metrics["output_kl_from_fp"], "mean_utility": metrics["mean_utility"],
                    "agreement_with_fp": metrics["prediction_agreement_with_fp"],
                    "error_auroc": metrics["error_auroc"], "aurc": metrics["aurc"],
                })
        lookup = {row["candidate_id"]: row for row in result["candidates"]}
        for method, candidate_id in (("KL", selected["kl"]), ("C3Q", selected["c3q"])):
            for layer, bit in enumerate(lookup[candidate_id]["bits"]):
                allocation_rows.append({"budget": float(budget), "method": method, "series": f"{method} / {budget}", "layer": layer, "bits": bit})

    heldout_rows, frontier_rows, outcome_rows = [], [], []
    for config in benchmark["configurations"]:
        metrics = config["metrics"]
        heldout_rows.append({
            "name": config["name"], "family": config["family"], "budget": config["budget"],
            "standalone_accuracy": metrics["standalone_accuracy"], "mean_utility": metrics["mean_utility"],
            "output_kl": metrics["output_kl_from_fp"], "agreement_with_fp": metrics["prediction_agreement_with_fp"],
            "uncertainty_spearman": metrics["uncertainty_spearman_with_fp"], "error_auroc": metrics["error_auroc"],
        })
        for rate, routing in metrics["routing"].items():
            frontier_rows.append({"name": config["name"], "family": config["family"], "target_rate": float(rate), "delegation_rate": routing["delegation_rate"], "cascade_accuracy": routing["cascade_accuracy"]})
            if rate == "0.3" and config["family"] in {"kl", "c3q"}:
                for label, field in (("Fallback correction", "fallback_correction_rate"), ("Harmful escalation", "harmful_escalation_rate"), ("Missed beneficial delegation", "missed_beneficial_delegation_rate")):
                    outcome_rows.append({"name": config["name"], "outcome": label, "rate": routing[field]})

    comparison = benchmark["comparisons"]["3.5"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": "Capability-Constrained Cascade Quantization: Iteration 5",
        "decision": {
            "label": "NO-GO",
            "reason": "C3Q preserved held-out standalone accuracy but reduced mean cascade utility relative to KL.",
            "arc_passed": comparison["arc_positive"],
            "mmlu_pro_run": False,
            "real_quantization_run": False,
        },
        "scope": {
            **search["scope"], **benchmark["scope"],
            "generated_candidates": 200, "exact_finalists": sum(row["exact_finalist"] for row in candidate_rows),
            "execution_adaptation": "All candidates used the additive layer-probe surrogate; six frozen finalists per budget received exact 750-example whole-model evaluation.",
        },
        "search_gate": search["gate"],
        "budget_diagnostics": {budget: {key: value for key, value in row.items() if key in {"candidate_count", "exact_candidate_count", "safe_counts", "limits", "headroom", "selection", "seed_robustness"}} for budget, row in search["budgets"].items()},
        "heldout_gate": benchmark["gate"],
        "heldout_comparison": comparison,
        "candidate_rows": candidate_rows,
        "exact_rows": exact_rows,
        "heldout_rows": heldout_rows,
        "frontier_rows": frontier_rows,
        "allocation_rows": allocation_rows,
        "outcome_rows": outcome_rows,
        "omissions": {
            "full_exact_candidate_pool": "A literal exact 200-candidate calibration run was projected to exceed practical runtime on the 8GB device; all candidates were screened with cached layer probes and only frozen finalists were scored exactly.",
            "best_safe_random": "The frozen random finalist failed the exact joint guardrail; no best-safe-random held-out comparison was available.",
            "MMLU-Pro": "The ARC gate failed.",
            "real_quantization": "Packed quantization was conditional on positive ARC and MMLU-Pro results.",
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(ROOT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
