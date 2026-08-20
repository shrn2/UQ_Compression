"""Independent integrity checks for Iteration 5."""

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
from tuac.scoring import kl_divergence


ROOT = Path("runs/iter5_arc")


def close(left, right) -> bool:
    return bool(np.isclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True))


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
    probes = MultiBitCalibrationArtifact.load("runs/iter4_arc/layer_probes.npz")
    search = json.loads((ROOT / "search/results.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "report/artifact.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "benchmark/probabilities.npz", allow_pickle=False) as saved:
        heldout_arrays = {key: saved[key] for key in saved.files}

    checks = {
        "calibration_count_750": len(calibration) == 750,
        "evaluation_count_299": len(evaluation) == 299,
        "splits_disjoint": {row.id for row in calibration}.isdisjoint({row.id for row in evaluation}),
        "cascade_calibration_ids_match": cascade.calibration_ids == tuple(row.id for row in calibration),
        "cascade_evaluation_ids_match": cascade.evaluation_ids == tuple(row.id for row in evaluation),
        "probe_ids_match": probes.example_ids == cascade.calibration_ids,
        "two_budgets": set(search["budgets"]) == {"3.5", "4.0"},
        "two_hundred_generated": sum(row["candidate_count"] for row in search["budgets"].values()) == 200,
        "twelve_exact_finalists": sum(row["exact_candidate_count"] for row in search["budgets"].values()) == 12,
        "three_generation_seeds": search["scope"]["generation_seeds"] == [11, 23, 37],
        "four_target_rates": search["scope"]["target_rates"] == [0.1, 0.2, 0.3, 0.5],
        "five_costs": search["scope"]["cost_coefficients"] == [0.0, 0.01, 0.025, 0.05, 0.1],
        "joint_guardrail": search["guardrail"]["kl_multiplier"] == 1.1 and search["guardrail"]["accuracy_tolerance"] == 0.01,
        "3_5_headroom": search["budgets"]["3.5"]["headroom"]["meaningful"] is True,
        "4_0_no_headroom": search["budgets"]["4.0"]["headroom"]["meaningful"] is False,
        "only_3_5_authorized": search["gate"]["authorized_budgets"] == ["3.5"],
        "heldout_arc_stopped": benchmark["gate"]["mmlu_authorized"] is False,
        "summary_no_go": summary["decision"]["label"] == "NO-GO",
        "later_stages_not_run": not summary["decision"]["mmlu_pro_run"] and not summary["decision"]["real_quantization_run"],
        "no_iter5_mmlu": not Path("runs/iter5_mmlu_pro").exists(),
        "no_packed_output": not (ROOT / "real_quantization").exists(),
        "report_html_exists": (ROOT / "report/report.html").is_file(),
        "report_html_nontrivial": (ROOT / "report/report.html").stat().st_size > 50_000,
        "report_five_charts": len(report["manifest"]["charts"]) == 5,
        "report_two_tables": len(report["manifest"]["tables"]) == 2,
    }

    # Candidate uniqueness, exact budgets, and exact guardrails.
    for budget, result in search["budgets"].items():
        candidates = result["candidates"]
        checks[f"candidate_count_{budget}"] = len(candidates) == 100
        checks[f"candidate_unique_{budget}"] = len({tuple(row["bits"]) for row in candidates}) == 100
        target_sum = round(float(budget) * 28)
        for index, row in enumerate(candidates):
            checks[f"budget_sum_{budget}_{index}"] = sum(row["bits"]) == target_sum
        exact = [row for row in candidates if "metrics" in row]
        kl = next(row for row in exact if row["candidate_id"] == result["selection"]["kl"])
        kl_limit = kl["metrics"]["output_kl_from_fp"] * 1.1
        accuracy_floor = kl["metrics"]["standalone_accuracy"] - 0.01
        for row in exact:
            expected = row["metrics"]["output_kl_from_fp"] <= kl_limit + 1e-12 and row["metrics"]["standalone_accuracy"] >= accuracy_floor - 1e-12
            checks[f"guardrail_{budget}_{row['candidate_id']}"] = row["safe_joint"] == expected

    # Recompute all held-out metrics from the saved probabilities and frozen calibration thresholds.
    search_budget = search["budgets"]["3.5"]
    candidate_lookup = {row["candidate_id"]: row for row in search_budget["candidates"]}
    calibration_lookup = {
        "kl": candidate_lookup[search_budget["selection"]["kl"]]["metrics"],
        "c3q": candidate_lookup[search_budget["selection"]["c3q"]]["metrics"],
        **{name: row["metrics"] for name, row in search_budget["baselines"].items()},
    }
    utility_vectors = {}
    for config in benchmark["configurations"]:
        probabilities = heldout_arrays[config["name"]]
        metrics = config["metrics"]
        errors = probabilities.argmax(axis=1) != cascade.evaluation_labels
        checks[f"accuracy_{config['name']}"] = close(np.mean(~errors), metrics["standalone_accuracy"])
        checks[f"kl_{config['name']}"] = close(np.mean(kl_divergence(cascade.slm_fp_evaluation, probabilities)), metrics["output_kl_from_fp"])
        scores = uncertainty_scores(probabilities)[search["scope"]["router_signal"]]
        vectors = []
        for rate in search["scope"]["target_rates"]:
            rate_key = str(float(rate))
            threshold = calibration_lookup[config["family"]]["thresholds"][rate_key]
            observed, correct, decisions = routing_outcomes(probabilities, cascade.fallback_evaluation, cascade.evaluation_labels, scores, threshold)
            stored = metrics["routing"][rate_key]
            checks[f"cascade_accuracy_{config['name']}_{rate_key}"] = close(observed["cascade_accuracy"], stored["cascade_accuracy"])
            checks[f"delegation_{config['name']}_{rate_key}"] = close(observed["delegation_rate"], stored["delegation_rate"])
            for cost in search["scope"]["cost_coefficients"]:
                vectors.append(correct.astype(float) - cost * decisions.astype(float))
        utility_vectors[config["name"]] = np.stack(vectors)
        checks[f"mean_utility_{config['name']}"] = close(np.mean(vectors), metrics["mean_utility"])
    observed_difference = float(np.mean(utility_vectors["c3q_3.5"] - utility_vectors["kl_3.5"]))
    checks["paired_difference_recomputed"] = close(observed_difference, benchmark["comparisons"]["3.5"]["c3q_vs_kl_mean_utility"]["difference"])
    checks["c3q_heldout_utility_not_above_kl"] = observed_difference <= 0
    checks["capability_gap_under_1pp"] = abs(
        next(row for row in benchmark["configurations"] if row["name"] == "c3q_3.5")["metrics"]["standalone_accuracy"]
        - next(row for row in benchmark["configurations"] if row["name"] == "kl_3.5")["metrics"]["standalone_accuracy"]
    ) < 0.01

    source_lookup = {source["id"]: source for source in report["manifest"]["sources"]}
    checks["all_charts_have_sql_sources"] = all(source_lookup[chart["sourceId"]].get("query", {}).get("sql") for chart in report["manifest"]["charts"])
    checks["all_tables_have_sql_sources"] = all(source_lookup[table["sourceId"]].get("query", {}).get("sql") for table in report["manifest"]["tables"])
    block_index = {block["id"]: index for index, block in enumerate(report["manifest"]["blocks"])}
    for chart in report["manifest"]["charts"]:
        chart_blocks = [block for block in report["manifest"]["blocks"] if block.get("chartId") == chart["id"]]
        checks[f"chart_{chart['id']}_adjacent_narrative"] = len(chart_blocks) == 1 and block_index[chart_blocks[0]["id"]] > 0 and report["manifest"]["blocks"][block_index[chart_blocks[0]["id"]] - 1]["type"] == "markdown"
    checks["required_report_sections"] = all(any(label in block.get("body", "") for block in report["manifest"]["blocks"]) for label in ("Technical summary", "Scope, data, and metric definitions", "Methodology and experimental design", "Limitations, uncertainty, and robustness", "Recommended next steps", "Further questions"))

    checks = {key: bool(value) for key, value in checks.items()}
    key_files = [ROOT / "search/results.json", ROOT / "benchmark/probabilities.npz", ROOT / "benchmark/results.json", ROOT / "summary.json", ROOT / "report/artifact.json", ROOT / "report/report.html"]
    output = {
        "assessment": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "all_checks_passed": all(checks.values()),
        "check_count": len(checks),
        "checks": checks,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "report_package_verification": "structural_only",
        "sha256": {str(path): digest(path) for path in key_files},
        "material_caveats": [
            "All 200 candidates used an additive one-layer-probe surrogate; exact claims cover only 12 frozen finalists.",
            "No frozen random finalist passed the exact guardrail, so best-safe-random held-out performance is unavailable.",
            "The paired C3Q-minus-KL interval crosses zero, but all four rate-specific point estimates favor KL.",
            "Fake quantization does not establish packed size, latency, memory, or throughput.",
            "Portable report packaging passed structurally; browser interaction and viewport QA were unavailable.",
        ],
    }
    (ROOT / "validation.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"assessment": output["assessment"], "passed": sum(checks.values()), "total": len(checks)}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
