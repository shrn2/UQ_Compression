"""Independent integrity and reproducibility checks for Iteration 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from tuac.artifacts import MultiBitCalibrationArtifact
from tuac.data import load_jsonl
from tuac.metrics import (
    classification_metrics,
    margin_uncertainty,
    pairwise_ordering_flip_rate,
    spearman_correlation,
)
from tuac.scoring import entropy, kl_divergence

ROOT = Path("runs/iter2_arc")
SOURCE = Path("runs/iter1_arc")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def valid_probabilities(values: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.allclose(values.sum(axis=-1), 1.0, atol=1e-5)
    )


def top_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    count = min(k, left.size)
    return len(set(np.argsort(left)[-count:]) & set(np.argsort(right)[-count:])) / count


def close(left, right) -> bool:
    return bool(np.isclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True))


def main() -> int:
    calibration = load_jsonl(SOURCE / "calibration_pool.jsonl")
    evaluation = load_jsonl(SOURCE / "evaluation.jsonl")
    artifact = MultiBitCalibrationArtifact.load(SOURCE / "calibration.npz")
    analysis = json.loads((ROOT / "layer_analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "report/artifact.json").read_text(encoding="utf-8"))
    calibration_ids = {item.id for item in calibration}
    evaluation_ids = {item.id for item in evaluation}
    checks: dict[str, bool] = {
        "calibration_count_750": len(calibration) == 750,
        "reserved_validation_count_299": len(evaluation) == 299,
        "split_ids_disjoint": calibration_ids.isdisjoint(evaluation_ids),
        "artifact_examples_match_calibration": set(artifact.example_ids) == calibration_ids,
        "artifact_shape_3bits_24layers_750examples": artifact.quantized_probabilities.shape[:3] == (3, 24, 750),
        "candidate_bits_2_4_8": artifact.probe_bits == (2, 4, 8),
        "student_probabilities_valid": valid_probabilities(artifact.student_probabilities),
        "teacher_probabilities_valid": valid_probabilities(artifact.teacher_probabilities),
        "probe_probabilities_valid": valid_probabilities(artifact.quantized_probabilities),
        "analysis_uses_all_examples": analysis["example_count"] == 750,
        "bootstrap_replicates_1000": analysis["bootstrap_replicates"] == 1000,
        "bootstrap_seed_fixed": analysis["bootstrap_seed"] == 20260817,
        "analysis_has_72_layer_rows": len(analysis["layer_rows"]) == 72,
        "analysis_has_9_region_rows": len(analysis["region_rows"]) == 9,
        "gate_failed": analysis["gate"]["passed"] is False,
        "summary_decision_no_go": summary["decision"]["label"] == "NO-GO",
        "policy_benchmark_declared_skipped": summary["decision"]["policy_benchmark_run"] is False,
        "mmlu_declared_skipped": summary["decision"]["mmlu_pro_run"] is False,
        "no_policy_file_after_failed_gate": not (ROOT / "policies.json").exists(),
        "no_benchmark_after_failed_gate": not (ROOT / "benchmark").exists(),
        "report_has_three_charts": len(report["manifest"]["charts"]) == 3,
        "report_has_two_tables": len(report["manifest"]["tables"]) == 2,
        "report_has_fifteen_blocks": len(report["manifest"]["blocks"]) == 15,
        "report_html_exists": (ROOT / "report/report.html").is_file(),
        "report_html_nontrivial": (ROOT / "report/report.html").stat().st_size > 50_000,
    }
    labels = artifact.labels
    fp = artifact.student_probabilities
    fp_entropy = entropy(fp, normalized=True)
    fp_margin = margin_uncertainty(fp)
    fp_predictions = fp.argmax(axis=1)
    fp_metrics = classification_metrics(fp, labels)
    fp_correct = fp_predictions == labels
    layer_lookup = {(row["bits"], row["layer_index"]): row for row in analysis["layer_rows"]}
    capability_by_bit: dict[int, np.ndarray] = {}
    trust_by_bit: dict[int, np.ndarray] = {}
    for bit_index, bit in enumerate(artifact.probe_bits):
        capability = np.empty(len(artifact.layer_names))
        trust = np.empty(len(artifact.layer_names))
        for layer_index in range(len(artifact.layer_names)):
            row = layer_lookup[(bit, layer_index)]
            probabilities = artifact.quantized_probabilities[bit_index, layer_index]
            current_entropy = entropy(probabilities, normalized=True)
            current_margin = margin_uncertainty(probabilities)
            current_metrics = classification_metrics(probabilities, labels)
            capability[layer_index] = np.mean(kl_divergence(fp, probabilities))
            trust[layer_index] = 1 - spearman_correlation(fp_entropy, current_entropy)
            expected = {
                "capability_kl": capability[layer_index],
                "capability_flip": np.mean(probabilities.argmax(axis=1) != fp_predictions),
                "capability_accuracy_drop": fp_metrics["accuracy"] - current_metrics["accuracy"],
                "trust_entropy_rank_disruption": trust[layer_index],
                "trust_margin_rank_disruption": 1 - spearman_correlation(fp_margin, current_margin),
                "trust_auroc_drop": fp_metrics["error_auroc"] - current_metrics["error_auroc"],
                "trust_aurc_increase": current_metrics["aurc"] - fp_metrics["aurc"],
                "trust_ece_increase": current_metrics["ece"] - fp_metrics["ece"],
                "trust_brier_increase": current_metrics["brier"] - fp_metrics["brier"],
                "trust_pairwise_ordering_flip": pairwise_ordering_flip_rate(fp_entropy, current_entropy),
                "trust_correct_rank_disruption": 1 - spearman_correlation(fp_entropy[fp_correct], current_entropy[fp_correct]),
                "trust_incorrect_rank_disruption": 1 - spearman_correlation(fp_entropy[~fp_correct], current_entropy[~fp_correct]),
            }
            for field, value in expected.items():
                checks[f"recompute_{bit}bit_L{layer_index}_{field}"] = close(row[field], value)
            checks[f"ci_order_capability_{bit}bit_L{layer_index}"] = row["capability_kl_ci_low"] <= row["capability_kl_ci_high"]
            checks[f"ci_order_trust_{bit}bit_L{layer_index}"] = row["trust_entropy_ci_low"] <= row["trust_entropy_ci_high"]
            checks[f"frequency_bounds_{bit}bit_L{layer_index}"] = (
                0 <= row["capability_top_quartile_frequency"] <= 1
                and 0 <= row["trust_top_quartile_frequency"] <= 1
            )
        capability_by_bit[bit] = capability
        trust_by_bit[bit] = trust
        diagnostic = analysis["ranking_diagnostics"][str(bit)]
        observed = spearman_correlation(capability, trust)
        checks[f"ranking_rho_recomputed_{bit}bit"] = close(
            diagnostic["capability_trust_spearman"], observed
        )
        checks[f"ranking_ci_finite_ordered_{bit}bit"] = (
            np.isfinite(diagnostic["capability_trust_spearman_bootstrap_95_ci"]).all()
            and diagnostic["capability_trust_spearman_bootstrap_95_ci"][0]
            <= diagnostic["capability_trust_spearman_bootstrap_95_ci"][1]
        )
        for k in (3, 5, 8):
            checks[f"top{k}_overlap_recomputed_{bit}bit"] = close(
                diagnostic["top_k_overlap"][str(k)], top_overlap(capability, trust, k)
            )
    checks["rho_2bit_above_negative_threshold"] = (
        analysis["ranking_diagnostics"]["2"]["capability_trust_spearman"] >= 0.7
    )
    checks["rho_4bit_above_negative_threshold"] = (
        analysis["ranking_diagnostics"]["4"]["capability_trust_spearman"] >= 0.7
    )
    checks["zero_stable_trust_only_layers"] = not any(
        row["stable_trust_only"] for row in analysis["layer_rows"]
    )
    checks["2bit_top_quartiles_identical"] = all(
        row["taxonomy"] not in {"capability_only", "trust_only"}
        for row in analysis["layer_rows"] if row["bits"] == 2
    )
    checks["4bit_top_quartiles_identical"] = all(
        row["taxonomy"] not in {"capability_only", "trust_only"}
        for row in analysis["layer_rows"] if row["bits"] == 4
    )
    gate_expected = any(
        analysis["ranking_diagnostics"][str(bit)]["capability_trust_spearman"] < 0.7
        and analysis["ranking_diagnostics"][str(bit)]["capability_trust_spearman_bootstrap_95_ci"][1] < 0.9
        for bit in (2, 4)
    )
    checks["gate_logic_recomputed"] = analysis["gate"]["passed"] == gate_expected
    dataset_names = set(report["snapshot"]["datasets"])
    checks["report_datasets_complete"] = dataset_names == {
        "scatter_2", "scatter_4", "depth_4", "rankings", "taxonomy"
    }
    source_ids = {source["id"] for source in report["manifest"]["sources"]}
    checks["all_charts_have_query_sources"] = all(
        chart["sourceId"] in source_ids
        and next(source for source in report["manifest"]["sources"] if source["id"] == chart["sourceId"]).get("query", {}).get("sql")
        for chart in report["manifest"]["charts"]
    )
    checks["all_tables_have_query_sources"] = all(
        table["sourceId"] in source_ids
        and next(source for source in report["manifest"]["sources"] if source["id"] == table["sourceId"]).get("query", {}).get("sql")
        for table in report["manifest"]["tables"]
    )
    checks = {key: bool(value) for key, value in checks.items()}
    refs = {}
    for model in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"):
        ref = Path(".cache/huggingface/hub") / f"models--Qwen--{model}" / "refs/main"
        refs[model] = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    key_files = [
        SOURCE / "calibration_pool.jsonl",
        SOURCE / "evaluation.jsonl",
        SOURCE / "calibration.npz",
        ROOT / "layer_analysis.json",
        ROOT / "summary.json",
        ROOT / "report/artifact.json",
        ROOT / "report/report.html",
    ]
    output = {
        "assessment": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model_revisions": refs,
        "report_package_verification": "structural_only",
        "sha256": {str(path): digest(path) for path in key_files},
        "material_caveats": [
            "Only ARC-Challenge and one 0.5B student model were analyzed.",
            "One-block probes cannot identify attention-versus-MLP or higher-order whole-model interactions.",
            "Fake quantization does not establish packed storage, memory, latency, or throughput.",
            "The held-out policy benchmark and MMLU-Pro were conditionally skipped after the ranking gate failed.",
            "Portable report packaging passed structurally, but browser QA did not run because Chromium is unavailable.",
        ],
    }
    (ROOT / "validation.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "assessment": output["assessment"],
                "checks": len(checks),
                "passed": sum(checks.values()),
            },
            indent=2,
        )
    )
    return 0 if output["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
