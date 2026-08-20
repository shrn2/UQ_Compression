"""Build one comparative UPGD report across the 1.5B and 3B runs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs")
BASE = ROOT / "iter11_upgd" / "report" / "artifact.json"
DEST = ROOT / "iter11_12_upgd" / "report"


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def query_source(source_id: str, label: str, rows: list[dict], columns: list[str], generated: str,
                 filters: list[str], definitions: list[str]) -> dict:
    values = ",\n    ".join(
        "(" + ", ".join(literal(row.get(column)) for column in columns) + ")" for row in rows
    )
    return {"id": source_id, "label": label, "query": {
        "engine": "SQLite", "language": "sql",
        "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
        "description": f"Materialize reviewed {label.lower()} rows.",
        "executed_at": generated, "filters": filters, "metric_definitions": definitions,
    }}


def metric_rows(analysis: dict, model: str, dataset: str, examples: int) -> list[dict]:
    rows = {row["condition"]: row for row in analysis["metric_rows"]}
    result = []
    for condition, label in (("baseline_aggressive", "One-shot"), ("upgd_aggressive", "UPGD")):
        row = rows[condition]
        result.append({
            "model": model, "dataset": dataset, "condition": label, "examples": examples,
            "accuracy": row["accuracy"], "kl": row["mean_kl_from_dense"],
            "agreement": row["prediction_agreement_with_dense"],
            "entropy_rho": row["entropy_spearman_with_dense"], "aurc": row["aurc"], "ece": row["ece"],
        })
    return result


def accuracy_rows(analysis: dict, model: str, dataset: str, examples: int) -> list[dict]:
    rows = {row["condition"]: row for row in analysis["metric_rows"]}
    return [
        {"dataset": dataset, "condition": f"{model} Dense", "accuracy": rows["dense"]["accuracy"], "examples": examples},
        {"dataset": dataset, "condition": f"{model} One-shot", "accuracy": rows["baseline_aggressive"]["accuracy"], "examples": examples},
        {"dataset": dataset, "condition": f"{model} UPGD", "accuracy": rows["upgd_aggressive"]["accuracy"], "examples": examples},
    ]


def paired_row(analysis: dict, model: str, dataset: str, examples: int) -> dict:
    paired = analysis["paired_comparison"]
    return {
        "model": model, "dataset": dataset, "examples": examples,
        "difference": paired["difference"], "ci_low": paired["bootstrap_95_ci"][0],
        "ci_high": paired["bootstrap_95_ci"][1], "gate": "PASS" if analysis["gate"]["passed"] else "FAIL",
    }


def stratum_rows(analysis: dict, model: str, dataset: str) -> list[dict]:
    rows = {row["condition"] + str(row["stratum"]): row for row in analysis["stratum_rows"]}
    result = []
    for stratum, label in enumerate(("Low", "Middle", "High")):
        upgd = rows["upgd_aggressive" + str(stratum)]
        baseline = rows["baseline_aggressive" + str(stratum)]
        result.append({
            "slice": f"{model} / {dataset}", "model": model, "dataset": dataset,
            "stratum": label, "accuracy_difference": upgd["accuracy"] - baseline["accuracy"],
            "examples": upgd["examples"],
        })
    return result


def transfer_rows(manifest: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    names = {"commonsenseqa": "CommonsenseQA", "hellaswag": "HellaSwag", "openbookqa": "OpenBookQA", "winogrande": "Winogrande"}
    models = {"1p5b": "1.5B BF16", "3b_nf4": "3B NF4"}
    accuracy, paired, metrics, strata = [], [], [], []
    for item in manifest:
        analysis = json.loads(Path(item["analysis"]).read_text(encoding="utf-8"))
        model = models[item["model_key"]]
        dataset = names[item["dataset"]]
        accuracy.extend(accuracy_rows(analysis, model, dataset, item["examples"]))
        paired.append(paired_row(analysis, model, dataset, item["examples"]))
        metrics.extend(metric_rows(analysis, model, dataset, item["examples"]))
        strata.extend(stratum_rows(analysis, model, dataset))
    return accuracy, paired, metrics, strata


def ablation_rows(manifest: dict) -> list[dict]:
    models = {"1p5b": "1.5B BF16", "3b_nf4": "3B NF4"}
    labels = {
        "random_seed1": "Random 1",
        "random_seed2": "Random 2",
        "random_seed3": "Random 3",
        "one_shot": "One-shot",
        "kl_only": "KL-only",
        "kl_entropy_margin": "KL + entropy + margin",
        "full_upgd": "Full UPGD",
    }
    order = {name: index for index, name in enumerate(labels)}
    rows = []
    for item in manifest["rows"]:
        rows.append({
            "model": models[item["model_key"]],
            "dataset": "HellaSwag",
            "method": labels[item["method"]],
            "method_key": item["method"],
            "method_order": order[item["method"]],
            "examples": item["examples"],
            "dense_accuracy": item["dense_accuracy"],
            "baseline_accuracy": item["baseline_accuracy"],
            "accuracy": item["accuracy"],
            "difference": item["difference"],
            "ci_low": item["ci_low"],
            "ci_high": item["ci_high"],
            "gate": "PASS" if item["gate"] else "FAIL",
            "kl": item["kl"],
            "agreement": item["agreement"],
            "entropy_rho": item["entropy_rho"],
            "aurc": item["aurc"],
            "ece": item["ece"],
            "entropy_mae": item["entropy_mae"],
            "margin_mae": item["margin_mae"],
        })
    return sorted(rows, key=lambda row: (row["model"], row["method_order"]))


def ablation_random_summary(rows: list[dict]) -> list[dict]:
    result = []
    for model in ("1.5B BF16", "3B NF4"):
        selected = [row for row in rows if row["model"] == model and row["method_key"].startswith("random_")]
        differences = [row["difference"] for row in selected]
        accuracies = [row["accuracy"] for row in selected]
        import statistics
        result.append({
            "model": model, "dataset": "HellaSwag", "seeds": len(selected),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "sd_accuracy": statistics.stdev(accuracies),
            "mean_difference": sum(differences) / len(differences),
            "sd_difference": statistics.stdev(differences),
        })
    return result


def previous_ablation_rows(manifest: dict) -> list[dict]:
    models = {"1p5b": "1.5B BF16", "3b_nf4": "3B NF4"}
    labels = {
        "random_seed1": "Random 1", "random_seed2": "Random 2", "random_seed3": "Random 3",
        "one_shot": "One-shot", "kl_only": "KL-only", "kl_entropy_margin": "KL + entropy + margin", "full_upgd": "Full UPGD",
    }
    order = {name: index for index, name in enumerate(labels)}
    rows = []
    for item in manifest["rows"]:
        rows.append({
            "model": models[item["model_key"]], "dataset": item["dataset"], "dataset_key": item["dataset_key"],
            "method": labels[item["method"]], "method_key": item["method"], "method_order": order[item["method"]],
            "examples": item["examples"], "dense_accuracy": item["dense_accuracy"], "baseline_accuracy": item["baseline_accuracy"],
            "accuracy": item["accuracy"], "difference": item["difference"], "ci_low": item["ci_low"], "ci_high": item["ci_high"],
            "gate": "PASS" if item["gate"] else "FAIL", "kl": item["kl"], "agreement": item["agreement"],
            "entropy_rho": item["entropy_rho"], "aurc": item["aurc"], "ece": item["ece"],
            "entropy_mae": item["entropy_mae"], "margin_mae": item["margin_mae"],
        })
    return sorted(rows, key=lambda row: (row["model"], row["dataset"], row["method_order"]))


def previous_random_summary(manifest: dict) -> list[dict]:
    models = {"1p5b": "1.5B BF16", "3b_nf4": "3B NF4"}
    return [
        {"model": models[item["model_key"]], "dataset": item["dataset"], "seeds": item["seeds"], "mean_accuracy": item["mean_accuracy"], "sd_accuracy": item["sd_accuracy"], "mean_difference": item["mean_difference"], "sd_difference": item["sd_difference"]}
        for item in manifest["random_summary"]
    ]


def uncertainty_study_rows(manifest: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Normalize the Iteration 16 lambda sweep for report datasets."""
    model_labels = {"1p5b": "1.5B BF16", "3b_nf4": "3B NF4"}
    dataset_labels = {"hellaswag": "HellaSwag", "arc_challenge": "ARC-Challenge", "mmlu_aux": "MMLU auxiliary"}
    summary = []
    for row in manifest["summary_rows"]:
        summary.append({**row, "model": model_labels[row["model_key"]], "dataset": dataset_labels[row["dataset_key"]]})
    paired = []
    for row in manifest["paired_lambda_vs_zero"]:
        paired.append({**row, "model": model_labels[row["model_key"]], "dataset": dataset_labels[row["dataset_key"]]})
    pareto = []
    for row in manifest["pareto_rows"]:
        pareto.append({**row, "model": model_labels[row["model_key"]], "dataset": dataset_labels[row["dataset_key"]]})
    return summary, paired, pareto


def main() -> int:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    iter11 = ROOT / "iter11_upgd"
    iter12 = ROOT / "iter12_q4_3b"
    dev11 = json.loads((iter11 / "development_analysis.json").read_text(encoding="utf-8"))
    mmlu11 = json.loads((iter11 / "mmlu_pro_analysis.json").read_text(encoding="utf-8"))
    arc12 = json.loads((iter12 / "arc_analysis.json").read_text(encoding="utf-8"))
    mmlu12 = json.loads((iter12 / "mmlu_pro_analysis.json").read_text(encoding="utf-8"))
    train11 = json.loads((iter11 / "training.json").read_text(encoding="utf-8"))
    train12 = json.loads((iter12 / "training.json").read_text(encoding="utf-8"))
    val11 = json.loads((iter11 / "validation.json").read_text(encoding="utf-8"))
    val12 = json.loads((iter12 / "validation.json").read_text(encoding="utf-8"))
    iter13 = ROOT / "iter13_transfer"
    transfer_manifest = json.loads((iter13 / "analysis_manifest.json").read_text(encoding="utf-8"))
    transfer_accuracy, transfer_paired, transfer_metrics, transfer_strata = transfer_rows(transfer_manifest)
    iter14 = ROOT / "iter14_ablation_hellaswag"
    ablation_manifest = json.loads((iter14 / "analysis_manifest.json").read_text(encoding="utf-8"))
    ablation = ablation_rows(ablation_manifest)
    ablation_random = ablation_random_summary(ablation)
    iter15 = ROOT / "iter15_ablation_previous"
    previous_manifest = json.loads((iter15 / "analysis_manifest.json").read_text(encoding="utf-8"))
    previous_ablation = previous_ablation_rows(previous_manifest)
    previous_random = previous_random_summary(previous_manifest)
    iter16 = ROOT / "iter16_uncertainty_reg"
    uncertainty_manifest = json.loads((iter16 / "analysis_manifest.json").read_text(encoding="utf-8"))
    uncertainty_summary, uncertainty_paired, uncertainty_pareto = uncertainty_study_rows(uncertainty_manifest)
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)

    copied = {
        "iter11_development_analysis.json": iter11 / "development_analysis.json",
        "iter11_mmlu_pro_analysis.json": iter11 / "mmlu_pro_analysis.json",
        "iter11_training.json": iter11 / "training.json",
        "iter11_validation.json": iter11 / "validation.json",
        "iter12_arc_analysis.json": iter12 / "arc_analysis.json",
        "iter12_mmlu_pro_analysis.json": iter12 / "mmlu_pro_analysis.json",
        "iter12_training.json": iter12 / "training.json",
        "iter12_validation.json": iter12 / "validation.json",
        "iter12_scoring_manifest.json": iter12 / "scoring_manifest.json",
        "iter13_analysis_manifest.json": iter13 / "analysis_manifest.json",
        "iter13_validation.json": iter13 / "validation.json",
        "iter14_ablation_manifest.json": iter14 / "ablation_manifest.json",
        "iter14_ablation_analysis_manifest.json": iter14 / "analysis_manifest.json",
        "iter14_ablation_validation.json": iter14 / "validation.json",
        "iter15_ablation_manifest.json": iter15 / "ablation_manifest.json",
        "iter15_ablation_analysis_manifest.json": iter15 / "analysis_manifest.json",
        "iter15_ablation_validation.json": iter15 / "validation.json",
        "iter16_study_manifest.json": iter16 / "study_manifest.json",
        "iter16_analysis_manifest.json": iter16 / "analysis_manifest.json",
        "iter16_calibration_manifest.json": iter16 / "calibration_manifest.json",
    }
    for dataset_path in (iter13 / "datasets").glob("*.jsonl"):
        copied[f"iter13_{dataset_path.name}"] = dataset_path
    for analysis_path in iter13.glob("*_analysis.json"):
        copied[f"iter13_{analysis_path.name}"] = analysis_path
    for analysis_path in iter14.glob("*_analysis.json"):
        copied[f"iter14_{analysis_path.name}"] = analysis_path
    for model_key in ("1p5b", "3b_nf4"):
        for training_path in (iter14 / model_key).glob("*_training.json"):
            copied[f"iter14_{model_key}_{training_path.name}"] = training_path
        for log_path in (iter14 / model_key).glob("*_log.jsonl"):
            copied[f"iter14_{model_key}_{log_path.name}"] = log_path
        for prediction_path in (iter14 / model_key / "scores").glob("*_predictions.jsonl"):
            copied[f"iter14_{model_key}_{prediction_path.name}"] = prediction_path
        copied[f"iter14_{model_key}_experiment_config.json"] = iter14 / model_key / "experiment_config.json"
    for dataset_path in (iter15 / "datasets").glob("*.jsonl"):
        copied[f"iter15_{dataset_path.name}"] = dataset_path
    for analysis_path in iter15.glob("*_analysis.json"):
        copied[f"iter15_{analysis_path.name}"] = analysis_path
    for model_key in ("1p5b", "3b_nf4"):
        copied[f"iter15_{model_key}_experiment_config.json"] = iter15 / model_key / "experiment_config.json"
        for analysis_path in (iter15 / model_key).glob("*_analysis.json"):
            copied[f"iter15_{model_key}_{analysis_path.name}"] = analysis_path
        copied[f"iter16_{model_key}_calibration.npz"] = iter16 / f"calibration_{model_key}.npz"
    for name, source in copied.items():
        shutil.copyfile(source, DEST / name)

    accuracy = (
        accuracy_rows(dev11, "1.5B BF16", "MMLU auxiliary", 600)
        + accuracy_rows(mmlu11, "1.5B BF16", "MMLU-Pro", 1000)
        + accuracy_rows(arc12, "3B NF4", "ARC-Challenge", 299)
        + accuracy_rows(mmlu12, "3B NF4", "MMLU-Pro", 1000)
        + transfer_accuracy
    )
    paired = [
        paired_row(dev11, "1.5B BF16", "MMLU auxiliary", 600),
        paired_row(mmlu11, "1.5B BF16", "MMLU-Pro", 1000),
        paired_row(arc12, "3B NF4", "ARC-Challenge", 299),
        paired_row(mmlu12, "3B NF4", "MMLU-Pro", 1000),
        *transfer_paired,
    ]
    metrics = (
        metric_rows(dev11, "1.5B BF16", "MMLU auxiliary", 600)
        + metric_rows(mmlu11, "1.5B BF16", "MMLU-Pro", 1000)
        + metric_rows(arc12, "3B NF4", "ARC-Challenge", 299)
        + metric_rows(mmlu12, "3B NF4", "MMLU-Pro", 1000)
        + transfer_metrics
    )
    strata = (
        stratum_rows(dev11, "1.5B BF16", "MMLU auxiliary")
        + stratum_rows(mmlu11, "1.5B BF16", "MMLU-Pro")
        + stratum_rows(arc12, "3B NF4", "ARC-Challenge")
        + stratum_rows(mmlu12, "3B NF4", "MMLU-Pro")
        + transfer_strata
    )

    sources = [
        {"id": "iter11-dev-json", "label": "1.5B BF16 MMLU auxiliary analysis", "path": "iter11_development_analysis.json"},
        {"id": "iter11-mmlu-json", "label": "1.5B BF16 MMLU-Pro analysis", "path": "iter11_mmlu_pro_analysis.json"},
        {"id": "iter11-training-json", "label": "1.5B BF16 UPGD training record", "path": "iter11_training.json"},
        {"id": "iter11-validation-json", "label": "1.5B BF16 independent validation", "path": "iter11_validation.json"},
        {"id": "iter12-arc-json", "label": "3B NF4 ARC analysis", "path": "iter12_arc_analysis.json"},
        {"id": "iter12-mmlu-json", "label": "3B NF4 MMLU-Pro analysis", "path": "iter12_mmlu_pro_analysis.json"},
        {"id": "iter12-training-json", "label": "3B NF4 UPGD training record", "path": "iter12_training.json"},
        {"id": "iter12-validation-json", "label": "3B NF4 independent validation", "path": "iter12_validation.json"},
        {"id": "iter12-manifest-json", "label": "3B NF4 scoring manifest", "path": "iter12_scoring_manifest.json"},
        {"id": "iter13-validation-json", "label": "Independent Iteration 13 transfer validation", "path": "iter13_validation.json"},
        query_source("accuracy-query", "accuracy by model, dataset, and condition", accuracy,
                     ["dataset", "condition", "accuracy", "examples"], generated,
                     ["same examples within each model/dataset slice", "50% FFN pruning for compressed conditions"],
                     ["accuracy is exact-choice accuracy"]),
        query_source("paired-query", "paired UPGD-minus-baseline accuracy intervals", paired,
                     ["model", "dataset", "examples", "difference", "ci_low", "ci_high", "gate"], generated,
                     ["5,000 paired bootstrap replicates", "UPGD and one-shot baseline scored on identical examples"],
                     ["difference is UPGD accuracy minus one-shot accuracy"]),
        query_source("metric-query", "compressed-model fidelity metrics", metrics,
                     ["model", "dataset", "condition", "examples", "accuracy", "kl", "agreement", "entropy_rho", "aurc", "ece"], generated,
                     ["same dense reference within each model/dataset slice", "50% FFN masks"],
                     ["KL, agreement, entropy rho, AURC, and ECE compare each compressed model with its dense reference"]),
        query_source("strata-query", "UPGD-minus-baseline accuracy by uncertainty stratum", strata,
                     ["slice", "model", "dataset", "stratum", "accuracy_difference", "examples"], generated,
                     ["three equal-frequency dense-entropy strata within each slice"],
                     ["difference is UPGD accuracy minus one-shot accuracy"]),
        query_source("design-query", "model and protocol comparison", [
            {"model": "1.5B BF16", "layers": 28, "ffn_channels": 8960, "quantization": "none", "mask_mode": "physical", "steps": 600, "mask_budget": 0.5},
            {"model": "3B NF4", "layers": 36, "ffn_channels": 11008, "quantization": "4bit NF4", "mask_mode": "logical hook", "steps": 300, "mask_budget": 0.5},
        ], ["model", "layers", "ffn_channels", "quantization", "mask_mode", "steps", "mask_budget"], generated,
                     ["same UPGD objective family", "resource-adapted 3B run"],
                     ["mask budget is retained FFN-channel fraction", "logical hook does not establish sparse-kernel speed"]),
        query_source("method-query", "UPGD methodological specification", [
            {"component": "Joint gate parameterization", "specification": "one learnable gate logit per FFN channel across all layers; model weights frozen", "value": "global channel decision"},
            {"component": "Exact-budget stochastic mask", "specification": "annealed logistic noise followed by exact top-k selection at 50% retention", "value": "temperature 0.5 to 0.1; noise 0.5 to 0.05"},
            {"component": "Dense-behavior distillation", "specification": "KL to dense answer distribution plus squared predictive-entropy and logit-margin matching", "value": "entropy weight 1.0; margin weight 0.5"},
            {"component": "Uncertainty robustness", "specification": "one example from each dense-entropy stratum and a smooth worst-stratum penalty", "value": "robust weight 0.5; robust temperature 0.1"},
            {"component": "Deployment mask", "specification": "deterministic retained-channel indices after optimization", "value": "physical export for 1.5B; logical hook for 3B NF4"},
        ], ["component", "specification", "value"], generated,
                     ["same objective family in both model runs", "exact 50% FFN-channel budget"],
                     ["UPGD treats uncertainty as dense behavior to preserve, not as a monotone local importance score"]),
        query_source("validation-query", "independent validation and test results", [
            {"model": "1.5B BF16", "validation_passed": 50, "validation_total": 50, "python_tests": 49},
            {"model": "3B NF4", "validation_passed": 37, "validation_total": 37, "python_tests": 49},
            {"model": "Iter13 transfer", "validation_passed": 80, "validation_total": 80, "python_tests": 49},
            {"model": "Iter14 ablation", "validation_passed": 157, "validation_total": 157, "python_tests": 49},
            {"model": "Iter15 prior-cohort ablation", "validation_passed": 169, "validation_total": 169, "python_tests": 49},
            {"model": "Iter16 lambda-UQ study", "validation_passed": 197, "validation_total": 197, "python_tests": 49},
        ], ["model", "validation_passed", "validation_total", "python_tests"], generated,
                     ["independent artifact validators", "repository pytest suite"],
                     ["validation counts are passed checks over total checks"]),
        query_source("transfer-query", "four additional transfer datasets", [
            {"model": item["model_key"], "dataset": item["dataset"], "examples": item["examples"], "difference": item["difference"], "ci_low": item["ci_low"], "ci_high": item["ci_high"], "gate": "PASS" if item["gate"] else "FAIL"}
            for item in transfer_manifest
        ], ["model", "dataset", "examples", "difference", "ci_low", "ci_high", "gate"], generated,
                     ["four fixed 500-example cohorts", "same logical-channel evaluation path for both models", "5,000 paired bootstrap replicates"],
                     ["difference is UPGD accuracy minus one-shot accuracy"]),
        {"id": "iter14-ablation-json", "label": "Iteration 14 HellaSwag ablation artifacts", "path": "iter14_ablation_analysis_manifest.json"},
        {"id": "iter14-ablation-validation-json", "label": "Iteration 14 ablation validation", "path": "iter14_ablation_validation.json"},
        query_source("ablation-query", "UPGD objective ablation on HellaSwag", ablation,
                     ["model", "dataset", "method", "method_key", "method_order", "examples", "dense_accuracy", "baseline_accuracy", "accuracy", "difference", "ci_low", "ci_high", "gate", "kl", "agreement", "entropy_rho", "aurc", "ece", "entropy_mae", "margin_mae"], generated,
                     ["same 500 HellaSwag examples within each model", "exact 50% FFN channel budget", "5,000 paired bootstrap replicates", "three independent random-mask seeds"],
                     ["difference is method accuracy minus one-shot accuracy", "KL, agreement, entropy rho, AURC, and ECE compare with dense", "entropy and margin MAE compare predictive geometry with dense"]),
        query_source("ablation-random-query", "random-mask ablation summary", ablation_random,
                     ["model", "dataset", "seeds", "mean_accuracy", "sd_accuracy", "mean_difference", "sd_difference"], generated,
                     ["three independent random masks per model", "same 500 HellaSwag examples"],
                     ["mean and sample standard deviation are across random-mask seeds"]),
        {"id": "iter15-ablation-json", "label": "Iteration 15 previous-cohort ablation artifacts", "path": "iter15_ablation_analysis_manifest.json"},
        {"id": "iter15-ablation-validation-json", "label": "Iteration 15 previous-cohort ablation validation", "path": "iter15_ablation_validation.json"},
        query_source("previous-ablation-query", "UPGD ablation on prior MMLU auxiliary and ARC-Challenge cohorts", previous_ablation,
                     ["model", "dataset", "dataset_key", "method", "method_key", "method_order", "examples", "dense_accuracy", "baseline_accuracy", "accuracy", "difference", "ci_low", "ci_high", "gate", "kl", "agreement", "entropy_rho", "aurc", "ece", "entropy_mae", "margin_mae"], generated,
                     ["same prior evaluation populations: 600 MMLU auxiliary and 299 ARC-Challenge examples", "exact 50% FFN channel budget", "5,000 paired bootstrap replicates", "same logical-channel evaluation path for both models"],
                     ["difference is method accuracy minus one-shot accuracy", "KL, agreement, entropy rho, AURC, and ECE compare with dense", "entropy and margin MAE compare predictive geometry with dense"]),
        query_source("previous-random-query", "random-mask summary on prior cohorts", previous_random,
                     ["model", "dataset", "seeds", "mean_accuracy", "sd_accuracy", "mean_difference", "sd_difference"], generated,
                     ["three independent random masks per model and cohort"],
                     ["mean and sample standard deviation are across random-mask seeds"]),
    ]

    sources.extend([
        {"id": "iter16-study-json", "label": "Iteration 16 uncertainty-regularization study artifacts", "path": "iter16_analysis_manifest.json"},
        {"id": "iter16-calibration-json", "label": "Iteration 16 disjoint multi-task calibration manifest", "path": "iter16_calibration_manifest.json"},
        query_source("lambda-summary-query", "lambda-UQ seed summaries", uncertainty_summary,
                     ["model", "dataset", "lambda_uq", "seeds", "accuracy_mean", "accuracy_sd", "delta_vs_dense_mean", "entropy_mae_mean", "entropy_mae_sd", "kl_mean", "aurc_mean", "ece_mean"], generated,
                     ["two models", "three final datasets", "three learned-mask seeds per lambda", "matched 300-step budget"],
                     ["accuracy and diagnostics are means across three learned-mask seeds", "lambda_uq is the uncertainty-regularization strength"]),
        query_source("lambda-paired-query", "paired lambda-UQ versus KL-only (lambda=0)", uncertainty_paired,
                     ["model", "dataset", "lambda_uq", "seed", "metric", "difference", "ci_low", "ci_high"], generated,
                     ["same examples and seed within each lambda comparison", "5,000 paired bootstrap replicates"],
                     ["difference is lambda-UQ metric minus lambda=0 metric"]),
        query_source("lambda-pareto-query", "lambda-UQ accuracy-loss versus entropy-error frontier", uncertainty_pareto,
                     ["model", "dataset", "lambda_uq", "seed", "accuracy_loss", "entropy_mae"], generated,
                     ["same final evaluation examples", "accuracy loss is dense accuracy minus masked accuracy"],
                     ["lower is better on both axes"]),
    ])

    charts = [
        {"id": "accuracy", "title": "Accuracy across model and dataset slices", "subtitle": "Dense reference, one-shot baseline, and UPGD at a 50% FFN channel budget.",
         "showDescription": True, "intent": "comparison", "question": "How does the same UPGD method behave across model scale and datasets?",
         "rationale": "Grouped bars make the shared MMLU-Pro comparison and the non-overlapping task slices explicit.",
         "comparisonContext": {"grain": "model, dataset, and condition", "unit": "accuracy", "denominator": "299, 600, or 1,000 examples"},
         "type": "bar", "dataset": "accuracy", "sourceId": "accuracy-query",
         "encodings": {"x": {"field": "dataset", "type": "nominal", "label": "Dataset"}, "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"}, "color": {"field": "condition", "type": "nominal", "label": "Model / condition"},
                       "tooltip": [{"field": "condition", "type": "text", "label": "Model / condition"}, {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"}, {"field": "examples", "type": "quantitative", "format": "number", "label": "Examples"}]},
         "palette": {"kind": "categorical", "name": "model-condition"}, "legend": {"position": "bottom", "title": "Model / condition"}, "layout": "full", "maxRows": 40, "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "uplift", "title": "UPGD accuracy difference versus one-shot pruning", "subtitle": "Paired UPGD-minus-baseline effects with 95% bootstrap intervals in the adjacent table.",
         "showDescription": True, "intent": "comparison", "question": "Is the UPGD gain consistent across model scale and datasets?",
         "rationale": "Signed bars use a common zero reference to expose positive, unresolved, and negative transfers.",
         "comparisonContext": {"grain": "model and dataset", "unit": "accuracy difference", "denominator": "same examples within each slice"},
         "type": "bar", "dataset": "paired", "sourceId": "paired-query",
         "encodings": {"x": {"field": "dataset", "type": "nominal", "label": "Dataset"}, "y": {"field": "difference", "type": "quantitative", "format": "percent", "label": "UPGD minus baseline"}, "color": {"field": "model", "type": "nominal", "label": "Model"},
                       "tooltip": [{"field": "model", "type": "text", "label": "Model"}, {"field": "dataset", "type": "text", "label": "Dataset"}, {"field": "difference", "type": "quantitative", "format": "percent", "label": "Difference"}, {"field": "gate", "type": "text", "label": "Gate"}]},
         "palette": {"kind": "categorical", "name": "model"}, "legend": {"position": "bottom", "title": "Model"}, "layout": "full", "maxRows": 20, "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "strata", "title": "UPGD accuracy difference by dense-entropy stratum", "subtitle": "Signed effects within each model/dataset slice; each slice is split into three equal-frequency strata.",
         "showDescription": True, "intent": "comparison", "question": "Does uncertainty-stratum robustness transfer across models and datasets?",
         "rationale": "A common zero reference reveals both within-slice losses and cross-slice reversals.",
         "comparisonContext": {"grain": "model/dataset slice and stratum", "unit": "accuracy difference", "denominator": "about one third of each slice"},
         "type": "bar", "dataset": "strata", "sourceId": "strata-query",
         "encodings": {"x": {"field": "stratum", "type": "nominal", "label": "Dense-entropy stratum"}, "y": {"field": "accuracy_difference", "type": "quantitative", "format": "percent", "label": "UPGD minus baseline"}, "color": {"field": "slice", "type": "nominal", "label": "Model / dataset"},
                       "tooltip": [{"field": "slice", "type": "text", "label": "Model / dataset"}, {"field": "stratum", "type": "text", "label": "Stratum"}, {"field": "accuracy_difference", "type": "quantitative", "format": "percent", "label": "Difference"}, {"field": "examples", "type": "quantitative", "format": "number", "label": "Examples"}]},
         "palette": {"kind": "categorical", "name": "slice"}, "legend": {"position": "bottom", "title": "Model / dataset"}, "layout": "full", "maxRows": 40, "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "ablation", "title": "UPGD objective ablation on HellaSwag", "subtitle": "Matched 50% masks: random controls, one-shot baseline, partial objectives, and full UPGD.",
         "showDescription": True, "intent": "comparison", "question": "Which objective components improve transfer beyond one-shot pruning?",
         "rationale": "A signed method-by-model chart separates random-mask variance from the incremental contribution of learned objectives.",
         "comparisonContext": {"grain": "model and ablation method", "unit": "accuracy difference versus one-shot", "denominator": "500 HellaSwag examples"},
         "type": "bar", "dataset": "ablation", "sourceId": "ablation-query",
         "encodings": {"x": {"field": "method", "type": "nominal", "label": "Ablation method", "sort": ["Random 1", "Random 2", "Random 3", "One-shot", "KL-only", "KL + entropy + margin", "Full UPGD"]}, "y": {"field": "difference", "type": "quantitative", "format": "percent", "label": "Accuracy versus one-shot"}, "color": {"field": "model", "type": "nominal", "label": "Model"},
                       "tooltip": [{"field": "model", "type": "text", "label": "Model"}, {"field": "method", "type": "text", "label": "Method"}, {"field": "difference", "type": "quantitative", "format": "percent", "label": "Versus one-shot"}, {"field": "ci_low", "type": "quantitative", "format": "percent", "label": "CI low"}, {"field": "ci_high", "type": "quantitative", "format": "percent", "label": "CI high"}]},
         "palette": {"kind": "categorical", "name": "model"}, "legend": {"position": "bottom", "title": "Model"}, "layout": "full", "maxRows": 20, "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "previous-ablation", "title": "UPGD objective ablation on prior cohorts", "subtitle": "MMLU auxiliary and ARC-Challenge, with the same 50% masks and paired comparison basis.",
         "showDescription": True, "intent": "comparison", "question": "Does the ablation pattern repeat on earlier evaluation cohorts?",
         "rationale": "Grouped signed bars show whether learned gates and uncertainty terms transfer beyond HellaSwag.",
         "comparisonContext": {"grain": "model, dataset, and ablation method", "unit": "accuracy difference versus one-shot", "denominator": "600 MMLU auxiliary or 299 ARC-Challenge examples"},
         "type": "bar", "dataset": "previous_ablation", "sourceId": "previous-ablation-query",
         "encodings": {"x": {"field": "method", "type": "nominal", "label": "Ablation method", "sort": ["Random 1", "Random 2", "Random 3", "One-shot", "KL-only", "KL + entropy + margin", "Full UPGD"]}, "y": {"field": "difference", "type": "quantitative", "format": "percent", "label": "Accuracy versus one-shot"}, "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                       "tooltip": [{"field": "model", "type": "text", "label": "Model"}, {"field": "dataset", "type": "text", "label": "Dataset"}, {"field": "method", "type": "text", "label": "Method"}, {"field": "difference", "type": "quantitative", "format": "percent", "label": "Versus one-shot"}, {"field": "ci_low", "type": "quantitative", "format": "percent", "label": "CI low"}, {"field": "ci_high", "type": "quantitative", "format": "percent", "label": "CI high"}]},
         "palette": {"kind": "categorical", "name": "dataset"}, "legend": {"position": "bottom", "title": "Dataset"}, "layout": "full", "maxRows": 30, "surface": {"surface": "explorer", "viewMode": "both"}},
    ]
    charts.extend([
        {"id": "lambda-accuracy", "title": "Accuracy across uncertainty-regularization strengths", "subtitle": "Means across three learned-mask seeds; lambda=0 is KL-only.", "showDescription": True, "intent": "comparison", "question": "Does increasing uncertainty regularization improve accuracy consistently?", "rationale": "Lines expose whether the lambda response is monotone or task-sensitive.", "comparisonContext": {"grain": "model, dataset, lambda", "unit": "accuracy", "denominator": "299, 500, or 600 examples"}, "type": "line", "dataset": "lambda_summary", "sourceId": "lambda-summary-query", "encodings": {"x": {"field": "lambda_uq", "type": "quantitative", "label": "lambda UQ"}, "y": {"field": "accuracy_mean", "type": "quantitative", "format": "percent", "label": "Mean accuracy"}, "color": {"field": "model", "type": "nominal", "label": "Model"}, "detail": {"field": "dataset", "type": "nominal"}, "tooltip": [{"field": "model", "type": "text", "label": "Model"}, {"field": "dataset", "type": "text", "label": "Dataset"}, {"field": "lambda_uq", "type": "number", "label": "lambda UQ"}, {"field": "accuracy_mean", "type": "quantitative", "format": "percent", "label": "Mean accuracy"}, {"field": "accuracy_sd", "type": "quantitative", "format": "percent", "label": "Seed SD"}]}, "palette": {"kind": "categorical", "name": "model"}, "legend": {"position": "bottom", "title": "Model"}, "layout": "full", "maxRows": 30, "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "lambda-entropy", "title": "Uncertainty fidelity across lambda", "subtitle": "Mean entropy MAE against the dense reference; lower is better.", "showDescription": True, "intent": "comparison", "question": "Does lambda-UQ reduce uncertainty drift?", "rationale": "The uncertainty metric directly tests the methodological contribution.", "comparisonContext": {"grain": "model, dataset, lambda", "unit": "entropy MAE", "denominator": "same final evaluation examples"}, "type": "line", "dataset": "lambda_summary", "sourceId": "lambda-summary-query", "encodings": {"x": {"field": "lambda_uq", "type": "quantitative", "label": "lambda UQ"}, "y": {"field": "entropy_mae_mean", "type": "quantitative", "label": "Entropy MAE"}, "color": {"field": "dataset", "type": "nominal", "label": "Dataset"}, "detail": {"field": "model", "type": "nominal"}, "tooltip": [{"field": "model", "type": "text", "label": "Model"}, {"field": "dataset", "type": "text", "label": "Dataset"}, {"field": "lambda_uq", "type": "number", "label": "lambda UQ"}, {"field": "entropy_mae_mean", "type": "quantitative", "label": "Entropy MAE"}]}, "palette": {"kind": "categorical", "name": "dataset"}, "legend": {"position": "bottom", "title": "Dataset"}, "layout": "full", "maxRows": 30, "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "lambda-pareto", "title": "Accuracy loss versus entropy error", "subtitle": "Each point is one learned-mask seed; lower-left is preferable.", "showDescription": True, "intent": "relationship", "question": "Is uncertainty fidelity purchased with task accuracy?", "rationale": "The frontier makes the accuracy–uncertainty tradeoff explicit.", "comparisonContext": {"grain": "model, dataset, lambda, seed", "unit": "accuracy loss and entropy MAE", "denominator": "same final evaluation examples"}, "type": "scatter", "dataset": "lambda_pareto", "sourceId": "lambda-pareto-query", "encodings": {"x": {"field": "accuracy_loss", "type": "quantitative", "label": "Accuracy loss vs dense"}, "y": {"field": "entropy_mae", "type": "quantitative", "label": "Entropy MAE"}, "color": {"field": "dataset", "type": "nominal", "label": "Dataset"}, "shape": {"field": "model", "type": "nominal", "label": "Model"}, "tooltip": [{"field": "model", "type": "text", "label": "Model"}, {"field": "dataset", "type": "text", "label": "Dataset"}, {"field": "lambda_uq", "type": "number", "label": "lambda UQ"}, {"field": "seed", "type": "number", "label": "Seed"}, {"field": "accuracy_loss", "type": "quantitative", "label": "Accuracy loss"}, {"field": "entropy_mae", "type": "quantitative", "label": "Entropy MAE"}]}, "palette": {"kind": "categorical", "name": "dataset"}, "legend": {"position": "bottom", "title": "Dataset"}, "layout": "full", "maxRows": 100, "surface": {"surface": "explorer", "viewMode": "both"}},
    ])
    tables = [
        {"id": "paired", "title": "Paired accuracy intervals and gates", "subtitle": "Exact paired differences, 5,000-bootstrap 95% intervals, and the predeclared gate outcome for each slice.", "showDescription": True, "dataset": "paired", "sourceId": "paired-query", "density": "spacious", "layout": "full",
         "columns": [{"field": "model", "label": "Model"}, {"field": "dataset", "label": "Dataset"}, {"field": "examples", "label": "Examples", "format": "number"}, {"field": "difference", "label": "UPGD minus baseline", "format": "percent"}, {"field": "ci_low", "label": "CI low", "format": "percent"}, {"field": "ci_high", "label": "CI high", "format": "percent"}, {"field": "gate", "label": "Gate"}]},
        {"id": "metrics", "title": "Compressed-model fidelity metrics", "subtitle": "Metrics are compared with each model's own dense reference; lower KL, AURC, and ECE are better.", "showDescription": True, "dataset": "metrics", "sourceId": "metric-query", "density": "spacious", "layout": "full",
         "columns": [{"field": "model", "label": "Model"}, {"field": "dataset", "label": "Dataset"}, {"field": "condition", "label": "Condition"}, {"field": "accuracy", "label": "Accuracy", "format": "percent"}, {"field": "kl", "label": "KL from dense", "format": "number"}, {"field": "agreement", "label": "Dense agreement", "format": "percent"}, {"field": "entropy_rho", "label": "Entropy rho", "format": "number"}, {"field": "aurc", "label": "AURC", "format": "number"}, {"field": "ece", "label": "ECE", "format": "number"}]},
        {"id": "ablation-table", "title": "Ablation results and uncertainty-preservation diagnostics", "subtitle": "Partial objectives and random controls are evaluated on the same 500 HellaSwag examples for both models.", "showDescription": True, "dataset": "ablation", "sourceId": "ablation-query", "density": "spacious", "layout": "full",
         "columns": [{"field": "model", "label": "Model"}, {"field": "method", "label": "Method"}, {"field": "accuracy", "label": "Accuracy", "format": "percent"}, {"field": "difference", "label": "Vs one-shot", "format": "percent"}, {"field": "ci_low", "label": "CI low", "format": "percent"}, {"field": "ci_high", "label": "CI high", "format": "percent"}, {"field": "kl", "label": "KL", "format": "number"}, {"field": "entropy_mae", "label": "Entropy MAE", "format": "number"}, {"field": "margin_mae", "label": "Margin MAE", "format": "number"}, {"field": "gate", "label": "Gate"}]},
        {"id": "previous-ablation-table", "title": "Prior-cohort ablation results", "subtitle": "Exact paired effects on the earlier MMLU auxiliary and ARC-Challenge populations.", "showDescription": True, "dataset": "previous_ablation", "sourceId": "previous-ablation-query", "density": "spacious", "layout": "full",
         "columns": [{"field": "model", "label": "Model"}, {"field": "dataset", "label": "Dataset"}, {"field": "method", "label": "Method"}, {"field": "accuracy", "label": "Accuracy", "format": "percent"}, {"field": "difference", "label": "Vs one-shot", "format": "percent"}, {"field": "ci_low", "label": "CI low", "format": "percent"}, {"field": "ci_high", "label": "CI high", "format": "percent"}, {"field": "kl", "label": "KL", "format": "number"}, {"field": "entropy_mae", "label": "Entropy MAE", "format": "number"}, {"field": "margin_mae", "label": "Margin MAE", "format": "number"}, {"field": "gate", "label": "Gate"}]},
    ]

    tables.extend([
        {"id": "lambda-summary", "title": "Lambda-UQ seed summaries", "subtitle": "Mean and sample SD across three learned masks for every model/dataset/lambda cell.", "showDescription": True, "dataset": "lambda_summary", "sourceId": "lambda-summary-query", "density": "spacious", "layout": "full", "columns": [{"field": "model", "label": "Model"}, {"field": "dataset", "label": "Dataset"}, {"field": "lambda_uq", "label": "lambda UQ", "format": "number"}, {"field": "accuracy_mean", "label": "Mean accuracy", "format": "percent"}, {"field": "accuracy_sd", "label": "Seed SD", "format": "percent"}, {"field": "delta_vs_dense_mean", "label": "Vs dense", "format": "percent"}, {"field": "entropy_mae_mean", "label": "Entropy MAE", "format": "number"}, {"field": "kl_mean", "label": "KL", "format": "number"}, {"field": "ece_mean", "label": "ECE", "format": "number"}]},
        {"id": "lambda-paired", "title": "Paired lambda-UQ effects versus lambda=0", "subtitle": "Seed-matched paired bootstrap deltas for accuracy, KL, entropy error, and margin error.", "showDescription": True, "dataset": "lambda_paired", "sourceId": "lambda-paired-query", "density": "spacious", "layout": "full", "columns": [{"field": "model", "label": "Model"}, {"field": "dataset", "label": "Dataset"}, {"field": "lambda_uq", "label": "lambda UQ", "format": "number"}, {"field": "seed", "label": "Seed", "format": "number"}, {"field": "metric", "label": "Metric"}, {"field": "difference", "label": "Delta vs lambda=0", "format": "number"}, {"field": "ci_low", "label": "CI low", "format": "number"}, {"field": "ci_high", "label": "CI high", "format": "number"}]},
    ])
    blocks = [
        {"id": "title", "type": "markdown", "body": "# UPGD: Consolidated Iterations 11–16"},
        {"id": "summary", "type": "markdown", "sourceId": "paired-query", "body": "## Technical summary\n\nThis single report consolidates the same Uncertainty-Preserving Gate Distillation (UPGD) method across Iterations 11–16, two model scales, the primary cohorts, transfer datasets, controlled ablations, and the new lambda-UQ study. It improves the matched 50% one-shot baseline on the 1.5B MMLU auxiliary slice (+7.50 points, 95% CI [+2.50, +12.50]) and on 3B ARC-Challenge (+5.69 points, [+0.67, +10.70]). The effect is unresolved on 1.5B MMLU-Pro (+1.90 points, [-0.60, +4.30]) and negative on 3B MMLU-Pro (-2.50 points, [-5.30, +0.30]). Iteration 16 directly compares lambda-UQ {0, .25, .5, 1} across three disjoint-task calibration sources and three learned-mask seeds per cell. The combined evidence supports UPGD as a promising joint gate optimizer, but dataset, model, and regularization-strength transfer remain sensitive."},
        {"id": "accuracy-text", "type": "markdown", "sourceId": "accuracy-query", "body": "## Aggregate accuracy is positive in three slices, but not uniformly\n\nThe accuracy chart puts dense, one-shot, and UPGD conditions on one scale across both models. The 1.5B BF16 run reaches 41.83% versus 34.33% on MMLU auxiliary and 13.20% versus 11.30% on MMLU-Pro. The 3B NF4 run reaches 31.77% versus 26.09% on ARC-Challenge, but 11.00% versus 13.50% on MMLU-Pro. The MMLU-Pro reversal is the key cross-model warning."},
        {"id": "accuracy-chart", "type": "chart", "chartId": "accuracy", "layout": "full"},
        {"id": "uplift-text", "type": "markdown", "sourceId": "paired-query", "body": "## The common MMLU-Pro dataset changes sign across model scale\n\nUPGD's paired advantage is +1.90 points on the 1.5B MMLU-Pro run but -2.50 points on the 3B NF4 run. Both intervals include zero, so this is evidence of instability rather than a definitive model-size effect. ARC is the only larger-model slice whose interval excludes zero, and its result should be treated as a task-specific positive transfer until repeated."},
        {"id": "uplift-chart", "type": "chart", "chartId": "uplift", "layout": "full"},
        {"id": "paired-table-block", "type": "table", "tableId": "paired", "layout": "full"},
        {"id": "transfer-text", "type": "markdown", "sourceId": "transfer-query", "body": "## Four additional datasets broaden the transfer picture\n\nOn the new 500-example cohorts, the 1.5B BF16 model gains +3.0 points on CommonsenseQA, +4.2 on HellaSwag, and +2.4 on OpenBookQA, but loses 0.8 on Winogrande. The 3B NF4 model gains 0.4 on CommonsenseQA and +2.2 on OpenBookQA, but loses 2.6 on HellaSwag and 2.2 on Winogrande. Seven of eight new paired intervals include zero; the lone interval excluding zero is the 1.5B HellaSwag gain, whose conservative gate still fails on other robustness criteria. These results are useful transfer diagnostics rather than confirmatory wins."},
        {"id": "strata-text", "type": "markdown", "sourceId": "strata-query", "body": "## Uncertainty-stratum behavior is not stable across slices\n\nThe 1.5B MMLU auxiliary run gains +18.0 and +7.5 points in its low and middle strata but loses 3.0 points in the high stratum. The 1.5B MMLU-Pro run loses 2.40 points in the low stratum and gains in the middle and high strata. The 3B NF4 ARC run gains in all three strata, while 3B NF4 MMLU-Pro loses in all three. The robust-term interpretation therefore does not transfer as a conditional guarantee."},
        {"id": "strata-chart", "type": "chart", "chartId": "strata", "layout": "full"},
        {"id": "metrics-text", "type": "markdown", "sourceId": "metric-query", "body": "## Fidelity metrics improve selectively, not universally\n\nUPGD reduces KL from the dense reference in all four slices. Dense agreement, AURC, and ECE also improve in the 1.5B slices and on 3B ARC. On 3B MMLU-Pro, KL and ECE improve, but dense agreement, entropy-rank preservation, and AURC worsen. Matching distributions alone is therefore not sufficient evidence of task accuracy or uncertainty robustness."},
        {"id": "metrics-table-block", "type": "table", "tableId": "metrics", "layout": "full"},
        {"id": "ablation-text", "type": "markdown", "sourceId": "ablation-query", "body": "## Ablation: joint gate learning is the main contribution, while extra terms are model-sensitive\n\nOn the strongest new-transfer cohort (500 HellaSwag examples), random masks fall below the one-shot baseline for both models. Across three random masks, the mean accuracy difference is -4.7 ± 3.4 points for 1.5B and -9.7 ± 4.6 points for 3B (sample SD). Learned KL-only gates recover +3.2 points on 1.5B but are flat on 3B (-0.2). Adding entropy and margin matching reaches +3.6 points on 1.5B and -1.6 on 3B; the reused full UPGD masks reach +4.4 and -2.6, respectively. Every paired interval crosses zero except the 1.5B full-UPGD interval, which is narrowly positive [+0.6, +8.4] on this rerun. The component pattern supports joint gate optimization as the methodological contribution, but does not support a model-invariant guarantee."},
        {"id": "ablation-chart", "type": "chart", "chartId": "ablation", "layout": "full"},
        {"id": "ablation-table-block", "type": "table", "tableId": "ablation-table", "layout": "full"},
        {"id": "previous-ablation-text", "type": "markdown", "sourceId": "previous-ablation-query", "body": "## Earlier cohorts repeat the positive task-specific pattern, not a universal objective ranking\n\nOn the prior MMLU auxiliary cohort, 1.5B KL-only gates are strongest at +9.5 points (95% CI [+4.5,+14.3]); full UPGD remains positive at +7.7 points ([+2.7,+12.8]), while KL+entropy+margin is nearly flat at +0.5. On ARC-Challenge, full UPGD is positive for both models: +2.3 points on 1.5B and +5.7 on 3B NF4, with the latter interval excluding zero ([+0.7,+10.7]). The cross-cohort counterexamples matter: full UPGD is -4.7 points on 3B MMLU auxiliary and only +2.3 on 1.5B ARC. These results reinforce that joint gate optimization is useful, but the uncertainty terms remain task- and model-sensitive."},
        {"id": "previous-ablation-chart", "type": "chart", "chartId": "previous-ablation", "layout": "full"},
        {"id": "previous-ablation-table-block", "type": "table", "tableId": "previous-ablation-table", "layout": "full"},
        {"id": "lambda-text", "type": "markdown", "sourceId": "lambda-summary-query", "body": "## Iteration 16: uncertainty regularization is a controllable sweep, not a universal knob\n\nThe new study runs lambda-UQ in {0, 0.25, 0.5, 1.0} for both models, three seeds per value, and the same 300-step matched budget. Lambda=0 is the KL-only joint-gate control; positive lambda adds entropy, margin, and worst-cell pressure. On 1.5B, mean HellaSwag accuracy rises from 30.1% at lambda=0 to 32.1% at lambda=1, and ARC rises modestly from 24.6% to 25.9%; MMLU auxiliary instead falls from 28.2% to 26.7% while entropy MAE drops. On 3B, HellaSwag rises from 28.9% to 33.1%, MMLU auxiliary peaks at 28.2% for lambda=.25, and ARC declines from 27.1% to 26.3%. Most seed-matched paired intervals cross zero; one 3B HellaSwag seed supplies a clear positive interval. The sweep therefore isolates a real accuracy–uncertainty tradeoff without supporting a universal best lambda. A separate held-out lambda-selection run and physical deployment benchmark were not executed in this pass; these results are comparative study evidence, not a selected production setting."},
        {"id": "lambda-accuracy-chart", "type": "chart", "chartId": "lambda-accuracy", "layout": "full"},
        {"id": "lambda-entropy-chart", "type": "chart", "chartId": "lambda-entropy", "layout": "full"},
        {"id": "lambda-pareto-chart", "type": "chart", "chartId": "lambda-pareto", "layout": "full"},
        {"id": "lambda-summary-table", "type": "table", "tableId": "lambda-summary", "layout": "full"},
        {"id": "lambda-paired-table", "type": "table", "tableId": "lambda-paired", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "design-query", "body": "## Scope, model specifications, and metric definitions\n\nThe method is held constant: stochastic exact-top-k FFN gate distillation through a frozen model, with dense answer-distribution KL, entropy and margin matching, and a smooth worst-stratum term. The 1.5B BF16 run has 28 layers and retains 4,480 of 8,960 channels per layer; its original masks are physically exported. The 3B NF4 run has 36 layers and retains 5,504 of 11,008 channels per layer; its original quality evaluation uses logical channel hooks around quantized Linear4bit modules. The expanded transfer pass uses the same logical-channel path for both models. Cohorts are 600 MMLU auxiliary examples, 1,000 MMLU-Pro examples, 299 ARC-Challenge examples, and four additional 500-example cohorts: CommonsenseQA, HellaSwag, OpenBookQA, and Winogrande. Accuracy is exact-choice accuracy; KL, dense agreement, entropy Spearman, AURC, and ECE compare each masked model with its own dense reference."},
        {"id": "method-contribution", "type": "markdown", "sourceId": "method-query", "body": "## What UPGD contributes methodologically\n\nUPGD is not an uncertainty score that ranks channels locally, and it is not a router that sends uncertain examples to a larger model. Its contribution is a **joint, exact-budget optimization of all FFN channel decisions through the frozen network**, with predictive uncertainty treated as behavior to preserve. The learned mask is therefore selected for its effect on the full answer distribution and uncertainty geometry, not because an individual channel has a high activation or uncertainty score.\n\nThe method combines four ideas in one constrained objective: global gate learning, stochastic exact-budget subnetworks, dense-behavior distillation, and an explicit worst-stratum pressure. That combination is what distinguishes UPGD from one-shot activation-importance pruning and from uncertainty-driven routing."},
        {"id": "method-mechanics", "type": "markdown", "sourceId": "method-query", "body": "## How one UPGD training step works\n\nFor an input x, the frozen dense model produces an answer distribution p_dense and predictive entropy H_dense. Each layer l has gate logits a_l for its FFN channels. UPGD adds annealed logistic noise, selects the exact top-k channels m_l (k = 50% of that layer), and uses a straight-through relaxation so the gate logits receive gradients while the forward pass respects the hard budget.\n\nThe compressed forward pass produces p_mask, entropy H_mask, and a logit margin M_mask. The optimization target is:\n\n`L = KL(p_dense || p_mask) + 1.0(H_dense - H_mask)^2 + 0.5(M_dense - M_mask)^2 + 0.5 L_smooth-worst-stratum`\n\nEach calibration batch contains examples from three dense-entropy strata. The smooth worst-stratum term prevents the mean loss from ignoring a weak uncertainty region, but the combined results show that this pressure is not yet a reliable conditional-robustness guarantee. After optimization, only the deterministic retained-channel indices are needed for inference."},
        {"id": "method", "type": "markdown", "sourceId": "design-query", "body": "## Method and protocol differences that matter\n\nBoth runs freeze model weights and learn one gate logit per FFN channel under an exact 50% budget. The 1.5B run completed the original 600-step schedule and physically removed unselected channels in its primary evaluation. The larger 3B NF4 run required explicit graph cleanup and a resource-adapted 300-step schedule after the original 600-step configuration reached a late GPU OOM. The expanded four-dataset pass uses logical channel hooks for both models so the new cross-model comparison shares one execution protocol. Thus the combined report validates method transfer across scale and data, but it is not an apples-to-apples deployment or optimization-budget comparison."},
        {"id": "limitations", "type": "markdown", "sourceId": "validation-query", "body": "## Limitations, uncertainty, and robustness checks\n\nThe underlying validation artifacts pass 50/50 checks for the 1.5B run, 37/37 for the 3B run, and 80/80 for the expanded transfer artifacts; the repository test suite passes 49 tests. There is one gate seed per model, no fresh benchmark family, different training-step budgets, and different pruning execution modes between primary and transfer runs. MMLU-Pro was used earlier in the research program and is exploratory rather than confirmatory. All four new cohorts are 500-example diagnostics, and none clears the conservative gate. The logical-mask path does not support packed size, sparse-kernel latency, peak-memory, energy, or throughput claims. The evidence establishes mixed transfer, not a general compression guarantee."},
        {"id": "history-text", "type": "markdown", "body": "## What the recent method changes\n\nThe recent UPGD method differs from earlier local uncertainty heuristics by optimizing all channel decisions jointly through the frozen network while treating uncertainty as a behavior-preservation target. The combined evidence supports that design choice as a useful research direction, while showing that its gains remain task- and model-dependent."},
        {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Run at least three seeds per model on a fresh multi-task evaluation mixture, including ARC- and MMLU-Pro-like task families.\n2. Compare the full objective with a KL-only joint-gate ablation at matched steps and memory.\n3. Re-run the 3B NF4 condition at 600 steps if memory can be bounded, to separate resource adaptation from model-scale transfer.\n4. Implement packed physical export for the quantized model and measure latency, peak VRAM, and throughput separately from quality.\n5. Keep the claim at “mixed transfer” until the shared MMLU-Pro sign reversal is resolved."},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\nDoes the ARC gain reflect calibration-format alignment rather than a general 3B benefit? Is the 3B MMLU-Pro loss caused by the shorter resource-adapted run or by a genuinely misaligned uncertainty objective? Can a multi-task calibration distribution stabilize the stratum effect without sacrificing aggregate accuracy? Does physical packing preserve the quality gains once quantized kernels are actually changed?"},
    ]

    snapshot = {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {
        "accuracy": accuracy, "paired": paired, "metrics": metrics, "strata": strata, "ablation": ablation, "ablation_random": ablation_random,
        "previous_ablation": previous_ablation, "previous_ablation_random": previous_random,
        "lambda_summary": uncertainty_summary, "lambda_paired": uncertainty_paired, "lambda_pareto": uncertainty_pareto,
    }}
    manifest = {"version": 1, "surface": "report", "title": "UPGD: Consolidated Iterations 11–16",
                "description": "Single consolidated report for the Iteration 11–16 UPGD method, model-scale, transfer, ablation, and uncertainty-regularization evaluations.",
                "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks}
    artifact = {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}
    (DEST / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    chart_map = [
        {"section": "Model and dataset accuracy", "question": "How does UPGD behave across slices?", "type": "grouped bar", "fields": ["dataset", "condition", "accuracy"], "claim": "Positive in three slices, negative on 3B MMLU-Pro."},
        {"section": "Paired transfer effect", "question": "Is the gain consistent?", "type": "signed grouped bar", "fields": ["model", "dataset", "difference"], "claim": "The shared MMLU-Pro slice changes sign across model scale."},
        {"section": "Uncertainty strata", "question": "Is robustness conditional?", "type": "signed grouped bar", "fields": ["slice", "stratum", "accuracy_difference"], "claim": "Stratum effects are slice-sensitive."},
        {"section": "Objective ablation", "question": "Which terms add transfer value?", "type": "signed grouped bar", "fields": ["model", "method", "difference"], "claim": "Joint learned gates improve over random controls, but component gains are model-sensitive."},
        {"section": "Prior-cohort objective ablation", "question": "Does the ablation pattern repeat?", "type": "signed grouped bar", "fields": ["model", "dataset", "method", "difference"], "claim": "Full UPGD is positive on prior ARC for both models and on prior MMLU auxiliary for 1.5B, but not universally."},
    ]
    (DEST / "chart_map.json").write_text(json.dumps(chart_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
