"""Build the portable technical report for the Qwen2.5-3B NF4 transfer."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter12_q4_3b")
DEST = ROOT / "report"


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
    values = ",\n    ".join("(" + ", ".join(literal(row.get(column)) for column in columns) + ")" for row in rows)
    return {"id": source_id, "label": label, "query": {
        "engine": "SQLite", "language": "sql",
        "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
        "description": f"Materialize reviewed {label.lower()} rows.", "executed_at": generated,
        "filters": filters, "metric_definitions": definitions,
    }}


def main() -> int:
    arc = json.loads((ROOT / "arc_analysis.json").read_text(encoding="utf-8"))
    mmlu = json.loads((ROOT / "mmlu_pro_analysis.json").read_text(encoding="utf-8"))
    training = json.loads((ROOT / "training.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "scoring_manifest.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for filename in ("arc_analysis.json", "mmlu_pro_analysis.json", "training.json", "validation.json", "scoring_manifest.json"):
        shutil.copyfile(ROOT / filename, DEST / filename)

    accuracy_rows, metric_rows, stratum_rows, paired_rows = [], [], [], []
    for dataset, analysis, n in (("ARC-Challenge", arc, 299), ("MMLU-Pro", mmlu, 1000)):
        metrics = {row["condition"]: row for row in analysis["metric_rows"]}
        for condition, label in (("dense", "Dense 3B NF4"), ("baseline_aggressive", "One-shot mask"), ("upgd_aggressive", "UPGD")):
            accuracy_rows.append({"dataset": dataset, "condition": label, "accuracy": metrics[condition]["accuracy"], "examples": n})
        for condition, label in (("baseline_aggressive", "One-shot mask"), ("upgd_aggressive", "UPGD")):
            metric_rows.append({"dataset": dataset, "condition": label, "examples": n,
                                "accuracy": metrics[condition]["accuracy"], "kl": metrics[condition]["mean_kl_from_dense"],
                                "agreement": metrics[condition]["prediction_agreement_with_dense"],
                                "entropy_rho": metrics[condition]["entropy_spearman_with_dense"],
                                "aurc": metrics[condition]["aurc"], "ece": metrics[condition]["ece"]})
        for row in analysis["stratum_rows"]:
            if row["condition"] == "upgd_aggressive":
                baseline_row = next(r for r in analysis["stratum_rows"] if r["condition"] == "baseline_aggressive" and r["stratum"] == row["stratum"])
                stratum_rows.append({"dataset": dataset, "stratum": f"Stratum {row['stratum'] + 1}",
                                     "accuracy_difference": row["accuracy"] - baseline_row["accuracy"], "examples": row["examples"]})
        paired = analysis["paired_comparison"]
        paired_rows.append({"dataset": dataset, "examples": n, "difference": paired["difference"],
                            "ci_low": paired["bootstrap_95_ci"][0], "ci_high": paired["bootstrap_95_ci"][1],
                            "gate": "PASS" if analysis["gate"]["passed"] else "FAIL"})

    sources = [
        {"id": "arc-json", "label": "3B NF4 ARC analysis", "path": "arc_analysis.json"},
        {"id": "mmlu-json", "label": "3B NF4 MMLU-Pro analysis", "path": "mmlu_pro_analysis.json"},
        {"id": "training-json", "label": "3B NF4 UPGD training record", "path": "training.json"},
        {"id": "validation-json", "label": "Independent Iteration 12 validation", "path": "validation.json"},
        {"id": "model-card", "label": "Qwen2.5-3B-Instruct model card", "href": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct"},
        {"id": "bnb-docs", "label": "bitsandbytes quantization documentation", "href": "https://github.com/bitsandbytes-foundation/bitsandbytes"},
        query_source("accuracy-query", "3B NF4 accuracy comparison", accuracy_rows, ["dataset", "condition", "accuracy", "examples"], generated,
                     ["299 ARC and 1,000 MMLU-Pro examples", "dense, one-shot, and UPGD conditions"], ["accuracy is exact-choice accuracy"]),
        query_source("metric-query", "3B NF4 uncertainty and calibration metrics", metric_rows,
                     ["dataset", "condition", "examples", "accuracy", "kl", "agreement", "entropy_rho", "aurc", "ece"], generated,
                     ["logical 50% FFN masks under NF4", "same examples within each dataset"], ["KL, agreement, entropy rho, AURC, and ECE are computed against the dense NF4 reference"]),
        query_source("stratum-query", "3B NF4 per-entropy-stratum differences", stratum_rows,
                     ["dataset", "stratum", "accuracy_difference", "examples"], generated,
                     ["three equal-frequency dense-entropy strata within each dataset"], ["difference is UPGD minus one-shot accuracy"]),
        query_source("paired-query", "3B NF4 paired accuracy intervals", paired_rows,
                     ["dataset", "examples", "difference", "ci_low", "ci_high", "gate"], generated,
                     ["5,000 paired bootstrap replicates", "same examples for UPGD and baseline"], ["difference is UPGD minus one-shot accuracy"]),
    ]

    charts = [
        {"id": "accuracy", "title": "Accuracy under 50% FFN masking", "subtitle": "Dense reference, one-shot mask, and UPGD on fixed ARC and MMLU-Pro cohorts.",
         "showDescription": True, "intent": "comparison", "question": "Does UPGD improve the larger quantized model across datasets?",
         "rationale": "Grouped bars expose the positive ARC transfer and negative MMLU-Pro transfer on a shared zero-based scale.",
         "comparisonContext": {"grain": "dataset and condition", "unit": "accuracy", "denominator": "299 or 1,000 examples"}, "type": "bar", "dataset": "accuracy", "sourceId": "accuracy-query",
         "encodings": {"x": {"field": "dataset", "type": "nominal", "label": "Dataset"}, "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"}, "color": {"field": "condition", "type": "nominal", "label": "Condition"},
                       "tooltip": [{"field": "condition", "type": "text", "label": "Condition"}, {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"}, {"field": "examples", "type": "quantitative", "format": "number", "label": "Examples"}]},
         "palette": {"kind": "categorical", "name": "condition"}, "legend": {"position": "bottom", "title": "Condition"}, "layout": "full", "maxRows": 6, "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "strata", "title": "UPGD accuracy difference by entropy stratum", "subtitle": "Signed UPGD-minus-baseline effect; each dataset is split into dense-entropy thirds.",
         "showDescription": True, "intent": "comparison", "question": "Is the larger-model gain stable across predictive-uncertainty strata?",
         "rationale": "A common zero reference makes the cross-dataset sign reversal explicit.",
         "comparisonContext": {"grain": "dataset and entropy stratum", "unit": "accuracy difference", "denominator": "about one third of each cohort"}, "type": "bar", "dataset": "strata", "sourceId": "stratum-query",
         "encodings": {"x": {"field": "stratum", "type": "nominal", "label": "Dense-entropy stratum"}, "y": {"field": "accuracy_difference", "type": "quantitative", "format": "percent", "label": "Accuracy difference"}, "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                       "tooltip": [{"field": "dataset", "type": "text", "label": "Dataset"}, {"field": "accuracy_difference", "type": "quantitative", "format": "percent", "label": "Difference"}, {"field": "examples", "type": "quantitative", "format": "number", "label": "Examples"}]},
         "palette": {"kind": "categorical", "name": "dataset"}, "legend": {"position": "bottom", "title": "Dataset"}, "layout": "full", "maxRows": 6, "surface": {"surface": "explorer", "viewMode": "both"}},
    ]
    tables = [{"id": "metrics", "title": "3B NF4 quality metrics", "subtitle": "Exact uncertainty and calibration metrics for both 50% logical masks.", "showDescription": True, "dataset": "metrics", "sourceId": "metric-query", "density": "spacious", "layout": "full",
               "columns": [{"field": "dataset", "label": "Dataset"}, {"field": "condition", "label": "Condition"}, {"field": "accuracy", "label": "Accuracy", "format": "percent"}, {"field": "kl", "label": "KL from dense", "format": "number"}, {"field": "agreement", "label": "Dense agreement", "format": "percent"}, {"field": "entropy_rho", "label": "Entropy rho", "format": "number"}, {"field": "aurc", "label": "AURC", "format": "number"}, {"field": "ece", "label": "ECE", "format": "number"}]}]

    arc_paired, mmlu_paired = paired_rows
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Qwen2.5-3B NF4 UPGD Transfer"},
        {"id": "summary", "type": "markdown", "sourceId": "paired-query", "body":
         "## Technical summary\n\nUPGD is feasible on the larger Qwen2.5-3B model in 4-bit NF4 on an 8GB RTX 3070, but transfer is mixed. On 299 ARC-Challenge examples, UPGD improves the one-shot 50% FFN mask by "
         f"{arc_paired['difference']*100:.2f} points with a paired 95% interval [{arc_paired['ci_low']*100:.2f}, {arc_paired['ci_high']*100:.2f}] and passes the predeclared gate. On 1,000 MMLU-Pro examples, it changes accuracy by {mmlu_paired['difference']*100:+.2f} points with interval [{mmlu_paired['ci_low']*100:.2f}, {mmlu_paired['ci_high']*100:.2f}] and fails the gate. The method is therefore promising on one additional task, but not yet a reliable cross-dataset compression recipe."},
        {"id": "accuracy-text", "type": "markdown", "sourceId": "accuracy-query", "body": "## UPGD helps ARC but hurts MMLU-Pro\n\nThe dense 3B NF4 reference scores 48.83% on ARC and 29.50% on MMLU-Pro. The one-shot masks score 26.09% and 13.50%; UPGD reaches 31.77% and 11.00%. The shared model, quantizer, channel budget, and scoring path make the sign reversal a transfer result rather than a hardware or metric mismatch."},
        {"id": "accuracy-chart", "type": "chart", "chartId": "accuracy", "layout": "full"},
        {"id": "strata-text", "type": "markdown", "sourceId": "stratum-query", "body": "## The dataset reversal is visible within uncertainty strata\n\nARC gains are +13.0, +2.0, and +2.02 points across low, middle, and high dense-entropy thirds. MMLU-Pro loses -2.99, -0.90, and -3.60 points in the corresponding thirds. This rules out a claim that the current uncertainty-preservation term protects a particular uncertainty region after transfer."},
        {"id": "strata-chart", "type": "chart", "chartId": "strata", "layout": "full"},
        {"id": "metrics-table", "type": "table", "tableId": "metrics", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "training-json", "body": "## Scope, data, and metric definitions\n\nThe model is Qwen2.5-3B-Instruct loaded with bitsandbytes NF4 4-bit quantization and float16 compute. The gate run uses 2,400 fixed MMLU auxiliary training examples, 300 optimization steps, one example from each of three dense-entropy strata per step, and an exact 50% per-layer FFN budget (5,504 of 11,008 channels across 36 layers). ARC-Challenge contains 299 labeled examples; MMLU-Pro contains 1,000. Accuracy is exact-choice accuracy. KL, dense agreement, entropy Spearman, AURC, and ECE compare each masked model with the same dense NF4 reference."},
        {"id": "method", "type": "markdown", "sourceId": "training-json", "body": "## Method and resource adaptation\n\nUPGD jointly learns gate logits through the frozen quantized model using stochastic exact-top-k masks, dense answer-distribution KL, entropy and margin matching, and a smooth worst-stratum term. The original 600-step configuration reached an OOM late in 3B NF4 backward training; explicit graph cleanup and expandable CUDA allocator segments allowed a 300-step run to complete in 581.9 seconds. Quantized Linear4bit modules were not physically replaced: evaluation applies the exact retained-channel mask immediately before down projection. This is a quality validation under NF4, not a sparse-kernel deployment benchmark."},
        {"id": "limitations", "type": "markdown", "sourceId": "validation-json", "body": "## Limitations, uncertainty, and robustness checks\n\nIndependent validation passes 37/37 checks and the Python test suite passes 49 tests. The transfer uses one larger model, one gate seed, one resource-adapted step count, and two fixed cohorts; MMLU-Pro was used earlier in the research program and is exploratory rather than confirmatory. The ARC interval excludes zero, while the MMLU-Pro interval includes zero and is negative point-estimate. The logical mask leaves dense quantized Linear4bit computation intact, so no packed size, sparse-kernel latency, peak-memory reduction, energy, or throughput claim is supported."},
        {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Run at least three 3B NF4 gate seeds on a fresh benchmark split before tuning the objective around ARC.\n2. Add an MMLU-Pro-like calibration split or a held-out task-family mixture; the current MMLU auxiliary calibration does not transfer reliably.\n3. Compare UPGD against a KL-only joint gate objective to isolate whether entropy/margin terms help or hurt transfer.\n4. Implement a true packed 4-bit structured export and measure latency and peak memory separately from quality."},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\nDoes the ARC gain reflect task-format alignment with the MMLU-style calibration data? Is the MMLU-Pro loss caused by under-training at 300 steps, or by a genuinely misaligned uncertainty objective? Can a multi-task calibration distribution stabilize the stratum effect without collapsing back to local activation importance?"},
    ]
    manifest_obj = {"version": 1, "surface": "report", "title": "Qwen2.5-3B NF4 UPGD Transfer", "description": "Larger-model quantized validation across ARC and MMLU-Pro.", "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks}
    artifact = {"surface": "report", "manifest": manifest_obj, "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {"accuracy": accuracy_rows, "metrics": metric_rows, "strata": stratum_rows, "paired": paired_rows}}, "sources": sources}
    (DEST / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    chart_map = [{"section": "Transfer accuracy", "question": "Does UPGD transfer?", "type": "grouped bar", "fields": ["dataset", "condition", "accuracy"], "claim": "ARC improves; MMLU-Pro declines."}, {"section": "Stratum stability", "question": "Are gains stable across entropy strata?", "type": "signed grouped bar", "fields": ["dataset", "stratum", "accuracy_difference"], "claim": "The effect is dataset-sensitive."}]
    (DEST / "chart_map.json").write_text(json.dumps(chart_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
