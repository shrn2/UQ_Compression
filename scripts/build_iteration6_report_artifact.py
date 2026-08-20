"""Build the canonical portable technical report artifact for Iteration 6."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter6_arc")
DEST = ROOT / "report"


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def query_source(source_id, label, rows, columns, generated, filters, definitions):
    values = ",\n    ".join(
        "(" + ", ".join(literal(row.get(column)) for column in columns) + ")"
        for row in rows
    )
    return {"id": source_id, "label": label, "query": {
        "engine": "SQLite", "language": "sql",
        "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
        "description": f"Materialize reviewed {label.lower()} rows.",
        "executed_at": generated, "filters": filters, "metric_definitions": definitions,
    }}


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for source_path, target_name in (
        (ROOT / "summary.json", "summary.json"),
        (ROOT / "analysis.json", "analysis.json"),
        (ROOT / "benchmark/results.json", "benchmark_results.json"),
    ):
        shutil.copyfile(source_path, DEST / target_name)

    static = summary["static_rows"]
    estimators = [
        {
            "estimator": row["estimator"], "auroc": row["auroc"], "auprc": row["auprc"],
            "auroc_ci_low": row["auroc_ci"][0], "auroc_ci_high": row["auroc_ci"][1],
            "precision_at_0.1": row["precision_at_0.1"],
            "precision_at_0.2": row["precision_at_0.2"],
            "precision_at_0.3": row["precision_at_0.3"],
        }
        for row in summary["estimator_rows"]
    ]
    frontier = summary["frontier_rows"]
    policies = summary["policy_rows"]
    comparisons = [
        {
            "target_rate": row["target_rate"],
            "difference": row["difference"],
            "ci_low": row["bootstrap_95_ci"][0],
            "ci_high": row["bootstrap_95_ci"][1],
            "dense_execution_rate_difference": row["dense_execution_rate_difference"],
        }
        for row in summary["uplift_entropy_comparisons"]
    ]
    sources = [
        {"id": "summary-json", "label": "Reviewed Iteration 6 summary", "path": "summary.json"},
        {"id": "analysis-json", "label": "Calibration uplift analysis", "path": "analysis.json"},
        {"id": "benchmark-json", "label": "Held-out ARC benchmark", "path": "benchmark_results.json"},
        query_source("static-query", "static sparse hierarchy", static,
            ["variant", "accuracy", "normalized_linear_flops", "prediction_agreement_with_dense", "output_kl_from_dense", "ece", "error_auroc", "aurc"], generated,
            ["750 ARC training examples", "physical nested FFN channel pruning"],
            ["normalized linear FLOPs is a transformer matrix-parameter proxy excluding embeddings and LM head"]),
        query_source("estimator-query", "densification-benefit discrimination", estimators,
            ["estimator", "auroc", "auprc", "precision_at_0.1", "precision_at_0.2", "precision_at_0.3"], generated,
            ["five-fold out-of-fold predictions for learned probes", "180 aggressive-to-dense positive events"],
            ["positive means aggressive is wrong and dense is correct", "AUROC and AUPRC target densification benefit"]),
        query_source("frontier-query", "held-out accuracy-compute frontier", frontier,
            ["method", "target_rate", "dense_execution_rate", "accuracy", "expected_sequential_compute", "kind"], generated,
            ["299 untouched ARC validation examples", "thresholds frozen on calibration", "oracle rows are diagnostic only"],
            ["sequential compute equals aggressive-pass proxy plus observed dense re-execution rate"]),
        query_source("policy-query", "held-out policy outcomes", policies,
            ["method", "target_rate", "dense_execution_rate", "accuracy", "expected_sequential_compute", "densification_correction_rate", "wasteful_densification_rate", "missed_correction_rate"], generated,
            ["calibration-frozen thresholds", "non-oracle policies only"],
            ["correction rate is conditional on dense execution", "waste is aggressive-correct among dense executions"]),
        query_source("comparison-query", "uplift versus entropy comparisons", comparisons,
            ["target_rate", "difference", "ci_low", "ci_high", "dense_execution_rate_difference"], generated,
            ["paired 2,000-replicate bootstrap intervals are stored in benchmark_results.json"],
            ["difference is uplift-probe minus entropy correctness on the same 299 examples"]),
    ]

    charts = [
        {
            "id": "static-frontier", "title": "Static accuracy falls monotonically with FFN pruning",
            "subtitle": "Nested physical channel removal on 750 ARC calibration examples.", "showDescription": True,
            "intent": "relationship", "question": "How much calibration accuracy is lost as the matrix-compute proxy falls?",
            "rationale": "A point view exposes the three measured capacity levels without implying runtime speedup.",
            "comparisonContext": {"grain": "sparsity variant", "unit": "proportion", "denominator": "750 calibration examples"},
            "type": "scatter", "dataset": "static", "sourceId": "static-query",
            "encodings": {
                "x": {"field": "normalized_linear_flops", "type": "quantitative", "format": "percent", "label": "Normalized matrix-compute proxy"},
                "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                "color": {"field": "variant", "type": "nominal", "label": "Variant"},
                "label": {"field": "variant", "type": "nominal", "label": "Variant"},
                "tooltip": [
                    {"field": "variant", "type": "text", "label": "Variant"},
                    {"field": "normalized_linear_flops", "type": "quantitative", "format": "percent", "label": "Compute proxy"},
                    {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                ],
            },
            "palette": {"kind": "categorical", "name": "capacity"}, "legend": {"position": "bottom", "title": "Variant"},
            "layout": "full", "maxRows": len(static), "surface": {"surface": "explorer", "viewMode": "both"},
        },
        {
            "id": "estimator-auroc", "title": "Prediction disagreement is strongest, but is not a sparse-only signal",
            "subtitle": "Calibration AUROC for aggressive-to-dense correction events; learned probes are out-of-fold.", "showDescription": True,
            "intent": "comparison", "question": "Which uncertainty signal best identifies examples corrected by densification?",
            "rationale": "Bars compare discrimination at a common target and sample.",
            "comparisonContext": {"grain": "estimator", "unit": "AUROC", "denominator": "750 calibration examples; 180 positives"},
            "type": "bar", "dataset": "estimators", "sourceId": "estimator-query",
            "encodings": {
                "x": {"field": "estimator", "type": "nominal", "label": "Estimator"},
                "y": {"field": "auroc", "type": "quantitative", "format": "number", "label": "Densification-benefit AUROC"},
                "color": {"field": "estimator", "type": "nominal", "label": "Estimator"},
                "tooltip": [
                    {"field": "estimator", "type": "text", "label": "Estimator"},
                    {"field": "auroc", "type": "quantitative", "format": "number", "label": "AUROC"},
                    {"field": "auprc", "type": "quantitative", "format": "number", "label": "AUPRC"},
                ],
            },
            "palette": {"kind": "categorical", "name": "estimators"}, "legend": {"position": "bottom", "title": "Estimator"},
            "layout": "full", "maxRows": len(estimators), "surface": {"surface": "explorer", "viewMode": "both"},
        },
        {
            "id": "dynamic-frontier", "title": "Held-out accuracy versus sequential compute",
            "subtitle": "Calibration-frozen thresholds; oracle policies are diagnostic ceilings, not deployable results.", "showDescription": True,
            "intent": "relationship", "question": "Do learned or uncertainty policies improve the held-out accuracy-compute frontier?",
            "rationale": "A compute-accuracy scatter preserves the actual execution rate induced by each frozen threshold.",
            "comparisonContext": {"grain": "method and calibration target", "unit": "proportion", "denominator": "299 held-out ARC examples"},
            "type": "scatter", "dataset": "frontier", "sourceId": "frontier-query",
            "encodings": {
                "x": {"field": "expected_sequential_compute", "type": "quantitative", "format": "number", "label": "Expected sequential compute proxy"},
                "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "label": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "method", "type": "text", "label": "Method"},
                    {"field": "target_rate", "type": "quantitative", "format": "percent", "label": "Calibration target"},
                    {"field": "dense_execution_rate", "type": "quantitative", "format": "percent", "label": "Held-out dense rate"},
                    {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                ],
            },
            "palette": {"kind": "categorical", "name": "policies"}, "legend": {"position": "bottom", "title": "Method"},
            "layout": "full", "maxRows": len(frontier), "surface": {"surface": "explorer", "viewMode": "both"},
        },
    ]

    tables = [
        {"id": "static-table", "title": "Static hierarchy diagnostics", "subtitle": "Accuracy, divergence, and calibration metrics for all three physical variants.", "showDescription": True,
         "dataset": "static", "sourceId": "static-query", "density": "spacious", "layout": "full",
         "columns": [
             {"field": "variant", "label": "Variant", "type": "text"},
             {"field": "accuracy", "label": "Accuracy", "format": "percent"},
             {"field": "normalized_linear_flops", "label": "Compute proxy", "format": "percent"},
             {"field": "prediction_agreement_with_dense", "label": "Dense agreement", "format": "percent"},
             {"field": "output_kl_from_dense", "label": "Output KL", "format": "number"},
             {"field": "aurc", "label": "AURC", "format": "number"},
         ]},
        {"id": "policy-table", "title": "Held-out non-oracle policy outcomes", "subtitle": "Actual held-out execution rates and outcomes from calibration-frozen thresholds.", "showDescription": True,
         "dataset": "policies", "sourceId": "policy-query", "density": "dense", "layout": "full",
         "columns": [
             {"field": "method", "label": "Method", "type": "text"},
             {"field": "target_rate", "label": "Target", "format": "percent"},
             {"field": "dense_execution_rate", "label": "Dense rate", "format": "percent"},
             {"field": "accuracy", "label": "Accuracy", "format": "percent"},
             {"field": "expected_sequential_compute", "label": "Sequential compute", "format": "number"},
             {"field": "densification_correction_rate", "label": "Correction / dense", "format": "percent"},
         ]},
    ]

    gate = summary["calibration_gate"]
    dense = next(row for row in static if row["variant"] == "dense")
    medium = next(row for row in static if row["variant"] == "medium")
    aggressive = next(row for row in static if row["variant"] == "aggressive")
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Uncertainty-Guided Progressive Densification: Iteration 6"},
        {"id": "summary", "type": "markdown", "sourceId": "summary-json", "body":
         "## Technical summary: useful correction events exist, but the learned router does not generalize better\n\n"
         f"**Decision: NO-GO for U-Dense in this setup.** Physical 50% FFN-channel pruning reduced the matrix-compute proxy to {aggressive['normalized_linear_flops']:.3f} and produced 180 calibration examples (24.0%) that dense inference corrected. Max-probability uncertainty cleared the literal Phase 1 gate (AUROC {gate['best_sparse_only_auroc']:.3f}), authorizing one held-out ARC evaluation. The learned uplift probe did not beat generic uncertainty on calibration ({gate['uplift_probe_auroc']:.3f} versus {gate['best_generic_auroc']:.3f}) and no held-out operating point beat entropy with a paired 95% interval above zero at equal-or-lower dense execution. The plan therefore stops before three-level routing, MMLU-Pro, and runtime claims."},
        {"id": "static-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Structured pruning creates a meaningful but steep capacity hierarchy\n\n"
         f"Calibration accuracy falls monotonically from {dense['accuracy']:.1%} dense to {medium['accuracy']:.1%} at 25% FFN pruning and {aggressive['accuracy']:.1%} at 50% FFN pruning. Dense prediction agreement falls to {aggressive['prediction_agreement_with_dense']:.1%} in the aggressive model. These are real dimension reductions in gate, up, and down projections; the x-axis is an analytical matrix-parameter proxy, not measured latency."},
        {"id": "static-chart-block", "type": "chart", "chartId": "static-frontier", "layout": "full"},
        {"id": "static-table-block", "type": "table", "tableId": "static-table", "layout": "full"},
        {"id": "estimator-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Generic uncertainty predicts benefit, but uplift learning adds no value\n\n"
         "The sparse-only max-probability signal reaches AUROC 0.633 and entropy reaches 0.630. The five-fold out-of-fold uplift probe reaches 0.622, below both, while the generic error probe is near chance for the uplift target (0.498). Prediction disagreement between aggressive and medium variants reaches 0.709, but it requires executing the intermediate model and was therefore retained as a diagnostic rather than the two-level router."},
        {"id": "estimator-chart-block", "type": "chart", "chartId": "estimator-auroc", "layout": "full"},
        {"id": "heldout-text", "type": "markdown", "sourceId": "benchmark-json", "body":
         "## Held-out ARC rejects the learned two-level policy\n\n"
         "Across 10%, 20%, 30%, and 50% calibration targets, uplift-minus-entropy accuracy differences are -0.67, +0.33, -0.33, and +1.34 percentage points. Every paired 95% interval includes zero. Actual dense-execution rates are reported because frozen calibration thresholds do not force exact held-out rates. Oracle uplift routing shows substantial headroom, so the failure is benefit identification rather than an absence of correctable examples."},
        {"id": "frontier-chart-block", "type": "chart", "chartId": "dynamic-frontier", "layout": "full"},
        {"id": "policy-table-block", "type": "table", "tableId": "policy-table", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "summary-json", "body":
         "## Scope, data, and metric definitions\n\nThe experiment uses Qwen2.5-1.5B-Instruct on ARC-Challenge: 750 training examples for calibration and one untouched 299-example validation evaluation. S0 is dense, S25 removes 25% of each FFN's intermediate channels, and S50 removes 50%, with S50 nested inside S25. Densification benefit is one when S50 is wrong and dense is correct. Sequential compute is the S50 transformer matrix-parameter proxy plus the observed rate of full dense re-execution."},
        {"id": "methods", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Methodology and experimental design\n\nChannel importance is final-prompt-token mean absolute FFN activation multiplied by the down-projection column norm. The same retained indices physically shrink gate, up, and down projections. Learned logistic probes use five-fold stratified out-of-fold prediction. Calibration discrimination uses AUROC, AUPRC, Precision@10/20/30%, and 2,000-replicate bootstrap intervals. The Phase 1 gate permits held-out evaluation when a cheap sparse-only estimator or uplift probe has AUROC at least 0.60 and exceeds the generic error probe. Held-out thresholds are numeric values frozen on calibration; paired correctness differences use 2,000 bootstrap replicates and exact McNemar diagnostics."},
        {"id": "limitations", "type": "markdown", "sourceId": "summary-json", "body":
         "## Limitations, uncertainty, and robustness\n\nThe fresh calibration dense pass differed from its cached BF16 reference by mean absolute probability 0.00287 and agreed on 98.53% of predictions; all within-run sparse comparisons use the fresh pass. ARC has only 299 held-out examples, and all learned-policy confidence intervals cross zero. Cross-sparsity disagreement is promising but requires an extra model level and was not authorized for held-out three-level evaluation after the two-level gate failed. Compute is an analytical transformer matrix-multiply parameter proxy excluding embeddings and the LM head; it does not establish latency, memory, energy, or throughput gains."},
        {"id": "next", "type": "markdown", "body":
         "## Recommended next steps\n\nStop this U-Dense branch for the current model and benchmark. Do not implement the three-level runtime, replicate on MMLU-Pro, or claim speedup. If revisiting adaptive capacity, first seek a router signal that predicts correction rather than generic error and validate it out-of-domain before system optimization."},
        {"id": "questions", "type": "markdown", "body":
         "## Further questions\n\nCan a representation-level probe predict dense correction without executing S25? Does training the sparse variants with structured-pruning recovery improve both the static frontier and the separability of correction events? Would a joint early-exit model expose cross-capacity disagreement at lower incremental cost?"},
    ]
    manifest = {
        "version": 1, "surface": "report", "title": summary["title"],
        "description": "Structured FFN pruning and uncertainty-guided densification on ARC-Challenge.",
        "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks,
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {
            "static": static, "estimators": estimators, "frontier": frontier,
            "policies": policies, "comparisons": comparisons,
        }},
        "sources": sources,
    }
    (DEST / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
