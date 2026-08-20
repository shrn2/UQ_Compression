"""Build the canonical portable technical report artifact for Iteration 7."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter7_arc")
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
        "executed_at": generated, "filters": filters,
        "metric_definitions": definitions,
    }}


def scatter(chart_id, title, subtitle, dataset, source_id, x, x_label, y, y_label, color, rows, denominator):
    return {
        "id": chart_id, "title": title, "subtitle": subtitle, "showDescription": True,
        "intent": "relationship", "question": f"How does {x_label.lower()} relate to {y_label.lower()}?",
        "rationale": "A point-level relationship view preserves measured operating points and avoids implying continuous measurements.",
        "comparisonContext": {"grain": "method and operating point", "unit": "proportion", "denominator": denominator},
        "type": "scatter", "dataset": dataset, "sourceId": source_id,
        "encodings": {
            "x": {"field": x, "type": "quantitative", "format": "number", "label": x_label},
            "y": {"field": y, "type": "quantitative", "format": "percent", "label": y_label},
            "color": {"field": color, "type": "nominal", "label": "Method"},
            "label": {"field": color, "type": "nominal", "label": "Method"},
            "tooltip": [
                {"field": color, "type": "text", "label": "Method"},
                {"field": "target_rate", "type": "quantitative", "format": "percent", "label": "Dense rate"},
                {"field": x, "type": "quantitative", "format": "number", "label": x_label},
                {"field": y, "type": "quantitative", "format": "percent", "label": y_label},
            ],
        },
        "palette": {"kind": "categorical", "name": chart_id},
        "legend": {"position": "bottom", "title": "Method"},
        "layout": "full", "maxRows": len(rows),
        "surface": {"surface": "explorer", "viewMode": "both"},
    }


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for source_path, target_name in (
        (ROOT / "summary.json", "summary.json"),
        (ROOT / "oracle_analysis.json", "oracle_analysis.json"),
        (ROOT / "development_analysis.json", "development_analysis.json"),
    ):
        shutil.copyfile(source_path, DEST / target_name)

    static = summary["static_rows"]
    oracle = summary["oracle_rows"]
    estimators = [{
        "estimator": row["estimator"], "deployability": row["deployability"],
        "auroc": row["auroc_positive_advantage"],
        "auprc": row["auprc_positive_advantage"],
        "loss_spearman": row["loss_advantage_spearman"],
        "pairwise_accuracy": row["signed_pairwise_accuracy"],
        "positive_folds": row["positive_fold_spearman_count"],
    } for row in summary["estimator_rows"]]
    frontier = summary["frontier_rows"]
    recovery = [
        row for row in frontier
        if row["method"] in {
            "random_mean_10_seeds", "entropy", "max_probability", "error_probe",
            "binary_uplift_probe", summary["gate_b"]["selected_method"],
            "actual_capacity_disagreement", "signed_advantage_oracle",
        }
    ]
    outcomes = summary["outcome_rows"]
    layers = summary["layer_rows"]
    sources = [
        {"id": "summary-json", "label": "Reviewed Iteration 7 summary", "path": "summary.json"},
        {"id": "oracle-json", "label": "ARC oracle sparsity selection", "path": "oracle_analysis.json"},
        {"id": "development-json", "label": "ARC cross-validated router development", "path": "development_analysis.json"},
        query_source("static-query", "static capacity hierarchy", static,
            ["variant", "accuracy", "normalized_linear_flops", "prediction_agreement_with_dense", "output_kl_from_dense", "ece", "error_auroc", "aurc"], generated,
            ["750 ARC training examples", "nested physical FFN channel pruning"],
            ["normalized linear FLOPs is a transformer matrix-parameter proxy excluding embeddings and LM head"]),
        query_source("oracle-query", "oracle densification frontiers", oracle,
            ["variant", "series", "target_rate", "expected_sequential_compute", "sparse_accuracy", "oracle_accuracy", "oracle_gain", "oracle_efficiency", "positive_events", "harmful_events"], generated,
            ["750 ARC training examples", "exact top-r ranking by signed correctness advantage"],
            ["oracle efficiency is oracle accuracy gain divided by incremental sequential compute"]),
        query_source("estimator-query", "capacity-advantage estimator diagnostics", estimators,
            ["estimator", "deployability", "auroc", "auprc", "loss_spearman", "pairwise_accuracy", "positive_folds"], generated,
            ["five-fold out-of-fold predictions", "181 positive signed-advantage events"],
            ["AUROC and AUPRC target dense-beneficial examples; Spearman targets log-loss advantage"]),
        query_source("frontier-query", "matched-compute routing frontiers", frontier,
            ["method", "series", "target_rate", "dense_execution_rate", "accuracy", "expected_sequential_compute", "net_advantage_capture", "oracle_recovery"], generated,
            ["exact top-r selection", "ARC train development only", "all learned predictions out of fold"],
            ["net advantage capture is mean signed correctness advantage among selected dense executions"]),
        query_source("recovery-query", "oracle recovery curves", recovery,
            ["method", "series", "target_rate", "accuracy", "net_advantage_capture", "oracle_recovery"], generated,
            ["exact matched dense rates", "ten-seed mean for random"],
            ["oracle recovery divides policy accuracy gain by signed-advantage oracle gain"]),
        query_source("outcome-query", "thirty-percent routing decomposition", outcomes,
            ["method", "outcome", "rate"], generated,
            ["30% matched dense execution", "750 ARC training examples"],
            ["beneficial, harmful, and neutral rates are conditional on densification; missed positive uses all examples"]),
        query_source("layer-query", "representation layer diagnostics", layers,
            ["layer", "target", "estimator", "auroc", "auprc", "loss_spearman", "positive_folds", "oracle_recovery_30"], generated,
            ["layers 7, 14, 21, and 28 plus concatenated representations", "ridge alpha 1.0"],
            ["all probe predictions are five-fold out of fold"]),
    ]

    charts = [
        scatter("oracle-headroom", "Oracle sparse-to-dense capacity frontiers", "S25, S40, and S50 at four exact densification rates on 750 ARC development examples.", "oracle", "oracle-query", "expected_sequential_compute", "Expected sequential compute proxy", "oracle_accuracy", "Oracle accuracy", "series", oracle, "750 ARC training examples"),
        {
            "id": "estimator-auroc", "title": "Positive-advantage discrimination by routing signal",
            "subtitle": "Five-fold out-of-fold ARC development predictions; actual cross-capacity signals are expensive diagnostics.", "showDescription": True,
            "intent": "comparison", "question": "Which score identifies examples where dense inference corrects S50?",
            "rationale": "A common-scale bar chart compares AUROC across deployable, diagnostic, and oracle scores.",
            "comparisonContext": {"grain": "routing estimator", "unit": "AUROC", "denominator": "750 examples; 181 positives"},
            "type": "bar", "dataset": "estimators", "sourceId": "estimator-query",
            "encodings": {
                "x": {"field": "estimator", "type": "nominal", "label": "Estimator"},
                "y": {"field": "auroc", "type": "quantitative", "format": "number", "label": "Positive-advantage AUROC"},
                "tooltip": [
                    {"field": "estimator", "type": "text", "label": "Estimator"},
                    {"field": "deployability", "type": "text", "label": "Signal class"},
                    {"field": "auroc", "type": "quantitative", "format": "number", "label": "AUROC"},
                    {"field": "auprc", "type": "quantitative", "format": "number", "label": "AUPRC"},
                ],
            },
            "palette": {"kind": "categorical", "name": "estimator-single"},
            "layout": "full", "maxRows": len(estimators), "surface": {"surface": "explorer", "viewMode": "both"},
        },
        scatter("accuracy-compute", "Matched-compute ARC routing frontiers", "Exact top-r selection; learned scores are out of fold and random is a ten-seed mean.", "frontier", "frontier-query", "expected_sequential_compute", "Expected sequential compute proxy", "accuracy", "Policy accuracy", "series", frontier, "750 ARC training examples"),
        {
            "id": "oracle-recovery", "title": "Oracle recovery across dense-execution rates",
            "subtitle": "Fraction of signed-advantage oracle gain captured at exact 10%, 20%, 30%, and 50% rates.", "showDescription": True,
            "intent": "trend", "question": "How much exploitable complementarity does each router capture as the budget grows?",
            "rationale": "An ordered line view connects the same metric across fixed routing budgets.",
            "comparisonContext": {"grain": "method and dense rate", "unit": "proportion", "denominator": "signed-advantage oracle gain"},
            "type": "line", "dataset": "recovery", "sourceId": "recovery-query",
            "encodings": {
                "x": {"field": "target_rate", "type": "quantitative", "format": "percent", "label": "Dense execution rate"},
                "y": {"field": "oracle_recovery", "type": "quantitative", "format": "percent", "label": "Oracle recovery"},
                "color": {"field": "series", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "series", "type": "text", "label": "Method"},
                    {"field": "target_rate", "type": "quantitative", "format": "percent", "label": "Dense rate"},
                    {"field": "oracle_recovery", "type": "quantitative", "format": "percent", "label": "Oracle recovery"},
                ],
            },
            "palette": {"kind": "categorical", "name": "recovery"}, "legend": {"position": "bottom", "title": "Method"},
            "settings": {"showPoints": True, "curve": "linear", "sort": "xAsc"},
            "layout": "full", "maxRows": len(recovery), "surface": {"surface": "explorer", "viewMode": "both"},
        },
        {
            "id": "outcomes", "title": "Routing outcomes at 30% dense execution",
            "subtitle": "Selected L14 probe, entropy, actual S50-S25 disagreement, and signed oracle on ARC development data.", "showDescription": True,
            "intent": "comparison", "question": "Which selected examples are beneficial, harmful, neutral, or missed?",
            "rationale": "Grouped bars expose why similar aggregate accuracy can arise from different selections.",
            "comparisonContext": {"grain": "method and outcome", "unit": "rate", "denominator": "selected examples except missed-positive rate"},
            "type": "bar", "dataset": "outcomes", "sourceId": "outcome-query",
            "encodings": {
                "x": {"field": "outcome", "type": "nominal", "label": "Outcome"},
                "y": {"field": "rate", "type": "quantitative", "format": "percent", "label": "Rate"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "method", "type": "text", "label": "Method"},
                    {"field": "outcome", "type": "text", "label": "Outcome"},
                    {"field": "rate", "type": "quantitative", "format": "percent", "label": "Rate"},
                ],
            },
            "palette": {"kind": "categorical", "name": "outcomes"}, "legend": {"position": "bottom", "title": "Method"},
            "layout": "full", "maxRows": len(outcomes), "surface": {"surface": "explorer", "viewMode": "both"},
        },
    ]
    tables = [
        {"id": "static-table", "title": "Static sparse hierarchy", "subtitle": "Physical nested pruning on the same 750 ARC development examples.", "showDescription": True,
         "dataset": "static", "sourceId": "static-query", "defaultSort": {"field": "normalized_linear_flops", "direction": "desc"}, "density": "spacious", "layout": "full",
         "columns": [
             {"field": "variant", "label": "Variant", "type": "text"},
             {"field": "accuracy", "label": "Accuracy", "format": "percent"},
             {"field": "normalized_linear_flops", "label": "Compute proxy", "format": "percent"},
             {"field": "prediction_agreement_with_dense", "label": "Dense agreement", "format": "percent"},
             {"field": "output_kl_from_dense", "label": "Output KL", "format": "number"},
         ]},
        {"id": "layer-table", "title": "Residual-layer probe diagnostics", "subtitle": "Signed, loss, and instability targets across four depths and two concatenated variants.", "showDescription": True,
         "dataset": "layers", "sourceId": "layer-query", "defaultSort": {"field": "auroc", "direction": "desc"}, "density": "dense", "layout": "full",
         "columns": [
             {"field": "layer", "label": "Representation", "type": "text"},
             {"field": "target", "label": "Target", "type": "text"},
             {"field": "auroc", "label": "AUROC", "format": "number"},
             {"field": "auprc", "label": "AUPRC", "format": "number"},
             {"field": "loss_spearman", "label": "Loss rho", "format": "number"},
             {"field": "positive_folds", "label": "Positive folds", "format": "number"},
             {"field": "oracle_recovery_30", "label": "OR@30%", "format": "percent"},
         ]},
    ]

    gate = summary["gate_b"]
    selected = gate["selected_method"]
    comparison30 = next(row for row in summary["selected_comparisons"] if row["target_rate"] == 0.3)
    actual_disagreement = next(row for row in estimators if row["estimator"] == "actual_capacity_disagreement")
    predicted_instability = max(
        (row for row in summary["all_estimator_rows"] if row["estimator"].startswith("predicted_instability_hidden")),
        key=lambda row: row["auroc_positive_advantage"],
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Counterfactual Capacity Routing: Iteration 7"},
        {"id": "summary", "type": "markdown", "sourceId": "summary-json", "body":
         "## Technical summary: complementarity is real, but cheap representation probes do not recover it\n\n"
         f"**Decision: NO-GO for the tested counterfactual-capacity routers.** Oracle selection chose S50 because it reaches 55.73% accuracy at a 30% dense rate with a 0.859 sequential-compute proxy, recovering 24.0 accuracy points over S50. Actual S50-S25 prediction disagreement remains strongly diagnostic (AUROC {actual_disagreement['auroc']:.3f}), but its best hidden-state predictor reaches only {predicted_instability['auroc_positive_advantage']:.3f}. The best deployable representation score, {selected}, has AUROC {gate['selected_auroc']:.3f} versus {gate['best_generic_auroc']:.3f} for ordinary uncertainty and captures {gate['selected_oracle_recovery_at_30']:.1%} of oracle gain at 30%. The development gate fails, so MMLU-Pro and runtime work were not run."},
        {"id": "oracle-text", "type": "markdown", "sourceId": "oracle-json", "body":
         "## S50 offers the most efficient oracle opportunity\n\nAll three sparse variants clear the 10-point oracle-headroom gate at 30% dense execution. S25 gains 12.8 points at total sequential compute 1.079, S40 gains 21.6 at 0.947, and S50 gains 24.0 at 0.859. The frozen selection rule therefore chooses S50; this is an oracle opportunity result, not a deployable performance claim."},
        {"id": "oracle-chart", "type": "chart", "chartId": "oracle-headroom", "layout": "full"},
        {"id": "static-table-block", "type": "table", "tableId": "static-table", "layout": "full"},
        {"id": "signal-text", "type": "markdown", "sourceId": "development-json", "body":
         "## Hidden states fail to transfer cross-capacity disagreement into a cheap signal\n\nActual S50-S25 prediction disagreement reaches AUROC 0.709, reproducing Iteration 6's strongest diagnostic. The best predicted-instability hidden probe is close to chance, and the best signed-advantage hidden probe reaches only 0.587. Max-probability uncertainty remains the strongest deployable discriminator at 0.632. The gap shows that observing two capacity levels is informative, but the tested single-pass residual states do not linearly encode that counterfactual."},
        {"id": "signal-chart", "type": "chart", "chartId": "estimator-auroc", "layout": "full"},
        {"id": "layer-table-block", "type": "table", "tableId": "layer-table", "layout": "full"},
        {"id": "frontier-text", "type": "markdown", "sourceId": "development-json", "body":
         f"## Matched-compute routing shows only a small, uncertain L14 advantage\n\nAt 30% exact dense execution, {selected} reaches 39.60% development accuracy versus 39.20% for entropy. The +0.40-point difference has a paired 95% interval from {comparison30['accuracy_difference_ci'][0]*100:.2f} to {comparison30['accuracy_difference_ci'][1]*100:.2f} points. Its net advantage capture is 0.080 versus 0.076 for the best uncertainty policy; this is too small and uncertain to justify a fresh benchmark."},
        {"id": "frontier-chart", "type": "chart", "chartId": "accuracy-compute", "layout": "full"},
        {"id": "recovery-text", "type": "markdown", "sourceId": "development-json", "body":
         "## Deployable routers recover less than one-third of 30% oracle gain\n\nThe selected L14 probe recovers 33.1% of the signed oracle's gain at a 30% dense budget. Entropy recovers 31.5%, while the actual cross-capacity disagreement diagnostic recovers more but requires another sparse-model execution. The oracle curve confirms substantial complementarity; the deployable curves confirm that the tested signals do not capture enough of it."},
        {"id": "recovery-chart", "type": "chart", "chartId": "oracle-recovery", "layout": "full"},
        {"id": "outcome-text", "type": "markdown", "sourceId": "development-json", "body":
         "## Harmful and neutral selections consume most of the routing budget\n\nAt 30%, even the selected L14 probe spends most dense executions on neutral cases and still selects harmful cases. The signed oracle prioritizes all beneficial cases before neutral or harmful ones. This decomposition explains why modest AUROC differences translate into limited net accuracy gain."},
        {"id": "outcome-chart", "type": "chart", "chartId": "outcomes", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "summary-json", "body":
         "## Scope, data, and metric definitions\n\nIteration 7 is a development-only study on the same 750 ARC-Challenge training examples used for Iteration 6 calibration. ARC validation was not reused. S25, S40, and S50 physically remove 25%, 40%, and 50% of every FFN's intermediate channels with nested masks. Signed advantage is dense correctness minus sparse correctness: +1 beneficial, 0 neutral, -1 harmful. Net advantage capture is the mean signed advantage among selected dense executions; oracle recovery divides policy gain by signed-oracle gain at the same exact dense rate."},
        {"id": "methods", "type": "markdown", "sourceId": "development-json", "body":
         "## Model specification and experimental design\n\nQwen2.5-1.5B-Instruct uses the unchanged activation-aware channel ordering from Iteration 6. Prompt-only final-token residual states were captured after blocks 7, 14, 21, and 28. Fixed-alpha ridge probes (alpha 1.0) predict signed correctness advantage, log-loss advantage, binary S50-S25 disagreement, and soft JS disagreement. Logistic output baselines predict sparse error and binary correction. Every learned score is five-fold stratified out of fold. Primary policies densify exactly the top 10%, 20%, 30%, or 50%; 2,000 paired bootstrap replicates quantify differences."},
        {"id": "limitations", "type": "markdown", "sourceId": "summary-json", "body":
         "## Limitations, uncertainty, and robustness\n\nThis is model and benchmark development evidence, not held-out generalization. Hidden features are linear-probe inputs without pruning recovery or task adaptation; a nonlinear or trained representation could behave differently. The fresh S50 probability rerun agrees with the hierarchy on 98.13% of predictions (MAE 0.00370), and within-probe targets use that fresh S50 pass. Actual cross-capacity disagreement is a diagnostic that requires executing S25 and is not a deployment-cost-equivalent baseline. Compute values are analytical transformer matrix-parameter proxies, not measured latency, memory, energy, or throughput."},
        {"id": "next", "type": "markdown", "body":
         "## Recommended next steps\n\nStop the current linear counterfactual-router branch. Do not run MMLU-Pro or runtime experiments under this iteration. If the broader research continues, change the source of the routing representation rather than tuning this probe: jointly train a small router during structured-pruning recovery, expose paired-capacity contrastive targets during training, or design an early-exit architecture where cross-capacity change is observable at low incremental cost."},
        {"id": "questions", "type": "markdown", "body":
         "## Further questions\n\nCan pruning-recovery training make signed capacity advantage linearly accessible without eliminating complementarity? Can a learned early-exit state approximate S50-S25 disagreement with materially less cost than executing S25? Does the oracle advantage persist on a genuinely untouched benchmark after the router representation itself changes?"},
    ]
    manifest = {
        "version": 1, "surface": "report", "title": summary["title"],
        "description": "Oracle headroom and hidden-state counterfactual capacity routing on ARC-Challenge.",
        "generatedAt": generated, "sources": sources, "charts": charts,
        "tables": tables, "blocks": blocks,
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {
            "static": static, "oracle": oracle, "estimators": estimators,
            "frontier": frontier, "recovery": recovery, "outcomes": outcomes,
            "layers": layers,
        }},
        "sources": sources,
    }
    (DEST / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    chart_map = [
        {"section": "Oracle opportunity", "question": "Which sparsity has the best oracle frontier?", "type": "scatter", "fields": ["expected_sequential_compute", "oracle_accuracy", "variant"], "claim": "S50 maximizes oracle efficiency."},
        {"section": "Signal prediction", "question": "Which score predicts positive advantage?", "type": "bar", "fields": ["estimator", "auroc"], "claim": "Actual disagreement is strong; predicted disagreement is weak."},
        {"section": "System frontier", "question": "Which router improves accuracy at matched compute?", "type": "scatter", "fields": ["expected_sequential_compute", "accuracy", "method"], "claim": "The L14 advantage over entropy is small and uncertain."},
        {"section": "Oracle recovery", "question": "How much complementarity is captured?", "type": "line", "fields": ["target_rate", "oracle_recovery", "method"], "claim": "Deployable routers recover less than one-third at 30%."},
        {"section": "Outcome decomposition", "question": "Where does the routing budget go?", "type": "grouped bar", "fields": ["outcome", "rate", "method"], "claim": "Neutral and harmful selections limit net gain."},
    ]
    (DEST / "chart_map.json").write_text(
        json.dumps(chart_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
