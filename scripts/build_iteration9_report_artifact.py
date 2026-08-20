"""Build the canonical portable technical report artifact for Iteration 9."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter9_mmlu_pro")
DEST = ROOT / "report"


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def query_source(source_id: str, label: str, rows: list[dict], columns: list[str], generated: str, filters: list[str], definitions: list[str]) -> dict:
    values = ",\n    ".join(
        "(" + ", ".join(literal(row.get(column)) for column in columns) + ")" for row in rows
    )
    return {"id": source_id, "label": label, "query": {
        "engine": "SQLite", "language": "sql",
        "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
        "description": f"Materialize reviewed {label.lower()} rows.", "executed_at": generated,
        "filters": filters, "metric_definitions": definitions,
    }}


def main() -> int:
    analysis = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
    manifest_source = json.loads((ROOT / "evaluation_manifest.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for path in ("analysis.json", "evaluation_manifest.json"):
        shutil.copyfile(ROOT / path, DEST / path)

    metrics = analysis["advantage_metrics"]
    metric_rows = [row for row in metrics if row["method"] not in {"positive_uplift_oracle", "signed_advantage_oracle"}]
    selected_methods = {"entropy", "max_probability", "ccr", "actual_capacity_disagreement", "actual_capacity_js", "signed_advantage_oracle"}
    frontiers = [row for row in analysis["policy_rows"] if row["method"] in selected_methods]
    frontiers.extend(analysis["random_summary"])
    frontiers.extend([
        {"method": "always_s50", "target_rate": 0.0, "accuracy": analysis["static"]["s50_accuracy"],
         "expected_sequential_compute": analysis["static"]["s50_compute_proxy"], "net_advantage_capture": 0.0, "oracle_recovery": 0.0},
        {"method": "always_dense", "target_rate": 1.0, "accuracy": analysis["static"]["dense_accuracy"],
         "expected_sequential_compute": 1.0, "net_advantage_capture": analysis["static"]["dense_accuracy"] - analysis["static"]["s50_accuracy"], "oracle_recovery": None},
    ])
    recovery = [row for row in frontiers if row["method"] in {"entropy", "max_probability", "ccr", "actual_capacity_js", "signed_advantage_oracle", "random_20_seed_mean"} and row.get("target_rate") in {0.1, 0.2, 0.3, 0.5}]
    outcomes = analysis["routing_outcomes_30"]
    thresholds = analysis["threshold_transfer"]
    comparisons = analysis["comparisons"]
    comparison_rows = [{
        "comparison": row["comparison"], "target_rate": row["target_rate"],
        "accuracy_difference": row["accuracy_difference"],
        "accuracy_ci_low": row["accuracy_difference_ci"][0], "accuracy_ci_high": row["accuracy_difference_ci"][1],
        "nac_difference": row["nac_difference"],
        "nac_ci_low": row["nac_difference_ci"][0], "nac_ci_high": row["nac_difference_ci"][1],
    } for row in comparisons]
    sources = [
        {"id": "analysis-json", "label": "Frozen CCR MMLU-Pro system analysis", "path": "analysis.json"},
        {"id": "cohort-json", "label": "Outcome-blind MMLU-Pro cohort manifest", "path": "evaluation_manifest.json"},
        {"id": "frozen-router", "label": "Pre-registered linear CCR weights", "path": "runs/iter9_mmlu_pro/frozen_ccr.npz"},
        query_source("metric-query", "positive-advantage ranking diagnostics", metric_rows,
            ["method", "deployability", "auroc", "auprc", "precision_at_10", "precision_at_20", "precision_at_30"], generated,
            ["1,000 fixed MMLU-Pro test examples", "212 positive dense-advantage events", "router frozen before outcome evaluation"],
            ["positive advantage means S50 wrong and dense correct", "AUROC and AUPRC use positive advantage as the binary target"]),
        query_source("frontier-query", "matched-compute routing frontiers", frontiers,
            ["method", "target_rate", "expected_sequential_compute", "accuracy", "net_advantage_capture", "oracle_recovery"], generated,
            ["exact top-r routing", "rates 10%, 20%, 30%, and 50%", "random is a 20-seed mean"],
            ["sequential compute equals 0.5588 plus dense execution rate", "compute is an analytical matrix proxy, not latency"]),
        query_source("recovery-query", "oracle recovery curves", recovery,
            ["method", "target_rate", "accuracy", "net_advantage_capture", "oracle_recovery", "expected_sequential_compute"], generated,
            ["exact top-r routing", "same 1,000 examples for every method"],
            ["oracle recovery divides policy net advantage by signed-oracle net advantage at the same rate"]),
        query_source("outcome-query", "thirty-percent routing outcomes", outcomes,
            ["method", "outcome", "selected_rate", "selected_count"], generated,
            ["exact 30% dense routing", "300 selected examples per method"],
            ["selected rate is conditional on the selected dense-execution subset"]),
        query_source("threshold-query", "CCR threshold transfer", thresholds,
            ["target_calibration_rate", "threshold", "actual_dense_rate", "accuracy", "expected_sequential_compute", "net_advantage_capture", "oracle_recovery", "beneficial_selected_rate", "harmful_selected_rate"], generated,
            ["threshold selected on the 600-example Iteration 8 held-out contrast split", "applied unchanged to MMLU-Pro"],
            ["threshold transfer is secondary to exact matched-rate ranking"]),
        query_source("comparison-query", "paired primary comparisons", comparison_rows,
            ["comparison", "target_rate", "accuracy_difference", "accuracy_ci_low", "accuracy_ci_high", "nac_difference", "nac_ci_low", "nac_ci_high"], generated,
            ["5,000 paired bootstrap replicates", "same sampled examples within each comparison"],
            ["differences are CCR minus the named uncertainty baseline"]),
    ]

    charts = [
        {"id": "advantage-auroc", "title": "Positive dense-advantage discrimination",
         "subtitle": "Frozen scores on 1,000 MMLU-Pro examples; 212 S50-wrong/dense-correct events.", "showDescription": True,
         "intent": "comparison", "question": "Which sparse-pass score identifies beneficial densification?",
         "rationale": "A common zero-based bar scale supports direct AUROC comparison across router classes.",
         "comparisonContext": {"grain": "routing score", "unit": "AUROC", "denominator": "1,000 examples; 212 positives"},
         "type": "bar", "dataset": "metrics", "sourceId": "metric-query",
         "encodings": {"x": {"field": "method", "type": "nominal", "label": "Score"},
                       "y": {"field": "auroc", "type": "quantitative", "format": "number", "label": "Positive-advantage AUROC"},
                       "tooltip": [{"field": "method", "type": "text", "label": "Score"},
                                   {"field": "deployability", "type": "text", "label": "Class"},
                                   {"field": "auroc", "type": "quantitative", "format": "number", "label": "AUROC"},
                                   {"field": "auprc", "type": "quantitative", "format": "number", "label": "AUPRC"}]},
         "palette": {"kind": "categorical", "name": "auroc-comparison"}, "layout": "full", "maxRows": len(metric_rows),
         "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "accuracy-compute", "title": "MMLU-Pro accuracy and sequential compute",
         "subtitle": "Exact matched dense rates; compute is a transformer matrix proxy and random is a 20-seed mean.", "showDescription": True,
         "intent": "relationship", "question": "Does CCR improve accuracy at the same analytical compute?",
         "rationale": "A point-level accuracy-compute view preserves every measured operating point without implying continuous measurements.",
         "comparisonContext": {"grain": "method and operating point", "unit": "proportion", "denominator": "1,000 examples"},
         "type": "scatter", "dataset": "frontiers", "sourceId": "frontier-query",
         "encodings": {"x": {"field": "expected_sequential_compute", "type": "quantitative", "format": "number", "label": "Expected sequential compute proxy"},
                       "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                       "color": {"field": "method", "type": "nominal", "label": "Method"},
                       "label": {"field": "method", "type": "nominal", "label": "Method"},
                       "tooltip": [{"field": "method", "type": "text", "label": "Method"},
                                   {"field": "target_rate", "type": "quantitative", "format": "percent", "label": "Dense rate"},
                                   {"field": "expected_sequential_compute", "type": "quantitative", "format": "number", "label": "Compute proxy"},
                                   {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"}]},
         "palette": {"kind": "categorical", "name": "frontier"}, "legend": {"position": "bottom", "title": "Method"},
         "layout": "full", "maxRows": len(frontiers), "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "oracle-recovery", "title": "Oracle recovery across dense-execution rates",
         "subtitle": "Fraction of signed-advantage oracle gain captured at exact 10%, 20%, 30%, and 50% rates.", "showDescription": True,
         "intent": "trend", "question": "How much adaptive headroom does each ranking recover as budget grows?",
         "rationale": "An ordered line chart compares the same recovery measure across four fixed rates.",
         "comparisonContext": {"grain": "method and exact dense rate", "unit": "proportion", "denominator": "signed-oracle gain"},
         "type": "line", "dataset": "recovery", "sourceId": "recovery-query",
         "encodings": {"x": {"field": "target_rate", "type": "quantitative", "format": "percent", "label": "Dense execution rate"},
                       "y": {"field": "oracle_recovery", "type": "quantitative", "format": "percent", "label": "Oracle recovery"},
                       "color": {"field": "method", "type": "nominal", "label": "Method"},
                       "tooltip": [{"field": "method", "type": "text", "label": "Method"},
                                   {"field": "target_rate", "type": "quantitative", "format": "percent", "label": "Dense rate"},
                                   {"field": "oracle_recovery", "type": "quantitative", "format": "percent", "label": "Oracle recovery"}]},
         "palette": {"kind": "categorical", "name": "recovery"}, "legend": {"position": "bottom", "title": "Method"},
         "settings": {"showPoints": True, "curve": "linear", "sort": "xAsc"}, "layout": "full", "maxRows": len(recovery),
         "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "outcomes", "title": "Routing outcome composition at 30% dense execution",
         "subtitle": "Each method selects exactly 300 of the same 1,000 MMLU-Pro examples.", "showDescription": True,
         "intent": "composition", "question": "What kinds of examples consume each router's 30% dense budget?",
         "rationale": "Grouped bars expose beneficial, harmful, and two neutral mechanisms at a shared denominator.",
         "comparisonContext": {"grain": "method and outcome", "unit": "selected-subset share", "denominator": "300 selected examples"},
         "type": "bar", "dataset": "outcomes", "sourceId": "outcome-query",
         "encodings": {"x": {"field": "outcome", "type": "nominal", "label": "Outcome"},
                       "y": {"field": "selected_rate", "type": "quantitative", "format": "percent", "label": "Share of selected examples"},
                       "color": {"field": "method", "type": "nominal", "label": "Method"},
                       "tooltip": [{"field": "method", "type": "text", "label": "Method"},
                                   {"field": "outcome", "type": "text", "label": "Outcome"},
                                   {"field": "selected_count", "type": "quantitative", "format": "number", "label": "Count"},
                                   {"field": "selected_rate", "type": "quantitative", "format": "percent", "label": "Selected share"}]},
         "palette": {"kind": "categorical", "name": "outcomes"}, "legend": {"position": "bottom", "title": "Method"},
         "layout": "full", "maxRows": len(outcomes), "surface": {"surface": "explorer", "viewMode": "both"}},
    ]
    tables = [{"id": "threshold-table", "title": "Frozen-threshold transfer",
        "subtitle": "Thresholds target 20% and 30% on the 600-example contrast validation split, then transfer unchanged.",
        "showDescription": True, "dataset": "thresholds", "sourceId": "threshold-query",
        "defaultSort": {"field": "target_calibration_rate", "direction": "asc"}, "density": "spacious", "layout": "full",
        "columns": [
            {"field": "target_calibration_rate", "label": "Calibration target", "format": "percent"},
            {"field": "actual_dense_rate", "label": "Evaluation dense rate", "format": "percent"},
            {"field": "accuracy", "label": "Accuracy", "format": "percent"},
            {"field": "expected_sequential_compute", "label": "Compute proxy", "format": "number"},
            {"field": "oracle_recovery", "label": "Oracle recovery", "format": "percent"},
        ]}]

    gate = analysis["gate"]
    metric_lookup = {row["method"]: row for row in metrics}
    policy_lookup = {(row["method"], row["target_rate"]): row for row in analysis["policy_rows"]}
    ccr20, ccr30 = policy_lookup[("ccr", 0.2)], policy_lookup[("ccr", 0.3)]
    max30 = policy_lookup[("max_probability", 0.3)]
    entropy20 = policy_lookup[("entropy", 0.2)]
    comp30 = next(row for row in comparisons if row["comparison"] == "ccr_minus_max_probability" and row["target_rate"] == 0.3)
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Capacity-Contrast Routing: Iteration 9"},
        {"id": "summary", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Technical summary: CCR predicts benefit slightly better but routes no better than uncertainty\n\n"
         f"**Decision: {gate['decision']}.** The frozen CCR router reaches positive-advantage AUROC {metric_lookup['ccr']['auroc']:.4f}, narrowly above entropy at {metric_lookup['entropy']['auroc']:.4f}, but its AUPRC is lower ({metric_lookup['ccr']['auprc']:.4f} versus {metric_lookup['entropy']['auprc']:.4f}). At exact 20% dense execution, CCR reaches {ccr20['accuracy']:.1%} accuracy versus {entropy20['accuracy']:.1%} for entropy. At 30%, CCR reaches {ccr30['accuracy']:.1%} versus {max30['accuracy']:.1%} for max-probability uncertainty, a {comp30['accuracy_difference']*100:+.1f}-point difference with 95% paired interval [{comp30['accuracy_difference_ci'][0]*100:.1f}, {comp30['accuracy_difference_ci'][1]*100:.1f}]. The primary system gate fails, so ablations, S40, and runtime work are not run."},
        {"id": "ranking-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## The small AUROC lead does not improve top-budget precision\n\n"
         "CCR is the strongest deployable AUROC score by a small margin, but entropy has higher AUPRC and higher Precision@10%, @20%, and @30%. Actual soft S50-S25 JS divergence is a stronger diagnostic, yet it requires the privileged S25 execution and is not deployment-equivalent. The ranking result is descriptive evidence of weak transfer, not evidence of a useful routing policy."},
        {"id": "ranking-chart", "type": "chart", "chartId": "advantage-auroc", "layout": "full"},
        {"id": "frontier-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## CCR trails uncertainty at both primary matched-compute points\n\n"
         "S50 accuracy is 12.4% and dense accuracy is 26.7%. CCR raises accuracy by routing some examples to dense, but it reaches only 16.3% at 20% dense execution and 17.7% at 30%. The best uncertainty baselines reach 16.7% and 18.3% at the same exact budgets. At 50%, CCR is 0.3 points higher than uncertainty, but the sequential proxy is 1.059—above always-dense compute—and accuracy remains 5.3 points below always dense, so this is not a meaningful system win."},
        {"id": "frontier-chart", "type": "chart", "chartId": "accuracy-compute", "layout": "full"},
        {"id": "recovery-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## CCR recovers less oracle headroom than uncertainty at 20% and 30%\n\n"
         "The signed oracle shows large capacity complementarity: it reaches 32.4% at 20% dense execution and 33.6% at 30%. CCR recovers 19.5% and 25.0% of that gain, below the best uncertainty recovery of 21.5% and 27.8%. The gap is small but directionally inconsistent with the method hypothesis."},
        {"id": "recovery-chart", "type": "chart", "chartId": "oracle-recovery", "layout": "full"},
        {"id": "outcome-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Neutral and harmful selections erase CCR's potential gain\n\n"
         "At 30%, CCR selects 74 beneficial and 21 harmful examples; max-probability uncertainty selects 73 beneficial but only 14 harmful examples. More than 60% of every selected subset is neutral and wrong under both models. CCR's slightly better global discrimination therefore does not translate into a better top-30% signed-advantage ranking."},
        {"id": "outcome-chart", "type": "chart", "chartId": "outcomes", "layout": "full"},
        {"id": "threshold-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Frozen thresholds under-route after domain transfer\n\n"
         "Thresholds calibrated to 20% and 30% dense execution on MMLU auxiliary validation transfer to only 13.9% and 22.7% on MMLU-Pro. This calibration drift reinforces that exact matched-rate ranking is the appropriate primary comparison and that raw CCR probabilities are not portable thresholds without recalibration."},
        {"id": "threshold-table-block", "type": "table", "tableId": "threshold-table", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "cohort-json", "body":
         f"## Scope, data, and metric definitions\n\nThe evaluation uses {manifest_source['final_evaluation_size']:,} deterministic category-stratified MMLU-Pro test examples selected before this iteration's model inference. All {manifest_source['evaluation_questions_checked']:,} questions were normalized and checked against 3,000 MMLU auxiliary training prompts; no exact or trivial-near duplicates were removed. Positive advantage means S50 is wrong and dense is correct; harmful advantage is the reverse. Net Advantage Capture is mean signed advantage over all examples selected for dense execution. Oracle recovery divides policy gain by the signed-oracle gain at the same exact rate."},
        {"id": "methods", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Frozen model specification and experimental design\n\n"
         "Qwen2.5-1.5B-Instruct uses the physical nested S50 and S25 channel masks from Iterations 6–8. CCR is the Iteration 8 regularized linear logistic router using S50 h14 plus entropy, margin uncertainty, max-probability uncertainty, and logit gap. Its weights were exported before evaluation outcomes were read and reproduce the Iteration 8 held-out AUROC exactly. S50, dense, and diagnostic S25 were loaded sequentially. Policies route the exact top 10%, 20%, 30%, or 50%. Primary 20% and 30% comparisons use 5,000 paired bootstrap samples and exact McNemar tests."},
        {"id": "limitations", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Limitations, uncertainty, and robustness checks\n\n"
         "This is one 1,000-example sample from one benchmark and one model family. The MMLU-Pro cohort was used in Iteration 3 for a different compression experiment, but it was not used to develop, select, or tune CCR; the confirmatory claim is therefore fresh relative to the CCR loop, not globally untouched. The generic error probe is trained on a separate 600-example labeled MMLU auxiliary calibration subset and transfers poorly. Actual S50-S25 disagreement and JS are expensive diagnostics. Compute values are analytical transformer matrix-parameter proxies, not measured latency, memory, energy, or throughput."},
        {"id": "next", "type": "markdown", "body":
         "## Recommended next steps\n\nStop the current cross-capacity-disagreement routing branch. Do not add a larger router, run the planned data/feature ablations, test S40, or claim runtime benefits. If adaptive sparsity work continues, change the supervision objective from prediction change to signed capacity value using a clean labeled calibration design, or redesign execution so the routing signal is causally closer to the restored computation."},
        {"id": "questions", "type": "markdown", "body":
         "## Further questions\n\nWhy does soft cross-capacity divergence align with dense benefit better than hard disagreement while remaining expensive? Can a label-efficient signed-advantage target preserve broad supervision without conflating uncertainty with usefulness? Which examples are neutral-wrong under both S50 and dense, and can they be rejected rather than densified?"},
    ]
    manifest = {"version": 1, "surface": "report", "title": "Capacity-Contrast Routing: Iteration 9",
        "description": "Frozen CCR system evaluation on MMLU-Pro at exact matched dense rates.",
        "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks}
    artifact = {"surface": "report", "manifest": manifest,
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {
            "metrics": metric_rows, "frontiers": frontiers, "recovery": recovery,
            "outcomes": outcomes, "thresholds": thresholds, "comparisons": comparison_rows,
        }}, "sources": sources}
    (DEST / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    chart_map = [
        {"section": "Benefit ranking", "question": "Which score predicts positive advantage?", "type": "bar", "fields": ["method", "auroc", "auprc"], "claim": "CCR has a marginal AUROC lead but lower top-budget precision."},
        {"section": "System frontier", "question": "Does CCR improve accuracy at matched compute?", "type": "scatter", "fields": ["expected_sequential_compute", "accuracy", "method"], "claim": "CCR trails uncertainty at 20% and 30%."},
        {"section": "Oracle recovery", "question": "How much adaptive headroom is captured?", "type": "line", "fields": ["target_rate", "oracle_recovery", "method"], "claim": "CCR recovers less than uncertainty at primary rates."},
        {"section": "Outcome mechanism", "question": "What consumes the 30% budget?", "type": "grouped bar", "fields": ["outcome", "selected_rate", "method"], "claim": "Neutral and harmful routes erase the global AUROC lead."},
    ]
    (DEST / "chart_map.json").write_text(json.dumps(chart_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
