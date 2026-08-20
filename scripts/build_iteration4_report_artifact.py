"""Build the canonical portable technical report artifact for Iteration 4."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter4_arc")
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
    values = ",\n    ".join("(" + ", ".join(literal(row.get(column)) for column in columns) + ")" for row in rows)
    return {"id": source_id, "label": label, "query": {
        "engine": "SQLite", "language": "sql",
        "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
        "description": f"Materialize reviewed {label.lower()} rows.", "executed_at": generated,
        "filters": filters, "metric_definitions": definitions,
    }}


def scatter(chart_id, title, subtitle, dataset, source_id, x, x_label, y, y_label, color, color_label, max_rows):
    return {
        "id": chart_id, "title": title, "subtitle": subtitle, "showDescription": True,
        "intent": "relationship", "question": f"How does {x_label.lower()} relate to {y_label.lower()}?",
        "rationale": "A point-level relationship view preserves configuration variation without implying a temporal trend.",
        "comparisonContext": {"grain": "compression configuration", "unit": "proportion", "denominator": "299 ARC evaluation examples"},
        "type": "scatter", "dataset": dataset, "sourceId": source_id,
        "encodings": {
            "x": {"field": x, "type": "quantitative", "format": "percent", "label": x_label},
            "y": {"field": y, "type": "quantitative", "format": "percent", "label": y_label},
            "color": {"field": color, "type": "nominal", "label": color_label},
            "label": {"field": "name", "type": "nominal", "label": "Configuration"},
            "tooltip": [
                {"field": "name", "type": "text", "label": "Configuration"},
                {"field": x, "type": "quantitative", "format": "percent", "label": x_label},
                {"field": y, "type": "quantitative", "format": "percent", "label": y_label},
                {"field": "actual_average_bits", "type": "quantitative", "format": "number", "label": "Average bits"},
            ],
        },
        "palette": {"kind": "categorical", "name": chart_id}, "legend": {"position": "bottom", "title": color_label},
        "layout": "full", "maxRows": max_rows, "surface": {"surface": "explorer", "viewMode": "both"},
    }


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for source_path in (ROOT / "summary.json", ROOT / "policies.json", ROOT / "benchmark/results.json"):
        target = "benchmark_results.json" if source_path.name == "results.json" else source_path.name
        shutil.copyfile(source_path, DEST / target)

    configs = summary["configuration_rows"]
    key_configs = [row for row in configs if row["family"] in {"fp", "uniform", "kl", "cuaq"}]
    frontier = []
    for row in summary["outcome_rows"]:
        frontier.append({**row, "actual_average_bits": next(item["actual_average_bits"] for item in configs if item["name"] == row["name"])})
    q4_sensitivity = [{**row, "name": f"L{row['layer']}", "actual_average_bits": 4} for row in summary["sensitivity_rows"] if row["bit"] == 4]
    allocation = [{**row, "series": f"{row['method']} / {row['budget']}-bit budget", "name": row["layer_name"], "actual_average_bits": row["budget"]} for row in summary["allocation_rows"]]
    outcome_long = []
    for row in summary["outcome_rows"]:
        if row["target_rate"] != 0.3:
            continue
        for metric, field in (
            ("Fallback correction", "fallback_correction_rate"),
            ("Harmful escalation", "harmful_escalation_rate"),
            ("Missed beneficial delegation", "missed_beneficial_delegation_rate"),
        ):
            outcome_long.append({"name": row["name"], "metric": metric, "rate": row[field], "actual_average_bits": next(item["actual_average_bits"] for item in configs if item["name"] == row["name"])})

    sources = [
        {"id": "summary-json", "label": "Reviewed Iteration 4 summary", "path": "summary.json"},
        {"id": "policies-json", "label": "Calibration-only sensitivity and policy analysis", "path": "policies.json"},
        {"id": "benchmark-json", "label": "Held-out ARC cascade benchmark", "path": "benchmark_results.json"},
        query_source("config-query", "Configuration-level performance", configs,
            ["name", "family", "budget", "actual_average_bits", "standalone_accuracy", "mean_cascade_accuracy", "mean_delegation_rate", "mean_cascade_utility", "prediction_agreement_with_fp", "output_kl_from_fp", "uncertainty_spearman_with_fp", "compression_ratio_vs_fp16"], generated,
            ["ARC-Challenge validation", "299 examples", "five cost coefficients", "four FP-calibrated thresholds"],
            ["mean cascade utility averages accuracy minus lambda times delegation rate over 20 rate/cost conditions"]),
        query_source("frontier-query", "Key cascade frontiers", frontier,
            ["name", "target_rate", "cascade_accuracy", "delegation_rate", "cascade_utility", "fallback_correction_rate", "harmful_escalation_rate", "missed_beneficial_delegation_rate", "actual_average_bits"], generated,
            ["FP, uniform Q4, KL, and CUAQ", "lambda=0.025", "frozen FP thresholds"],
            ["cascade output uses the 7B fallback only when the compressed SLM score exceeds the frozen threshold"]),
        query_source("comparison-query", "Paired utility differences", summary["comparison_rows"],
            ["budget", "comparison", "difference", "ci_low", "ci_high", "allocation_hamming_kl_vs_cuaq"], generated,
            ["2,000 paired bootstrap replicates", "same 299 evaluation examples"], ["difference averages per-example utility over four rates and five cost coefficients"]),
        query_source("allocation-query", "KL and CUAQ layer allocations", allocation,
            ["budget", "method", "series", "layer", "layer_name", "bits", "name", "actual_average_bits"], generated,
            ["28 Qwen2.5-1.5B transformer blocks", "exact parameter-weighted budget"], ["bits is the assigned fake-quantization width for one block"]),
        query_source("sensitivity-query", "Four-bit local sensitivity relationship", q4_sensitivity,
            ["bit", "layer", "name", "kl_sensitivity", "cuaq_sensitivity", "uq_sensitivity", "dbaq_sensitivity", "actual_average_bits"], generated,
            ["750 ARC calibration examples", "one-block 4-bit interventions"], ["KL is mean output divergence", "CUAQ is mean FP-minus-probe cascade utility over frozen thresholds and costs"]),
        query_source("outcome-query", "Thirty-percent-target cascade outcomes", outcome_long,
            ["name", "metric", "rate", "actual_average_bits"], generated,
            ["target delegation rate 30% on FP calibration", "lambda-independent outcome metrics"], ["fallback correction is conditional on delegated examples", "harmful and missed-benefit rates use all 299 examples"]),
    ]

    charts = [
        scatter("accuracy-utility", "Standalone accuracy and cascade utility", "Twenty-five compressed configurations plus FP; utility averages four thresholds and five cost coefficients.", "configs", "config-query", "standalone_accuracy", "Standalone SLM accuracy", "mean_cascade_utility", "Mean cascade utility", "family", "Policy family", len(configs)),
        scatter("cascade-frontier", "Cascade accuracy and fallback-call rate", "FP, uniform Q4, KL, and CUAQ across four frozen-threshold operating points.", "frontier", "frontier-query", "delegation_rate", "Delegation rate", "cascade_accuracy", "Cascade accuracy", "name", "Configuration", len(frontier)),
        {
            "id": "allocation", "title": "Layer bit allocations", "subtitle": "KL and CUAQ at exactly matched 3.5- and 4.0-bit block-weight budgets.", "showDescription": True,
            "intent": "trend", "question": "Which transformer blocks receive different precision?", "rationale": "An ordered layer view preserves model depth and exposes policy changes.",
            "comparisonContext": {"grain": "transformer block", "unit": "bits per weight", "denominator": "28 blocks"}, "type": "line", "dataset": "allocation", "sourceId": "allocation-query",
            "encodings": {"x": {"field": "layer", "type": "quantitative", "format": "number", "label": "Block index"}, "y": {"field": "bits", "type": "quantitative", "format": "number", "label": "Assigned bits"}, "color": {"field": "series", "type": "nominal", "label": "Policy / budget"}, "tooltip": [{"field": "layer", "type": "quantitative", "format": "number", "label": "Block"}, {"field": "series", "type": "text", "label": "Policy"}, {"field": "bits", "type": "quantitative", "format": "number", "label": "Bits"}]},
            "palette": {"kind": "categorical", "name": "allocation"}, "settings": {"showPoints": True, "curve": "linear", "sort": "xAsc"}, "legend": {"position": "bottom", "title": "Policy / budget"}, "layout": "full", "maxRows": len(allocation), "surface": {"surface": "explorer", "viewMode": "both"},
        },
        scatter("sensitivity", "KL and CUAQ local sensitivity at four bits", "One point per transformer block on the 750-example calibration pool.", "sensitivity", "sensitivity-query", "kl_sensitivity", "KL sensitivity", "cuaq_sensitivity", "CUAQ sensitivity", "bit", "Probe bits", len(q4_sensitivity)),
        {
            "id": "outcomes", "title": "Cascade outcome components at the 30% target", "subtitle": "FP-calibrated threshold transferred to each configuration; 299 evaluation examples.", "showDescription": True,
            "intent": "comparison", "question": "Where do system gains and losses come from?", "rationale": "Grouped outcome bars compare correction, harm, and missed-benefit components at one fixed operating point.",
            "comparisonContext": {"grain": "configuration and outcome", "unit": "rate", "denominator": "delegated examples for correction; all examples otherwise"}, "type": "bar", "dataset": "outcomes", "sourceId": "outcome-query",
            "encodings": {"x": {"field": "name", "type": "nominal", "label": "Configuration"}, "y": {"field": "rate", "type": "quantitative", "format": "percent", "label": "Outcome rate"}, "color": {"field": "metric", "type": "nominal", "label": "Outcome"}, "tooltip": [{"field": "name", "type": "text", "label": "Configuration"}, {"field": "metric", "type": "text", "label": "Outcome"}, {"field": "rate", "type": "quantitative", "format": "percent", "label": "Rate"}]},
            "palette": {"kind": "categorical", "name": "outcomes"}, "legend": {"position": "bottom", "title": "Outcome"}, "layout": "full", "maxRows": len(outcome_long), "surface": {"surface": "explorer", "viewMode": "both"},
        },
    ]

    tables = [
        {"id": "comparisons", "title": "Paired held-out utility comparisons", "subtitle": "CUAQ minus each baseline; 95% intervals use 2,000 example-level bootstrap replicates.", "showDescription": True, "dataset": "comparisons", "sourceId": "comparison-query", "defaultSort": {"field": "budget", "direction": "asc"}, "density": "spacious", "layout": "full", "columns": [
            {"field": "budget", "label": "Budget", "format": "number"}, {"field": "comparison", "label": "Comparison", "type": "text"}, {"field": "difference", "label": "Utility difference", "format": "percent"}, {"field": "ci_low", "label": "CI low", "format": "percent"}, {"field": "ci_high", "label": "CI high", "format": "percent"}, {"field": "allocation_hamming_kl_vs_cuaq", "label": "KL/CUAQ changed blocks", "format": "number"}]},
        {"id": "configurations", "title": "Complete configuration summary", "subtitle": "Exact standalone, routing, distortion, and averaged system metrics for all evaluated configurations.", "showDescription": True, "dataset": "configs", "sourceId": "config-query", "defaultSort": {"field": "mean_cascade_utility", "direction": "desc"}, "density": "dense", "layout": "full", "columns": [
            {"field": "name", "label": "Configuration", "type": "text"}, {"field": "family", "label": "Family", "type": "text"}, {"field": "actual_average_bits", "label": "Avg bits", "format": "number"}, {"field": "standalone_accuracy", "label": "SLM accuracy", "format": "percent"}, {"field": "mean_cascade_accuracy", "label": "Mean cascade accuracy", "format": "percent"}, {"field": "mean_delegation_rate", "label": "Mean delegation", "format": "percent"}, {"field": "mean_cascade_utility", "label": "Mean utility", "format": "percent"}, {"field": "uncertainty_spearman_with_fp", "label": "FP rank rho", "format": "number"}]},
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Cascade-Aware Compression: Iteration 4"},
        {"id": "summary", "type": "markdown", "sourceId": "summary-json", "body": "## Technical summary: local CUAQ fails the ARC gate\n\n**Decision: NO-GO.** At 3.5 average bits, CUAQ trails KL-sensitive allocation by 0.53 utility points (95% CI -5.14 to +4.53) and is statistically indistinguishable from the random mean. At 4.0 bits, it trails KL by 13.31 points (-18.22 to -8.00), while its advantage over the random mean remains uncertain. The planned Stage-1 gate fails at both budgets, so DBAQ held-out evaluation, CARQ, MMLU-Pro, and packed quantization are not run."},
        {"id": "utility-text", "type": "markdown", "sourceId": "benchmark-json", "body": "## Standalone quality remains a strong guide to cascade utility\n\nAcross the 25 compressed configurations, standalone accuracy and mean cascade utility have Spearman rho 0.878. The relationship is not perfect, but it is strong enough that this experiment does not support replacing model-quality sensitivity with local cascade utility. CUAQ is slightly less accurate than KL at 3.5 bits and collapses from 42.8% to 23.4% standalone accuracy relative to KL/uniform Q4 at the 4-bit budget."},
        {"id": "utility-chart", "type": "chart", "chartId": "accuracy-utility", "layout": "full"},
        {"id": "frontier-text", "type": "markdown", "sourceId": "benchmark-json", "body": "## The cascade frontier does not rescue CUAQ\n\nThe relationship view plots actual fallback-call rate rather than the nominal calibration target because thresholds are frozen from FP calibration. At 3.5 bits, CUAQ and KL occupy broadly similar frontier regions. At 4 bits, KL coincides with uniform Q4 while CUAQ is worse across the operating range; changed delegation cannot offset its capability loss."},
        {"id": "frontier-chart", "type": "chart", "chartId": "cascade-frontier", "layout": "full"},
        {"id": "comparison-text", "type": "markdown", "sourceId": "benchmark-json", "body": "## Paired evidence rejects a 4-bit CUAQ gain\n\nCUAQ exceeds the ten-seed random mean by 2.25 points at 3.5 bits and 1.31 points at 4 bits, but both intervals cross zero. It fails to beat the best random allocation at either budget. The 4-bit loss against KL is significant at every operating rate by McNemar's test and in the aggregate paired bootstrap."},
        {"id": "comparison-table", "type": "table", "tableId": "comparisons", "layout": "full"},
        {"id": "allocation-text", "type": "markdown", "sourceId": "policies-json", "body": "## CUAQ changes the allocation, but not productively\n\nCUAQ differs from KL in 12 of 28 blocks at the 3.5-bit budget and six blocks at 4.0 bits. The negative result is therefore not caused by the methods collapsing to the same allocation. At 4 bits, KL selects uniform Q4, whereas CUAQ trades four Q2 blocks for two Q8 blocks; that locally favorable trade performs poorly as a whole-model configuration."},
        {"id": "allocation-chart", "type": "chart", "chartId": "allocation", "layout": "full"},
        {"id": "sensitivity-text", "type": "markdown", "sourceId": "policies-json", "body": "## Local cascade sensitivity is nearly orthogonal to KL at four bits\n\nThe layer-level 4-bit CUAQ and KL sensitivity rankings correlate at rho -0.089. This creates a genuinely different policy, but one-block cascade effects are not additive enough to predict the mixed-policy result. UQ and DBAQ proxy sensitivities remain closely aligned with KL (rho 0.974 and 0.971), consistent with earlier iterations."},
        {"id": "sensitivity-chart", "type": "chart", "chartId": "sensitivity", "layout": "full"},
        {"id": "outcome-text", "type": "markdown", "sourceId": "benchmark-json", "body": "## Delegation outcomes explain rather than reverse the loss\n\nAt the frozen 30% calibration threshold, the component view separates fallback corrections, harmful escalations, and missed beneficial delegation. CUAQ's 4-bit capability degradation creates more errors for the router to manage; its altered allocation does not convert enough of those errors into profitable fallback corrections."},
        {"id": "outcome-chart", "type": "chart", "chartId": "outcomes", "layout": "full"},
        {"id": "config-text", "type": "markdown", "sourceId": "summary-json", "body": "## Scope and metric definitions\n\nThe study uses all 750 ARC-Challenge training examples for calibration and the untouched 299-example validation split for evaluation. Qwen2.5-1.5B is the SLM and cached Qwen2.5-7B outputs are the fallback. Eighty-four one-block probes cover 28 transformer blocks at 2, 4, and 8 bits. Utility is cascade accuracy minus lambda times the actual fallback-call rate, averaged over lambda values 0, 0.01, 0.025, 0.05, and 0.1 and FP target rates 10%, 20%, 30%, and 50%."},
        {"id": "config-table", "type": "table", "tableId": "configurations", "layout": "full"},
        {"id": "methods", "type": "markdown", "sourceId": "policies-json", "body": "## Experimental design\n\nKL sensitivity is mean output KL between FP and a one-block probe. CUAQ sensitivity is the loss of end-to-end cascade utility under the same one-block intervention. Both allocate from {2,4,8} bits with a multiple-choice knapsack that spends the same maximum attainable parameter-weighted budget; each is compared with ten exact-budget random allocations. Margin uncertainty and FP-calibrated thresholds are frozen across methods. All policies are selected before held-out inference, and paired comparisons resample the same 299 examples 2,000 times."},
        {"id": "limitations", "type": "markdown", "sourceId": "summary-json", "body": "## Limitations and robustness\n\nThis is a decisive Stage-1 result for one model pair, one benchmark, controlled fake quantization, and a layer-local additive allocator. It does not test context-aware CUAQ, a trained router, non-additive search, packed kernels, latency, memory, or throughput. Those omissions are intentional consequences of the preregistered gate, not missing positive-result evidence. The 3.5-bit CUAQ-versus-KL interval is wide enough that small gains or losses remain unresolved; the 4-bit harm is clear."},
        {"id": "next", "type": "markdown", "body": "## Recommended next step\n\nStop cascade-aware precision allocation for this setup. Do not spend compute on DBAQ held-out evaluation, CARQ, MMLU-Pro replication, external MPQ ports, or AWQ/GPTQ validation. Retain Iteration 3's delegation-stability audit as the strongest result and treat post-quantization router recalibration as the practical intervention."},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\nWould a non-additive search succeed if initialized from KL without relying on one-block CUAQ sensitivities? Does cascade-aware compression become identifiable when the fallback advantage is much larger? Can a separate lightweight router capture deployment utility more reliably than perturbing the SLM's precision allocation? These are new directions, not continuations authorized by the failed Stage-1 gate."},
    ]

    manifest = {"version": 1, "surface": "report", "title": summary["title"], "description": "Stage-1 test of cascade-utility-aware mixed-precision quantization.", "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks}
    artifact = {"surface": "report", "manifest": manifest, "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {"configs": configs, "frontier": frontier, "comparisons": summary["comparison_rows"], "allocation": allocation, "sensitivity": q4_sensitivity, "outcomes": outcome_long}}, "sources": sources}
    (DEST / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
