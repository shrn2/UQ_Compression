"""Build the canonical portable technical report artifact for Iteration 5."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter5_arc")
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


def scatter(chart_id, title, subtitle, dataset, source_id, x, x_label, y, y_label, color, color_label, rows, denominator):
    x_format = "percent" if x in {"standalone_accuracy", "delegation_rate"} else "number"
    return {
        "id": chart_id, "title": title, "subtitle": subtitle, "showDescription": True,
        "intent": "relationship", "question": f"How does {x_label.lower()} relate to {y_label.lower()}?",
        "rationale": "A point-level relationship view shows candidate spread without implying a trend.",
        "comparisonContext": {"grain": "compression configuration", "unit": "proportion", "denominator": denominator},
        "type": "scatter", "dataset": dataset, "sourceId": source_id,
        "encodings": {
            "x": {"field": x, "type": "quantitative", "format": x_format, "label": x_label},
            "y": {"field": y, "type": "quantitative", "format": "percent", "label": y_label},
            "color": {"field": color, "type": "nominal", "label": color_label},
            "label": {"field": "name", "type": "nominal", "label": "Configuration"},
            "tooltip": [
                {"field": "name", "type": "text", "label": "Configuration"},
                {"field": x, "type": "quantitative", "format": x_format, "label": x_label},
                {"field": y, "type": "quantitative", "format": "percent", "label": y_label},
                {"field": color, "type": "text", "label": color_label},
            ],
        },
        "palette": {"kind": "categorical", "name": chart_id}, "legend": {"position": "bottom", "title": color_label},
        "layout": "full", "maxRows": len(rows), "surface": {"surface": "explorer", "viewMode": "both"},
    }


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for source_path, target_name in ((ROOT / "summary.json", "summary.json"), (ROOT / "search/results.json", "search_results.json"), (ROOT / "benchmark/results.json", "benchmark_results.json")):
        shutil.copyfile(source_path, DEST / target_name)

    candidates = summary["candidate_rows"]
    exact = summary["exact_rows"]
    heldout = summary["heldout_rows"]
    frontier = summary["frontier_rows"]
    allocation = summary["allocation_rows"]
    outcomes = summary["outcome_rows"]
    comparison = summary["heldout_comparison"]["c3q_vs_kl_mean_utility"]

    sources = [
        {"id": "summary-json", "label": "Reviewed Iteration 5 summary", "path": "summary.json"},
        {"id": "search-json", "label": "Calibration candidate search", "path": "search_results.json"},
        {"id": "benchmark-json", "label": "Held-out ARC benchmark", "path": "benchmark_results.json"},
        query_source("candidate-query", "screened candidate cloud", candidates,
            ["name", "budget", "family", "changed_layers", "generation_seed", "screening_safe", "screening_accuracy", "screening_kl", "screening_utility", "exact_finalist", "exact_safe", "status"], generated,
            ["100 exact-budget configurations per budget", "additive layer-probe screening surrogate", "750 ARC calibration rows in each cached probe"],
            ["screening utility averages cascade accuracy minus lambda times delegation rate over four rates and five lambda values"]),
        query_source("exact-query", "exact finalist metrics", exact,
            ["name", "budget", "family", "selected", "safe", "standalone_accuracy", "output_kl", "mean_utility", "agreement_with_fp", "error_auroc", "aurc"], generated,
            ["six frozen finalists per budget", "whole-model fake quantization", "750 ARC calibration examples"],
            ["joint safe means output KL no more than 1.10 times KL baseline and calibration accuracy no more than 1 pp below baseline"]),
        query_source("heldout-query", "held-out configuration metrics", heldout,
            ["name", "family", "budget", "standalone_accuracy", "mean_utility", "output_kl", "agreement_with_fp", "uncertainty_spearman", "error_auroc"], generated,
            ["ARC validation", "299 examples", "3.5-bit budget authorized by calibration headroom"],
            ["mean utility averages per-example cascade correctness minus lambda times delegation over four frozen thresholds and five lambda values"]),
        query_source("frontier-query", "held-out cascade frontiers", frontier,
            ["name", "family", "target_rate", "delegation_rate", "cascade_accuracy"], generated,
            ["candidate-specific calibration thresholds frozen before held-out inference", "299 ARC validation examples"],
            ["delegation rate is actual held-out fallback-call frequency"]),
        query_source("allocation-query", "KL and C3Q allocations", allocation,
            ["budget", "method", "series", "layer", "bits"], generated,
            ["28 Qwen2.5-1.5B transformer blocks", "exact parameter-weighted bit budget"],
            ["bits is the assigned symmetric fake-quantization width"]),
        query_source("outcome-query", "thirty-percent delegation outcomes", outcomes,
            ["name", "outcome", "rate"], generated,
            ["held-out ARC", "30% calibration target", "KL and C3Q only"],
            ["fallback correction is conditional on delegated examples; harmful and missed-benefit rates use all examples"]),
    ]

    charts = [
        scatter("candidate-cloud", "Capability-screened candidate cloud", "Two hundred exact-budget allocations; surrogate metrics select frozen exact finalists.", "candidates", "candidate-query", "screening_kl", "Surrogate output KL", "screening_utility", "Surrogate mean utility", "status", "Candidate status", candidates, "750 ARC calibration examples per probe"),
        scatter("exact-cloud", "Exact capability and utility among finalists", "Whole-model scores for six frozen finalists per budget; color shows selection role.", "exact", "exact-query", "standalone_accuracy", "Standalone accuracy", "mean_utility", "Mean cascade utility", "selected", "Selection role", exact, "750 ARC calibration examples"),
        scatter("frontier", "Held-out cascade accuracy and delegation rate", "Authorized 3.5-bit configurations across four frozen candidate-specific thresholds.", "frontier", "frontier-query", "delegation_rate", "Delegation rate", "cascade_accuracy", "Cascade accuracy", "name", "Configuration", frontier, "299 ARC validation examples"),
        {
            "id": "allocation", "title": "KL and C3Q block allocations", "subtitle": "Matched 3.5- and 4.0-bit budgets over 28 transformer blocks.", "showDescription": True,
            "intent": "trend", "question": "Which blocks differ between the KL seed and C3Q selection?", "rationale": "An ordered line view preserves model depth.",
            "comparisonContext": {"grain": "transformer block", "unit": "bits per weight", "denominator": "28 blocks"}, "type": "line", "dataset": "allocation", "sourceId": "allocation-query",
            "encodings": {"x": {"field": "layer", "type": "quantitative", "format": "number", "label": "Block index"}, "y": {"field": "bits", "type": "quantitative", "format": "number", "label": "Assigned bits"}, "color": {"field": "series", "type": "nominal", "label": "Policy / budget"}, "tooltip": [{"field": "layer", "type": "quantitative", "format": "number", "label": "Block"}, {"field": "series", "type": "text", "label": "Policy"}, {"field": "bits", "type": "quantitative", "format": "number", "label": "Bits"}]},
            "palette": {"kind": "categorical", "name": "allocation"}, "settings": {"showPoints": True, "curve": "linear", "sort": "xAsc"}, "legend": {"position": "bottom", "title": "Policy / budget"}, "layout": "full", "maxRows": len(allocation), "surface": {"surface": "explorer", "viewMode": "both"},
        },
        {
            "id": "outcomes", "title": "Delegation outcomes at the 30% calibration target", "subtitle": "KL and C3Q on the same 299 held-out ARC examples.", "showDescription": True,
            "intent": "comparison", "question": "Which routing outcomes explain the utility difference?", "rationale": "Grouped bars compare useful, harmful, and missed delegation outcomes.",
            "comparisonContext": {"grain": "configuration and outcome", "unit": "rate", "denominator": "delegated examples for correction; all examples otherwise"}, "type": "bar", "dataset": "outcomes", "sourceId": "outcome-query",
            "encodings": {"x": {"field": "name", "type": "nominal", "label": "Configuration"}, "y": {"field": "rate", "type": "quantitative", "format": "percent", "label": "Outcome rate"}, "color": {"field": "outcome", "type": "nominal", "label": "Outcome"}, "tooltip": [{"field": "name", "type": "text", "label": "Configuration"}, {"field": "outcome", "type": "text", "label": "Outcome"}, {"field": "rate", "type": "quantitative", "format": "percent", "label": "Rate"}]},
            "palette": {"kind": "categorical", "name": "outcomes"}, "legend": {"position": "bottom", "title": "Outcome"}, "layout": "full", "maxRows": len(outcomes), "surface": {"surface": "explorer", "viewMode": "both"},
        },
    ]

    tables = [
        {"id": "exact-table", "title": "Exact calibration finalists", "subtitle": "Whole-model metrics and joint guardrail status for the frozen finalist set.", "showDescription": True, "dataset": "exact", "sourceId": "exact-query", "defaultSort": {"field": "budget", "direction": "asc"}, "density": "dense", "layout": "full", "columns": [
            {"field": "name", "label": "Candidate", "type": "text"}, {"field": "budget", "label": "Budget", "format": "number"}, {"field": "selected", "label": "Role", "type": "text"}, {"field": "safe", "label": "Joint safe", "type": "boolean"}, {"field": "standalone_accuracy", "label": "Accuracy", "format": "percent"}, {"field": "output_kl", "label": "Output KL", "format": "number"}, {"field": "mean_utility", "label": "Mean utility", "format": "percent"}]},
        {"id": "heldout-table", "title": "Held-out ARC configuration summary", "subtitle": "All baselines evaluated at the authorized 3.5-bit stage; uniform Q4 is shown as a higher-budget reference.", "showDescription": True, "dataset": "heldout", "sourceId": "heldout-query", "defaultSort": {"field": "mean_utility", "direction": "desc"}, "density": "spacious", "layout": "full", "columns": [
            {"field": "name", "label": "Configuration", "type": "text"}, {"field": "standalone_accuracy", "label": "SLM accuracy", "format": "percent"}, {"field": "mean_utility", "label": "Mean utility", "format": "percent"}, {"field": "output_kl", "label": "Output KL", "format": "number"}, {"field": "uncertainty_spearman", "label": "FP rank rho", "format": "number"}, {"field": "error_auroc", "label": "Error AUROC", "format": "number"}]},
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Capability-Constrained Cascade Quantization: Iteration 5"},
        {"id": "summary", "type": "markdown", "sourceId": "summary-json", "body": f"## Technical summary: calibration headroom does not generalize\n\n**Decision: NO-GO for static cascade-aware precision search in this setup.** At 3.5 bits, C3Q increased calibration utility by 3.50 points while satisfying the joint capability guardrail, but on held-out ARC it reduced mean cascade utility by 2.03 points versus KL (95% CI {comparison['bootstrap_95_ci'][0]*100:.2f} to {comparison['bootstrap_95_ci'][1]*100:.2f}). Standalone accuracy remained matched (28.76% versus 28.43%), isolating the failure to routing/generalization rather than capability. At 4.0 bits, no exact alternative survived the guardrail. The ARC gate fails, so MMLU-Pro and packed quantization were not run."},
        {"id": "candidate-text", "type": "markdown", "sourceId": "search-json", "body": "## Screening finds apparent local flexibility at 3.5 bits only\n\nAll 200 exact-budget allocations were evaluated with the cached additive layer-probe surrogate before six finalists per budget were frozen. Exact whole-model calibration retained five of six finalists at 3.5 bits, with a 4.0-point utility range, but retained only KL itself at 4.0 bits. The candidate cloud is a screening diagnostic; only highlighted exact finalists support guardrail and selection claims."},
        {"id": "candidate-chart", "type": "chart", "chartId": "candidate-cloud", "layout": "full"},
        {"id": "exact-text", "type": "markdown", "sourceId": "search-json", "body": "## Exact calibration selects a materially different C3Q configuration\n\nAt 3.5 bits, C3Q changes blocks 7, 11, 14, and 27 relative to KL. Its exact calibration accuracy is 31.2% versus 26.3%, output KL is 0.420 versus 0.472, and mean utility is 38.25% versus 34.75%. Two seeds produce safe local winners, but their modified-layer Jaccard overlap is only 0.33; the third seed has no exact-safe winner among the frozen finalists."},
        {"id": "exact-chart", "type": "chart", "chartId": "exact-cloud", "layout": "full"},
        {"id": "exact-table-block", "type": "table", "tableId": "exact-table", "layout": "full"},
        {"id": "heldout-text", "type": "markdown", "sourceId": "benchmark-json", "body": "## Held-out ARC reverses the calibration ranking\n\nC3Q loses to KL at every target delegation rate: cascade-accuracy differences are -1.34, -0.67, -2.01, and -4.01 points at 10%, 20%, 30%, and 50%. Every interval crosses zero, and the aggregate mean-utility difference is -2.03 points. The calibration-selected improvement therefore does not generalize."},
        {"id": "frontier-chart", "type": "chart", "chartId": "frontier", "layout": "full"},
        {"id": "heldout-table-block", "type": "table", "tableId": "heldout-table", "layout": "full"},
        {"id": "allocation-text", "type": "markdown", "sourceId": "search-json", "body": "## C3Q searches beyond KL without violating the bit budget\n\nAt 3.5 bits, the selected configuration moves four blocks while keeping the exact parameter-weighted total fixed. At 4.0 bits, C3Q collapses to KL because no alternative exact finalist satisfies both the 10% KL tolerance and 1-point calibration-accuracy tolerance."},
        {"id": "allocation-chart", "type": "chart", "chartId": "allocation", "layout": "full"},
        {"id": "outcome-text", "type": "markdown", "sourceId": "benchmark-json", "body": "## Worse delegation outcomes explain the held-out loss\n\nAt the 30% calibration target, C3Q has a lower fallback correction rate than KL (30.8% versus 39.2%), a higher harmful-escalation rate (3.68% versus 3.01%), and a higher missed-benefit rate (26.1% versus 24.7%). The small standalone accuracy gain is therefore offset by poorer prioritization of useful fallback calls."},
        {"id": "outcome-chart", "type": "chart", "chartId": "outcomes", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "summary-json", "body": "## Scope, data, and metric definitions\n\nThe study uses 750 ARC-Challenge training examples for calibration and the untouched 299-example validation split once for evaluation. Qwen2.5-1.5B-Instruct is the SLM and cached Qwen2.5-7B-Instruct outputs are the fallback. Candidate block widths are {2,4,8}; budgets are exact parameter-weighted averages of 3.5 and 4.0 bits. Joint safety requires output KL no more than 1.10 times KL's and calibration accuracy no more than 1 point below KL's. Utility is cascade accuracy minus lambda times actual delegation rate, averaged over four target rates and five lambda values."},
        {"id": "methods", "type": "markdown", "sourceId": "search-json", "body": "## Methodology and experimental design\n\nEach budget begins from Iteration 4's KL allocation. Eighty-four cached one-block probes define an additive log-probability surrogate used only to screen 100 exact-budget candidates per budget. Before exact whole-model results are observed, six finalists are frozen: KL, top seed-specific local candidates, a random control when available, and highest-ranked remaining candidates. Exact fake quantization uses symmetric group-size-128 weights. Candidate-specific matched-rate margin thresholds are calibrated on ARC train and frozen for held-out ARC. Paired bootstrap intervals use 2,000 example-level replicates; McNemar tests compare final correctness at each rate."},
        {"id": "limitations", "type": "markdown", "sourceId": "summary-json", "body": "## Limitations, uncertainty, and robustness\n\nThe planned literal exact evaluation of all 200 candidates was computationally impractical on the 8GB GPU, so candidate-cloud and random-pool coverage rely on an additive layer-probe screening surrogate. Exact conclusions are limited to 12 frozen finalists. The single frozen random finalist did not pass the exact guardrail, so a best-safe-random held-out baseline is unavailable. The paired interval around C3Q's held-out loss crosses zero, but all point estimates favor KL and the positive-result gate fails. Fake quantization does not establish packed memory, latency, or throughput."},
        {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\nStop static mixed-precision cascade optimization for this model pair and benchmark. Do not run MMLU-Pro or packed AWQ/GPTQ validation under this plan. If the broader project continues, pivot to adaptive per-input precision or a compression-aware router, where routing is optimized directly rather than encoded indirectly in a static layer allocation."},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\nCan a Q4/Q8/7B runtime policy exploit per-input disagreement more reliably than static layer bits? Would a learned router targeting fallback uplift generalize better than uncertainty thresholding? Does capability-safe routing headroom replicate under a fallback with materially larger accuracy advantage?"},
    ]

    manifest = {"version": 1, "surface": "report", "title": summary["title"], "description": "Capability-constrained whole-model cascade quantization search on ARC-Challenge.", "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks}
    artifact = {"surface": "report", "manifest": manifest, "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {"candidates": candidates, "exact": exact, "heldout": heldout, "frontier": frontier, "allocation": allocation, "outcomes": outcomes}}, "sources": sources}
    (DEST / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
