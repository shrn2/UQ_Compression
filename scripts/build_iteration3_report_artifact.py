"""Build the canonical portable technical report artifact for Iteration 3."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter3_arc")
DEST = ROOT / "report"


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def source(source_id: str, label: str, rows: list[dict], columns: list[str], generated: str,
           filters: list[str], definitions: list[str]) -> dict:
    values = ",\n    ".join(
        "(" + ", ".join(literal(row.get(column)) for column in columns) + ")" for row in rows
    )
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
            "description": f"Materialize reviewed {label.lower()} rows.",
            "executed_at": generated,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def line_chart(chart_id: str, title: str, subtitle: str, dataset: str, source_id: str,
               x_field: str, x_label: str, y_field: str, y_label: str, color_field: str,
               color_label: str, question: str, max_rows: int) -> dict:
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "showDescription": True,
        "intent": "trend",
        "question": question,
        "rationale": "A common-scale line comparison exposes how the metric changes over ordered precision or delegation levels.",
        "comparisonContext": {"grain": "experimental condition", "unit": "proportion", "denominator": "held-out examples"},
        "type": "line",
        "dataset": dataset,
        "sourceId": source_id,
        "encodings": {
            "x": {"field": x_field, "type": "quantitative", "format": "number", "label": x_label},
            "y": {"field": y_field, "type": "quantitative", "format": "percent", "label": y_label},
            "color": {"field": color_field, "type": "nominal", "label": color_label},
            "tooltip": [
                {"field": x_field, "type": "quantitative", "format": "number", "label": x_label},
                {"field": color_field, "type": "text", "label": color_label},
                {"field": y_field, "type": "quantitative", "format": "percent", "label": y_label},
            ],
        },
        "palette": {"kind": "categorical", "name": chart_id},
        "settings": {"showPoints": True, "curve": "linear", "sort": "xAsc"},
        "legend": {"position": "bottom", "title": color_label},
        "layout": "full",
        "maxRows": max_rows,
        "surface": {"surface": "explorer", "viewMode": "both"},
    }


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "summary.json", ROOT / "analysis.json", Path("runs/iter3_mmlu_pro/analysis.json")):
        shutil.copyfile(path, DEST / path.name if path.parent == ROOT else DEST / "mmlu_analysis.json")

    # Tidy metric rows allow accuracy and rank preservation to appear as separate series.
    metric_rows = []
    for row in summary["precision_rows"]:
        for metric, value in (("SLM accuracy", row["accuracy"]), ("Uncertainty-rank Spearman", row["uncertainty_spearman"])):
            metric_rows.append({"dataset": row["dataset"], "bit_width": row["bit_width"], "precision": row["precision"], "metric": metric, "value": value})
    arc_metrics = [row for row in metric_rows if row["dataset"] == "ARC-Challenge"]
    mmlu_metrics = [row for row in metric_rows if row["dataset"] == "MMLU-Pro"]
    arc_dfr = [row for row in summary["dfr_rows"] if row["dataset"] == "ARC-Challenge"]
    mmlu_dfr = [row for row in summary["dfr_rows"] if row["dataset"] == "MMLU-Pro" and row["method"] == "matched_rate"]
    frontier = [
        {**row, "series": f"{row['precision']} / {row['method']}"}
        for row in summary["frontier_rows"]
    ]
    mmlu_frontier = [
        {**row, "series": f"{row['precision']} / {row['method']}"}
        for row in summary["mmlu_frontier_rows"]
    ]
    recalibration = [row for row in summary["recalibration_rows"] if row["precision"] in {"q4", "q2"}]
    for row in recalibration:
        row["series"] = f"{row['precision']} / {row['method']}"
    preserved = [row for row in summary["accuracy_preserved_rows"] if row["method"] == "matched_rate"]
    for row in preserved:
        row["series"] = f"{row['dataset']} / {row['precision']}"
    flip_taxonomy = [row for row in summary["flip_taxonomy_rows"] if row["precision"] in {"q4", "q2"}]

    sources = [
        {"id": "summary-json", "label": "Reviewed Iteration 3 summary", "path": "summary.json"},
        {"id": "arc-analysis", "label": "ARC-Challenge cascade analysis", "path": "analysis.json"},
        {"id": "mmlu-analysis", "label": "MMLU-Pro cascade analysis", "path": "mmlu_analysis.json"},
        {"id": "mmlu-card", "label": "Official MMLU-Pro dataset card", "href": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"},
        source("arc-metrics-query", "ARC precision metrics", arc_metrics, ["dataset", "bit_width", "precision", "metric", "value"], generated,
               ["ARC-Challenge validation", "299 examples", "Qwen2.5-1.5B SLM"], ["accuracy = exact multiple-choice accuracy", "Spearman compares FP and quantized uncertainty ranks"]),
        source("mmlu-metrics-query", "MMLU-Pro precision metrics", mmlu_metrics, ["dataset", "bit_width", "precision", "metric", "value"], generated,
               ["MMLU-Pro test sample", "1,000 examples", "Qwen2.5-1.5B SLM"], ["accuracy = exact multiple-choice accuracy", "Spearman compares FP and quantized uncertainty ranks"]),
        source("arc-dfr-query", "ARC frozen-threshold delegation flips", arc_dfr, ["precision", "target_rate", "delegation_flip_rate"], generated,
               ["frozen FP thresholds", "299 evaluation examples"], ["DFR = P(D_Q differs from D_FP)"]),
        source("mmlu-dfr-query", "MMLU-Pro matched-rate delegation flips", mmlu_dfr, ["precision", "target_rate", "delegation_flip_rate"], generated,
               ["exact matched delegation rate", "1,000 evaluation examples"], ["DFR = P(D_Q differs from D_FP)"]),
        source("frontier-query", "ARC cascade frontier", frontier, ["precision", "method", "series", "target_rate", "delegation_rate", "cascade_accuracy"], generated,
               ["Qwen2.5-7B fallback", "matched-rate uncertainty routers", "random mean over 10 seeds"], ["cascade output uses fallback on delegated examples", "oracle prioritizes SLM errors"]),
        source("mmlu-frontier-query", "MMLU-Pro cascade frontier", mmlu_frontier, ["precision", "method", "series", "target_rate", "delegation_rate", "cascade_accuracy"], generated,
               ["Qwen2.5-3B fallback", "matched-rate uncertainty routers", "random mean over 10 seeds"], ["cascade output uses fallback on delegated examples", "oracle prioritizes SLM errors"]),
        source("recalibration-query", "ARC recalibration comparisons", recalibration, ["precision", "series", "target_rate", "method", "delegation_rate", "cascade_accuracy"], generated,
               ["Q4 and Q2", "299 evaluation examples"], ["frozen transfers FP threshold", "retuned and matched use quantization-specific calibration", "isotonic maps uncertainty to error probability"]),
        source("preserved-query", "Accuracy-preserved delegation flips", preserved, ["dataset", "precision", "series", "target_rate", "same_prediction_examples", "delegation_flip_rate"], generated,
               ["FP and quantized SLM predict the same answer", "matched delegation rate"], ["DFR is conditional on identical predictions"]),
        source("routing-only-query", "Routing-only paired comparisons", summary["routing_only_rows"], ["dataset", "precision", "target_rate", "fp_ranked_minus_q_ranked_accuracy", "ci_low", "ci_high", "mcnemar_p"], generated,
               ["same Q SLM predictions", "same fallback predictions", "same exact delegation rate", "2,000 paired bootstrap replicates"], ["difference = cascade accuracy under FP ranking minus accuracy under quantized ranking"]),
        source("precision-query", "Complete precision diagnostics", summary["precision_rows"], ["dataset", "precision", "bit_width", "accuracy", "error_auroc", "uncertainty_spearman", "ordering_inversion_rate", "prediction_agreement", "top20_overlap"], generated,
               ["ARC-Challenge evaluation and MMLU-Pro test sample", "FP/Q8/Q4/Q2"], ["IOR is pairwise uncertainty-order inversion", "agreement is equality of FP and quantized argmax predictions"]),
        source("taxonomy-query", "ARC frozen-threshold routing flip taxonomy", flip_taxonomy, ["precision", "target_rate", "category", "rate"], generated,
               ["Q4 and Q2", "frozen FP thresholds"], ["HFA keeps a fallback-correct Q error local", "WFE escalates a Q-correct example"]),
    ]

    charts = [
        line_chart("arc-precision", "ARC accuracy and uncertainty-rank preservation", "299 held-out examples; FP is plotted at 16 bits.", "arc_metrics", "arc-metrics-query", "bit_width", "Bit width", "value", "Metric value", "metric", "Metric", "Do accuracy and uncertainty ordering degrade together?", len(arc_metrics)),
        line_chart("mmlu-precision", "MMLU-Pro accuracy and uncertainty-rank preservation", "Deterministic category-stratified sample of 1,000 test examples.", "mmlu_metrics", "mmlu-metrics-query", "bit_width", "Bit width", "value", "Metric value", "metric", "Metric", "Does the precision pattern replicate on a broader benchmark?", len(mmlu_metrics)),
        line_chart("arc-dfr", "ARC delegation flip rate under frozen thresholds", "FP thresholds target 10%, 20%, 30%, and 50% delegation on calibration data.", "arc_dfr", "arc-dfr-query", "target_rate", "Target delegation rate", "delegation_flip_rate", "Delegation flip rate", "precision", "Precision", "How often does direct FP-policy transfer change a decision?", len(arc_dfr)),
        line_chart("mmlu-dfr", "MMLU-Pro delegation flip rate at matched cost", "Both routers delegate exactly the same fraction, isolating selection disagreement.", "mmlu_dfr", "mmlu-dfr-query", "target_rate", "Target delegation rate", "delegation_flip_rate", "Delegation flip rate", "precision", "Precision", "How often do FP and quantized rankings select different examples?", len(mmlu_dfr)),
        line_chart("arc-frontier", "ARC cascade accuracy by delegation rate", "Uncertainty, random, and oracle policies use the same precision-specific SLM and 7B fallback.", "frontier", "frontier-query", "delegation_rate", "Delegation rate", "cascade_accuracy", "Cascade accuracy", "series", "Router", "How much accuracy does each router obtain at a given fallback-call rate?", len(frontier)),
        line_chart("mmlu-frontier", "MMLU-Pro cascade accuracy by delegation rate", "Uncertainty, random, and oracle policies use the same precision-specific SLM and 3B fallback.", "mmlu_frontier", "mmlu-frontier-query", "delegation_rate", "Delegation rate", "cascade_accuracy", "Cascade accuracy", "series", "Router", "Does routing behavior translate into system accuracy on MMLU-Pro?", len(mmlu_frontier)),
        line_chart("recalibration", "ARC cascade accuracy after router recalibration", "Q4 and Q2 under frozen, retuned, temperature, isotonic, and exact matched-rate policies.", "recalibration", "recalibration-query", "target_rate", "Target delegation rate", "cascade_accuracy", "Cascade accuracy", "series", "Condition", "Does calibration recovery restore the frontier?", len(recalibration)),
        line_chart("preserved", "Delegation drift when the answer is unchanged", "Matched-rate control restricted to examples with identical FP and quantized predictions.", "preserved", "preserved-query", "target_rate", "Target delegation rate", "delegation_flip_rate", "Delegation flip rate", "series", "Dataset / precision", "Does decision drift remain without prediction drift?", len(preserved)),
    ]

    tables = [
        {"id": "routing-only", "title": "Routing-only paired accuracy differences", "subtitle": "ARC intervals cross zero; two MMLU-Pro Q2 intervals favor the quantized ranking.", "showDescription": True, "dataset": "routing_only", "sourceId": "routing-only-query", "defaultSort": {"field": "dataset", "direction": "asc"}, "density": "dense", "layout": "full", "columns": [
            {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "precision", "label": "Precision", "type": "text"}, {"field": "target_rate", "label": "Target rate", "format": "percent"}, {"field": "fp_ranked_minus_q_ranked_accuracy", "label": "FP-rank minus Q-rank", "format": "percent"}, {"field": "ci_low", "label": "CI low", "format": "percent"}, {"field": "ci_high", "label": "CI high", "format": "percent"}, {"field": "mcnemar_p", "label": "McNemar p", "format": "number"}]},
        {"id": "precision-table", "title": "Complete precision diagnostics", "subtitle": "Task, uncertainty, ranking, agreement, and top-20 overlap metrics.", "showDescription": True, "dataset": "precision_rows", "sourceId": "precision-query", "density": "dense", "layout": "full", "columns": [
            {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "precision", "label": "Precision", "type": "text"}, {"field": "accuracy", "label": "Accuracy", "format": "percent"}, {"field": "error_auroc", "label": "Error AUROC", "format": "number"}, {"field": "uncertainty_spearman", "label": "Rank rho", "format": "number"}, {"field": "ordering_inversion_rate", "label": "IOR", "format": "percent"}, {"field": "prediction_agreement", "label": "Prediction agreement", "format": "percent"}, {"field": "top20_overlap", "label": "Top-20% overlap", "format": "percent"}]},
        {"id": "taxonomy-table", "title": "Frozen-threshold ARC flip taxonomy", "subtitle": "Rates use the full 299-example evaluation denominator.", "showDescription": True, "dataset": "flip_taxonomy", "sourceId": "taxonomy-query", "density": "dense", "layout": "full", "columns": [
            {"field": "precision", "label": "Precision", "type": "text"}, {"field": "target_rate", "label": "Target rate", "format": "percent"}, {"field": "category", "label": "Category", "type": "text"}, {"field": "rate", "label": "Rate", "format": "percent"}]},
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# When Compression Changes Who Gets Help: Iteration 3"},
        {"id": "summary", "type": "markdown", "sourceId": "summary-json", "body": "## Technical summary: routing drift is real, but routing-only harm is not established\n\n**Decision: MODERATE POSITIVE.** The 0.5B ARC router failed the preregistered AUROC gate, so the experiment scaled to Qwen2.5-1.5B with a Qwen2.5-7B fallback. Margin uncertainty passed on calibration (AUROC 0.613). Q4 and Q2 damaged uncertainty ordering and changed delegation decisions on ARC and MMLU-Pro. All 12 routing-only ARC accuracy intervals cross zero. On MMLU-Pro, two Q2 intervals at 30% and 50% delegation significantly favor the quantized ranking rather than FP ranking. The significant end-to-end Q4/Q2 gaps therefore cannot be credited to harmful routing damage."},
        {"id": "arc-metric-text", "type": "markdown", "sourceId": "arc-analysis", "body": "## Key finding 1: aggressive quantization separates accuracy from uncertainty order\n\nOn ARC, Q8 preserves both behavior and ordering (rank rho 0.989). Q4 lowers accuracy from 49.8% to 42.8% and rank rho to 0.562; Q2 lowers them to 28.1% and 0.161. This establishes compression-induced uncertainty drift, but aggregate accuracy is not preserved at the aggressive settings."},
        {"id": "arc-metric-chart", "type": "chart", "chartId": "arc-precision", "layout": "full"},
        {"id": "mmlu-metric-text", "type": "markdown", "sourceId": "mmlu-analysis", "body": "## Cross-benchmark replication: MMLU-Pro shows the same ranking collapse\n\nOn 1,000 MMLU-Pro test examples, Q8 rank rho is 0.994, Q4 is 0.589, and Q2 is -0.011. The 3B fallback reaches 32.7% accuracy versus 26.3% for the FP SLM, enabling a complete cascade comparison with a +6.4 percentage-point fallback advantage."},
        {"id": "mmlu-metric-chart", "type": "chart", "chartId": "mmlu-precision", "layout": "full"},
        {"id": "arc-dfr-text", "type": "markdown", "sourceId": "arc-analysis", "body": "## Key finding 2: frozen FP policies are unstable below 8 bits\n\nARC frozen-threshold delegation flip rates span 16.4%–27.1% for Q4 and 13.7%–45.2% for Q2 across operating points. Q8 remains at or below 5.0%. Some drift is threshold-scale transfer, so matched-rate controls are required before inferring selection harm."},
        {"id": "arc-dfr-chart", "type": "chart", "chartId": "arc-dfr", "layout": "full"},
        {"id": "mmlu-dfr-text", "type": "markdown", "sourceId": "mmlu-analysis", "body": "## Matched-rate decision drift replicates on MMLU-Pro\n\nAt identical fallback-call rates, Q4 changes 12.0%–28.8% of routing decisions and Q2 changes 18.8%–49.0%. This remains a decision-level result because fallback outputs are unavailable."},
        {"id": "mmlu-dfr-chart", "type": "chart", "chartId": "mmlu-dfr", "layout": "full"},
        {"id": "frontier-text", "type": "markdown", "sourceId": "arc-analysis", "body": "## Cascade frontier: fallback quality is necessary, and model degradation dominates the gap\n\nThe 7B fallback is 4.35 percentage points more accurate than the FP SLM on ARC evaluation. Uncertainty routing is compared against ten-seed random escalation and a precision-specific oracle. Q4/Q2 end-to-end curves lag FP, but those curves combine changed local predictions with changed rankings."},
        {"id": "frontier-chart", "type": "chart", "chartId": "arc-frontier", "layout": "full"},
        {"id": "mmlu-frontier-text", "type": "markdown", "sourceId": "mmlu-analysis", "body": "## MMLU-Pro cascade frontier\n\nThe broader benchmark repeats the matched-rate, random, and oracle comparisons with a Qwen2.5-3B fallback. It completes the planned system-level scale-up while retaining a meaningful fallback capability gap."},
        {"id": "mmlu-frontier-chart", "type": "chart", "chartId": "mmlu-frontier", "layout": "full"},
        {"id": "routing-only-text", "type": "markdown", "sourceId": "summary-json", "body": "## Causal control: no significant harmful routing-only cascade loss\n\nFor each precision and rate, the primary control routes the same quantized SLM and same fallback using either FP or quantized uncertainty ranks. Every ARC interval includes zero. On MMLU-Pro, Q2 quantized ranking is 2.2 points better at 30% delegation (95% CI 0.2–4.3 points) and 4.2 points better at 50% (1.8–6.7 points). Compression changes selection, but the evidence does not show that this change harms cascade accuracy."},
        {"id": "routing-only-table", "type": "table", "tableId": "routing-only", "layout": "full"},
        {"id": "recalibration-text", "type": "markdown", "sourceId": "arc-analysis", "body": "## Recalibration: rate recovery is possible; ranking recovery is not\n\nRetuned thresholds, temperature scaling, and isotonic error calibration repair score-scale and rate shifts to varying degrees. Exact matched-rate routing removes count differences by construction. None changes the underlying quantized ranking, and the held-fixed control shows no detectable accuracy penalty from that remaining rank disagreement at n=299."},
        {"id": "recalibration-chart", "type": "chart", "chartId": "recalibration", "layout": "full"},
        {"id": "preserved-text", "type": "markdown", "sourceId": "summary-json", "body": "## Accuracy-preserved control: routing decisions still change\n\nDelegation flips persist among examples where FP and quantized SLMs choose the same answer. On MMLU-Pro, Q4 matched-rate DFR rises from 5.8% to 27.2% across operating points within 555 identical-prediction examples; Q2 reaches 53.7% within 136. Compression therefore changes the help-seeking decision beyond answer flips, even though system harm is not identified."},
        {"id": "preserved-chart", "type": "chart", "chartId": "preserved", "layout": "full"},
        {"id": "taxonomy-text", "type": "markdown", "sourceId": "arc-analysis", "body": "## Flip taxonomy under direct policy transfer\n\nFrozen-threshold Q4 harmful false accepts reach 2.0%–3.7% of all ARC examples; Q2 reaches 3.7%–11.7%. On MMLU-Pro the maxima are 2.3% and 5.7%. These are descriptive policy-transfer failures, not routing-only causal effects, because the quantized local prediction also changes."},
        {"id": "taxonomy-table-block", "type": "table", "tableId": "taxonomy-table", "layout": "full"},
        {"id": "precision-table-text", "type": "markdown", "sourceId": "summary-json", "body": "## Scope and metric definitions\n\nARC uses 750 calibration and 299 held-out validation examples. MMLU-Pro uses all 70 validation examples for calibration and a deterministic category-stratified sample of 1,000 test examples. Uncertainty is the calibration-selected signal: margin for ARC and entropy for MMLU-Pro. DFR is the fraction of examples whose binary delegation decision differs from FP. IOR is the pairwise uncertainty-order inversion rate."},
        {"id": "precision-table-block", "type": "table", "tableId": "precision-table", "layout": "full"},
        {"id": "methods", "type": "markdown", "sourceId": "summary-json", "body": "## Methods and experimental design\n\nWhole-model Q8/Q4/Q2 conditions use symmetric group-size-128 fake weight quantization over all transformer blocks. Router signals and thresholds are selected using calibration data only, at target delegation rates of 10%, 20%, 30%, and 50%. Evaluation includes frozen FP thresholds, precision-specific thresholds, temperature scaling, isotonic error calibration, exact matched-rate routing, always-small, always-large, ten random seeds, and oracle error routing. Accuracy differences use 2,000 paired bootstrap replicates and McNemar tests."},
        {"id": "limitations", "type": "markdown", "sourceId": "summary-json", "body": "## Limitations and robustness\n\nFake quantization establishes numerical sensitivity but not packed size, memory, latency, throughput, or kernel behavior. ARC has only 299 evaluation examples, and MMLU-Pro uses a deterministic 1,000-example sample rather than the entire test split. The fallbacks differ by benchmark (7B on ARC, 3B on MMLU-Pro), so absolute frontiers are not directly comparable. Portable-report verification is structural because no compatible browser was available."},
        {"id": "next", "type": "markdown", "body": "## Recommended next step\n\nDo not proceed to packed-quantizer deployment claims under the preregistered strong-positive rule: neither benchmark shows harmful routing-only regret. The useful engineering direction is compression-aware router monitoring and post-quantization recalibration. A research follow-up should target settings with a larger fallback advantage or train a separate router, then require a held-fixed routing-only loss before spending on AWQ/GPTQ validation."},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\nDoes routing-only regret appear on harder subsets where the fallback advantage is larger? Can rank-preserving calibration or a separate router recover the FP selection set? How do packed Q4 kernels compare with controlled fake quantization? Are harmful flips concentrated in particular MMLU-Pro categories or far from the threshold?"},
    ]

    manifest = {"version": 1, "surface": "report", "title": summary["title"], "description": "Quantization-induced delegation drift in Qwen small-to-large cascades.", "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks}
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {
            "arc_metrics": arc_metrics, "mmlu_metrics": mmlu_metrics, "arc_dfr": arc_dfr, "mmlu_dfr": mmlu_dfr,
            "frontier": frontier, "mmlu_frontier": mmlu_frontier, "recalibration": recalibration, "preserved": preserved,
            "routing_only": summary["routing_only_rows"], "precision_rows": summary["precision_rows"], "flip_taxonomy": flip_taxonomy,
        }},
        "sources": sources,
    }
    (DEST / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
