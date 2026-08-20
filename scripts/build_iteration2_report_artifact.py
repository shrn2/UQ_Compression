"""Build the canonical portable technical report artifact for Iteration 2."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("runs/iter2_arc")
DEST = ROOT / "report"


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def query_source(source_id: str, label: str, rows: list[dict], columns: list[str], generated: str, filters: list[str], definitions: list[str]) -> dict:
    values = ",\n    ".join(
        "(" + ", ".join(sql_literal(row.get(column)) for column in columns) + ")" for row in rows
    )
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
            "description": f"Materialize the reviewed rows for {label.lower()}.",
            "executed_at": generated,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def scatter_chart(bit: int, source_id: str) -> dict:
    return {
        "id": f"capability-trust-{bit}",
        "title": f"Capability and trust sensitivity at {bit} bits",
        "subtitle": "One point per transformer block; full 750-example ARC-Challenge probing pool.",
        "showDescription": True,
        "intent": "relationship",
        "question": "Do KL-critical and uncertainty-rank-critical blocks occupy different parts of the map?",
        "rationale": "A layer-level scatter directly exposes ranking alignment, outliers, and taxonomy membership.",
        "comparisonContext": {"grain": "transformer block", "unit": "sensitivity", "denominator": "24 blocks"},
        "type": "scatter",
        "dataset": f"scatter_{bit}",
        "sourceId": source_id,
        "encodings": {
            "x": {"field": "capability_kl", "type": "quantitative", "format": "number", "label": "Mean KL from FP"},
            "y": {"field": "trust_rank_disruption", "type": "quantitative", "format": "number", "label": "1 - entropy-rank Spearman"},
            "color": {"field": "region", "type": "nominal", "label": "Depth region"},
            "label": {"field": "layer", "type": "nominal", "label": "Block"},
            "tooltip": [
                {"field": "layer", "type": "text", "label": "Block"},
                {"field": "region", "type": "text", "label": "Region"},
                {"field": "capability_kl", "type": "quantitative", "format": "number", "label": "Capability KL"},
                {"field": "trust_rank_disruption", "type": "quantitative", "format": "number", "label": "Trust disruption"},
                {"field": "rank_displacement", "type": "quantitative", "format": "number", "label": "Rank displacement"},
            ],
        },
        "palette": {"kind": "categorical", "name": "depth-region"},
        "legend": {"position": "bottom", "title": "Depth region"},
        "layout": "full",
        "maxRows": 24,
        "surface": {"surface": "explorer", "viewMode": "both"},
    }


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "summary.json", DEST / "summary.json")
    shutil.copyfile(ROOT / "layer_analysis.json", DEST / "layer_analysis.json")
    scatter_2 = [row for row in summary["scatter_rows"] if row["bits"] == 2]
    scatter_4 = [row for row in summary["scatter_rows"] if row["bits"] == 4]
    depth_4 = [row for row in summary["depth_rows"] if row["bits"] == 4]
    rankings = summary["robustness_rows"]
    taxonomy = sorted(
        summary["taxonomy_rows"], key=lambda row: (row["bits"], -row["rank_displacement"])
    )
    sources = [
        {"id": "iteration-summary", "label": "Iteration 2 reviewed summary", "path": "summary.json"},
        {"id": "layer-analysis", "label": "Iteration 2 layer analysis", "path": "layer_analysis.json"},
        {"id": "arc-dataset", "label": "allenai/ai2_arc dataset card", "href": "https://huggingface.co/datasets/allenai/ai2_arc"},
        {"id": "student-model", "label": "Qwen2.5-0.5B-Instruct model card", "href": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"},
        query_source(
            "scatter-2-query", "2-bit capability/trust layer rows", scatter_2,
            ["bits", "layer", "layer_index", "region", "capability_kl", "trust_rank_disruption", "taxonomy", "rank_displacement"], generated,
            ["ARC-Challenge train pool", "750 examples", "one-block 2-bit probes", "24 transformer blocks"],
            ["capability_kl = mean KL(FP student || one-block quantized)", "trust_rank_disruption = 1 - Spearman(FP entropy, probe entropy)"],
        ),
        query_source(
            "scatter-4-query", "4-bit capability/trust layer rows", scatter_4,
            ["bits", "layer", "layer_index", "region", "capability_kl", "trust_rank_disruption", "taxonomy", "rank_displacement"], generated,
            ["ARC-Challenge train pool", "750 examples", "one-block 4-bit probes", "24 transformer blocks"],
            ["capability_kl = mean KL(FP student || one-block quantized)", "trust_rank_disruption = 1 - Spearman(FP entropy, probe entropy)"],
        ),
        query_source(
            "depth-4-query", "4-bit normalized sensitivity by depth", depth_4,
            ["bits", "layer", "layer_index", "sensitivity", "normalized_score"], generated,
            ["ARC-Challenge train pool", "4-bit probes", "min-max normalized within metric"],
            ["Capability uses mean output KL", "Trust uses entropy-rank disruption"],
        ),
        query_source(
            "ranking-query", "Ranking and robustness diagnostics", rankings,
            ["bits", "capability_trust_rho", "rho_ci_low", "rho_ci_high", "capability_margin_rho", "entropy_margin_map_rho", "correct_incorrect_map_rho", "top3_overlap", "top5_overlap", "top8_overlap", "stable_trust_only_layers"], generated,
            ["1,000 example-bootstrap replicates", "top quartile = six of 24 blocks"],
            ["rho CI recomputes capability and trust maps after resampling examples", "overlap = intersection size / k"],
        ),
        query_source(
            "taxonomy-query", "Layer taxonomy and rank stability", taxonomy,
            ["bits", "layer", "region", "taxonomy", "capability_rank", "trust_rank", "rank_displacement", "capability_top_frequency", "trust_top_frequency", "stable_trust_only"], generated,
            ["all 24 blocks", "2/4/8-bit probes", "top quartile threshold"],
            ["stable trust-only requires trust top-quartile frequency >=0.8 and capability frequency <=0.2"],
        ),
    ]
    rho2 = rankings[0]["capability_trust_rho"]
    rho4 = rankings[1]["capability_trust_rho"]
    ci2 = (rankings[0]["rho_ci_low"], rankings[0]["rho_ci_high"])
    ci4 = (rankings[1]["rho_ci_low"], rankings[1]["rho_ci_high"])
    title = "Where Does Self-Knowledge Break? Iteration 2"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Layer-localization test of capability versus trust sensitivity under fake quantization.",
        "generatedAt": generated,
        "sources": sources,
        "charts": [
            scatter_chart(2, "scatter-2-query"),
            scatter_chart(4, "scatter-4-query"),
            {
                "id": "depth-4",
                "title": "Capability and trust sensitivity peak in the same early blocks",
                "subtitle": "4-bit one-block probes; each series min-max normalized over 24 blocks.",
                "showDescription": True,
                "intent": "trend",
                "question": "Do the two sensitivity maps peak in different depth regions?",
                "rationale": "An ordered line comparison preserves transformer depth and makes coincident peaks visible.",
                "comparisonContext": {"grain": "transformer block", "unit": "normalized sensitivity", "denominator": "24 blocks"},
                "type": "line",
                "dataset": "depth_4",
                "sourceId": "depth-4-query",
                "encodings": {
                    "x": {"field": "layer_index", "type": "quantitative", "format": "number", "label": "Transformer block index"},
                    "y": {"field": "normalized_score", "type": "quantitative", "format": "number", "label": "Normalized sensitivity"},
                    "color": {"field": "sensitivity", "type": "nominal", "label": "Sensitivity map"},
                    "tooltip": [
                        {"field": "layer", "type": "text", "label": "Block"},
                        {"field": "sensitivity", "type": "text", "label": "Map"},
                        {"field": "normalized_score", "type": "quantitative", "format": "number", "label": "Normalized score"},
                    ],
                },
                "palette": {"kind": "categorical", "name": "sensitivity-map"},
                "settings": {"showPoints": True, "curve": "linear", "sort": "xAsc"},
                "legend": {"position": "bottom", "title": "Sensitivity map"},
                "layout": "full",
                "maxRows": 48,
                "surface": {"surface": "explorer", "viewMode": "both"},
            },
        ],
        "tables": [
            {
                "id": "ranking-diagnostics",
                "title": "Ranking, overlap, and estimator robustness",
                "subtitle": "Bootstrap intervals use 1,000 example-level resamples.",
                "showDescription": True,
                "dataset": "rankings",
                "sourceId": "ranking-query",
                "defaultSort": {"field": "bits", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "bits", "label": "Bits", "format": "number"},
                    {"field": "capability_trust_rho", "label": "C-vs-R rho", "format": "number"},
                    {"field": "rho_ci_low", "label": "CI low", "format": "number"},
                    {"field": "rho_ci_high", "label": "CI high", "format": "number"},
                    {"field": "top5_overlap", "label": "Top-5 overlap", "format": "percent"},
                    {"field": "entropy_margin_map_rho", "label": "Entropy-margin map rho", "format": "number"},
                    {"field": "stable_trust_only_layers", "label": "Stable trust-only", "format": "number"},
                ],
            },
            {
                "id": "layer-taxonomy",
                "title": "Complete layer taxonomy",
                "subtitle": "Sorted by bit-width and absolute capability/trust rank displacement.",
                "showDescription": True,
                "dataset": "taxonomy",
                "sourceId": "taxonomy-query",
                "defaultSort": {"field": "rank_displacement", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "bits", "label": "Bits", "format": "number"},
                    {"field": "layer", "label": "Block", "type": "text"},
                    {"field": "region", "label": "Region", "type": "text"},
                    {"field": "taxonomy", "label": "Taxonomy", "type": "text"},
                    {"field": "capability_rank", "label": "C rank", "format": "number"},
                    {"field": "trust_rank", "label": "R rank", "format": "number"},
                    {"field": "rank_displacement", "label": "|rank delta|", "format": "number"},
                    {"field": "trust_top_frequency", "label": "Trust top frequency", "format": "percent"},
                    {"field": "stable_trust_only", "label": "Stable trust-only", "type": "boolean"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical-summary", "type": "markdown", "sourceId": "iteration-summary",
                "body": (
                    "## Technical summary: no distinct trust-critical layer map\n\n"
                    f"**Decision: NO-GO.** Capability sensitivity and entropy-rank disruption are almost collinear across blocks: **rho={rho2:.3f}** at 2 bits (bootstrap 95% CI {ci2[0]:.3f} to {ci2[1]:.3f}) and **rho={rho4:.3f}** at 4 bits ({ci4[0]:.3f} to {ci4[1]:.3f}). Every capability top-quartile block is also trust top-quartile at both meaningful precisions, and no layer satisfies the bootstrap-stable trust-only rule. The preregistered rho<0.7 gate therefore failed. Whole-model policy validation and MMLU-Pro replication were not run."
                ),
            },
            {
                "id": "findings", "type": "markdown", "sourceId": "layer-analysis",
                "body": (
                    "## Key finding: prediction drift and uncertainty-order drift share a source\n\n"
                    "At 2 bits, top-5 overlap is 80%; at 4 bits it is 100%. The strongest 4-bit blocks under both maps are L4, L2, L5, L3, and L1. At 2 bits, only one of the top five differs, while the top-eight sets are identical. The scatter plots retain exact layer values and labels; this is a high alignment result, not merely a failure to reach an arbitrary threshold."
                ),
            },
            {"id": "scatter-2-block", "type": "chart", "chartId": "capability-trust-2", "layout": "full"},
            {"id": "scatter-4-block", "type": "chart", "chartId": "capability-trust-4", "layout": "full"},
            {
                "id": "depth-findings", "type": "markdown", "sourceId": "layer-analysis",
                "body": (
                    "## Depth analysis: early blocks dominate both maps\n\n"
                    "For 4-bit probes, the early third averages 0.481 normalized capability sensitivity and 0.546 normalized trust sensitivity. The middle and late thirds are much lower on both measures. Because only cached whole-block interventions exist, attention-versus-MLP and projection-level localization is not identified in this iteration."
                ),
            },
            {"id": "depth-chart-block", "type": "chart", "chartId": "depth-4", "layout": "full"},
            {
                "id": "scope", "type": "markdown", "sourceId": "iteration-summary",
                "body": (
                    "## Scope and metric definitions\n\n"
                    "The analysis reuses all 750 deterministic ARC-Challenge training examples and 24 one-block Qwen2.5-0.5B-Instruct probes at 2, 4, and 8 bits. Capability is mean KL(FP || one-block quantized). The primary trust metric is 1 minus Spearman correlation between FP and probe predictive-entropy rankings. Prediction flips, accuracy drop, error AUROC, AURC, ECE, Brier, margin uncertainty, correct/error subsets, and teacher agreement are secondary diagnostics. The teacher is never an allocation signal."
                ),
            },
            {
                "id": "method", "type": "markdown", "sourceId": "layer-analysis",
                "body": (
                    "## Experimental design and statistical method\n\n"
                    "Each transformer block is independently fake-quantized while all other components remain in floating point, using symmetric group-size-128 weights. Scores are computed on identical examples for every intervention. For each bit width, 1,000 bootstrap samples resample examples with replacement and recompute all 24 capability and trust scores, their Spearman correlation, top-k overlaps (k=3,5,8), confidence intervals, and top-quartile selection frequencies. A layer is called uniquely trust-critical only if its trust top-quartile frequency is at least 0.8 while its capability frequency is at most 0.2."
                ),
            },
            {
                "id": "robustness", "type": "markdown", "sourceId": "iteration-summary",
                "body": (
                    "## Robustness: the negative result survives alternative uncertainty views\n\n"
                    "Capability-versus-margin sensitivity rho is 0.957 at 2 bits and 0.984 at 4 bits. Entropy- and margin-derived trust maps correlate at 0.968 and 0.943, respectively. At 8 bits the primary C-vs-R rho rises to 0.993. Correct-only and incorrect-only disruption maps, exact top-k overlaps, confidence intervals, and complete layer stability frequencies are preserved in the tables."
                ),
            },
            {"id": "ranking-table-block", "type": "table", "tableId": "ranking-diagnostics", "layout": "full"},
            {"id": "taxonomy-table-block", "type": "table", "tableId": "layer-taxonomy", "layout": "full"},
            {
                "id": "limitations", "type": "markdown", "sourceId": "iteration-summary",
                "body": (
                    "## Limitations and claim boundary\n\n"
                    "This is a layer-localization result for one 0.5B model, one multiple-choice dataset, and controlled fake quantization. It does not identify within-block attention/MLP components, prove the absence of trust-only structure in other architectures or tasks, or establish packed size, latency, memory, or throughput. The reserved 299-example ARC validation set was not consumed because the policy-stage gate failed. Portable-report verification is structural only where no compatible Chromium browser is installed."
                ),
            },
            {
                "id": "next-steps", "type": "markdown",
                "body": (
                    "## Recommended next step: move from localization to routing drift\n\n"
                    "Stop UQ-aware mixed-precision allocation for this setup. Measure whole-model quantization-induced uncertainty and delegation drift instead: freeze an SLM-to-LLM routing rule, quantify threshold crossings and error-routing failures across precision, test recalibration as a recovery mechanism, and compare accuracy at matched delegation cost. This direction remains useful even when capability- and trust-critical layers coincide."
                ),
            },
            {
                "id": "further-questions", "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "Do whole-model interactions create uncertainty drift that one-block probes cannot predict? Are routing failures concentrated near delegation thresholds? Can post-quantization calibration restore selective risk without changing weights? Does the shared early-layer sensitivity map replicate in a larger student or a non-multiple-choice task?"
                ),
            },
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "scatter_2": scatter_2,
                "scatter_4": scatter_4,
                "depth_4": depth_4,
                "rankings": rankings,
                "taxonomy": taxonomy,
            },
        },
        "sources": sources,
    }
    (DEST / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
