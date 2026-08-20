"""Build the canonical portable technical report artifact for Iteration 8."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter8_arc")
DEST = ROOT / "report"


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def query_source(rows: list[dict], generated: str) -> dict:
    columns = [
        "method", "auroc", "auprc", "precision_at_10", "precision_at_20",
        "precision_at_30", "best_epoch", "epochs_run", "validation_loss",
    ]
    values = ",\n    ".join(
        "(" + ", ".join(literal(row.get(column)) for column in columns) + ")" for row in rows
    )
    return {
        "id": "contrast-query", "label": "Held-out capacity-contrast diagnostics",
        "query": {
            "engine": "SQLite", "language": "sql",
            "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
            "description": "Materialize the four reviewed held-out router results.",
            "executed_at": generated,
            "filters": [
                "600 deterministic class-stratified held-out MMLU auxiliary-train prompts",
                "MMLU-Pro excluded", "S50-only deployment features",
            ],
            "metric_definitions": [
                "AUROC and AUPRC target hard S50-S25 prediction disagreement",
                "precision-at-k selects the highest scoring exact k-percent subset",
            ],
        },
    }


def main() -> int:
    analysis = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "analysis.json", DEST / "analysis.json")
    rows = analysis["contrast_metrics"]
    sources = [
        {"id": "analysis-json", "label": "Iteration 8 gated analysis", "path": "analysis.json"},
        {"id": "paired-npz", "label": "Paired S50/S25 supervision artifact", "path": "runs/iter8_arc/paired_capacity.npz"},
        query_source(rows, generated),
    ]
    charts = [{
        "id": "contrast-auroc", "title": "Held-out S50-S25 disagreement prediction",
        "subtitle": "600 deterministic held-out MMLU auxiliary-train prompts; higher AUROC is better.",
        "showDescription": True, "intent": "comparison",
        "question": "Does a nonlinear CCD router outperform the matched linear instability router?",
        "rationale": "A zero-based bar comparison makes the small matched-model differences explicit.",
        "comparisonContext": {"grain": "router regime", "unit": "AUROC", "denominator": "600 held-out prompts"},
        "type": "bar", "dataset": "contrast", "sourceId": "contrast-query",
        "encodings": {
            "x": {"field": "method", "type": "nominal", "label": "Router"},
            "y": {"field": "auroc", "type": "quantitative", "format": "number", "label": "Disagreement AUROC"},
            "tooltip": [
                {"field": "method", "type": "text", "label": "Router"},
                {"field": "auroc", "type": "quantitative", "format": "number", "label": "AUROC"},
                {"field": "auprc", "type": "quantitative", "format": "number", "label": "AUPRC"},
            ],
        },
        "palette": {"kind": "categorical", "name": "ccd-comparison"},
        "layout": "full", "maxRows": len(rows), "surface": {"surface": "explorer", "viewMode": "both"},
    }]
    tables = [{
        "id": "contrast-table", "title": "Capacity-contrast router diagnostics",
        "subtitle": "All metrics use the same fixed 600-example held-out split.", "showDescription": True,
        "dataset": "contrast", "sourceId": "contrast-query",
        "defaultSort": {"field": "auroc", "direction": "desc"}, "density": "spacious", "layout": "full",
        "columns": [
            {"field": "method", "label": "Router", "type": "text"},
            {"field": "auroc", "label": "AUROC", "format": "number"},
            {"field": "auprc", "label": "AUPRC", "format": "number"},
            {"field": "precision_at_10", "label": "Precision@10%", "format": "percent"},
            {"field": "precision_at_30", "label": "Precision@30%", "format": "percent"},
            {"field": "best_epoch", "label": "Best epoch", "format": "number"},
        ],
    }]
    gate = analysis["gate_1"]
    selected = gate["selected_regime"].replace("ccd_", "CCD-").title().replace("Ccd", "CCD")
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Capacity-Contrast Distillation: Iteration 8"},
        {"id": "summary", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Technical summary: distillation is learnable, but nonlinear CCD adds no value over a matched linear router\n\n"
         f"**Decision: STOP at Gate 1.** {selected} achieved {gate['selected_auroc']:.4f} held-out AUROC for S50-S25 prediction disagreement, clearing the 0.65 absolute heuristic and substantially exceeding the Iteration 7 linear result of {gate['iteration7_linear_auroc']:.4f}. However, the matched linear router trained on the same 2,400 examples reached {gate['linear_heldout_auroc']:.4f}; the nonlinear result is {gate['selected_auroc'] - gate['linear_heldout_auroc']:+.4f}, not the required clear improvement. The prescribed stop therefore prevents ARC advantage, matched-compute, MMLU-Pro, and runtime claims."},
        {"id": "finding", "type": "markdown", "sourceId": "analysis-json", "body":
         "## More paired data improves predictability, not nonlinear advantage\n\n"
         "All four routers were evaluated on the same deterministic held-out split. The linear and CCD-Hard models are effectively tied, CCD-MultiTask is slightly lower, and CCD-Soft is materially weaker for hard disagreement ranking. This supports paired-capacity disagreement as a learnable single-pass signal, but does not support the proposed nonlinear CCD architecture as the source of the gain."},
        {"id": "chart", "type": "chart", "chartId": "contrast-auroc", "layout": "full"},
        {"id": "table-text", "type": "markdown", "sourceId": "analysis-json", "body":
         "## The hard target is more useful than standalone JS regression\n\n"
         "CCD-Hard nearly matches the linear router, whereas CCD-Soft produces the weakest hard-disagreement AUROC. Multi-task training does not recover the gap. Precision metrics and early-stopping diagnostics below use the identical validation population, so the comparison is paired and denominator-consistent."},
        {"id": "table", "type": "table", "tableId": "contrast-table", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Scope, data, and metric definitions\n\n"
         "The paired corpus contains 3,000 deterministic `cais/mmlu` auxiliary-train prompts; MMLU-Pro is excluded. A class-stratified deterministic split assigns 2,400 prompts to training and 600 to validation with zero ID overlap. Hard contrast is the indicator that S50 and S25 predict different choices. Soft contrast is Jensen-Shannon divergence between their answer distributions. AUROC and AUPRC measure held-out hard-contrast ranking."},
        {"id": "methods", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Model specification and experimental design\n\n"
         "Qwen2.5-1.5B-Instruct uses the frozen nested physical S50 and S25 FFN-channel masks from Iteration 7. The deployment input contains S50 final-prompt-token residual states from blocks 7, 14, 21, and 28 plus entropy, margin, max-probability uncertainty, and logit gap. Each nonlinear router projects every layer to 64 dimensions, then applies 128- and 32-unit GELU layers with dropout 0.2. CCD-Hard uses weighted binary cross-entropy, CCD-Soft uses JS mean-squared error, and CCD-MultiTask combines both. AdamW, weight decay, gradient clipping, and validation-loss early stopping are fixed across regimes."},
        {"id": "limitations", "type": "markdown", "sourceId": "analysis-json", "body":
         "## Limitations, uncertainty, and robustness checks\n\n"
         "This is one deterministic train/validation split and one training seed, not cross-dataset confirmation. The linear comparator uses L14 plus the four scalar features and is stronger than the Iteration 7 ARC-only ridge probe, so the negative result is specifically about nonlinear incremental value under matched supervision. The gate was frozen before the run: AUROC had to reach 0.65 and exceed both linear comparators by at least 0.01. No ARC outcome labels influenced model fitting or Gate 1 selection."},
        {"id": "next", "type": "markdown", "body":
         "## Recommended next steps\n\n"
         "Keep the paired-capacity supervision pipeline, but do not advance the current MLP as a positive CCD result. The next informative experiment is a representation-changing intervention—such as a small adapter trained on capacity contrast—or a pre-registered multi-seed study that tests whether the near tie is stable. Do not run MMLU-Pro or runtime benchmarks for the current frozen method."},
        {"id": "questions", "type": "markdown", "body":
         "## Further questions\n\n"
         "Does the matched linear gain come mainly from the larger dataset, the four scalar features, or MMLU domain structure? Can a capacity-contrast adapter improve held-out AUROC without sacrificing the single-S50-pass deployment constraint? Does hard disagreement remain the best privileged target when the downstream objective is signed dense advantage?"},
    ]
    manifest = {
        "version": 1, "surface": "report", "title": "Capacity-Contrast Distillation: Iteration 8",
        "description": "Held-out evaluation of linear and nonlinear S50-to-S25 disagreement routers.",
        "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks,
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {"contrast": rows}},
        "sources": sources,
    }
    (DEST / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    chart_map = [{
        "section": "Gate 1 contrast prediction",
        "question": "Does nonlinear CCD beat a matched linear router?", "type": "bar",
        "fields": ["method", "auroc", "auprc"],
        "claim": "CCD-Hard clears 0.65 but does not beat the matched linear router.",
        "palette": "categorical; direct x-axis labels; no redundant legend",
    }]
    (DEST / "chart_map.json").write_text(
        json.dumps(chart_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
