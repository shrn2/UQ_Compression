"""Build the canonical Data Analytics report artifact for the ARC pilot."""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("runs/pilot_arc")
DEST = ROOT / "report"

LABELS = {
    "full_precision": "FP student",
    "uniform_4.0": "Uniform 4-bit",
    "student_only_4.0": "Student-only 4.0-bit",
    "combined_4.0": "Proposed combined 4.0-bit",
    "teacher_uncertainty_4.0": "Teacher-uncertainty 4.0-bit",
    "student_only_3.5": "Student-only 3.5-bit",
    "combined_3.5": "Proposed combined 3.5-bit",
    "teacher_uncertainty_3.5": "Teacher-uncertainty 3.5-bit",
    "uniform_3.0": "Uniform 3-bit",
}


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(float(value)) if isinstance(value, float) else str(value)


def materialize(rows: list[dict], columns: list[str], order_by: str) -> tuple[str, list[dict]]:
    values = ",\n    ".join(
        "(" + ", ".join(sql_literal(row.get(column)) for column in columns) + ")" for row in rows
    )
    sql = f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed ORDER BY {order_by}"
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    materialized = [dict(row) for row in connection.execute(sql)]
    connection.close()
    return sql, materialized


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "summary.json", DEST / "summary.json")

    rows = []
    chart_rows = []
    for source in summary["main_table"]:
        row = {
            "method": LABELS[source["name"]],
            "budget": source["actual_average_bits"],
            "budget_label": f"{source['actual_average_bits']:.1f}",
            "accuracy": source["accuracy"],
            "ece": source["ece"],
            "brier": source["brier"],
            "error_auroc": source["error_auroc"],
            "aurc": source["aurc"],
            "uq_spearman": source["student_uncertainty_spearman"],
            "accuracy_ci_low": source["accuracy_wilson_95_ci"][0],
            "accuracy_ci_high": source["accuracy_wilson_95_ci"][1],
        }
        rows.append(row)
        if source["name"] != "full_precision":
            chart_rows.append(row)
    for budget, values in summary["random_summaries"].items():
        if budget in ("4.0", "3.5"):
            chart_rows.append(
                {
                    "method": "Random mixed precision (mean)",
                    "budget": float(budget),
                    "budget_label": budget,
                    "accuracy": values["accuracy"]["mean"],
                    "accuracy_std": values["accuracy"]["std"],
                }
            )

    table_columns = [
        "method", "budget", "budget_label", "accuracy", "ece", "brier", "error_auroc",
        "aurc", "uq_spearman", "accuracy_ci_low", "accuracy_ci_high",
    ]
    chart_columns = ["method", "budget", "budget_label", "accuracy", "accuracy_std"]
    table_sql, rows = materialize(rows, table_columns, "budget DESC, method ASC")
    chart_sql, chart_rows = materialize(chart_rows, chart_columns, "budget_label ASC, method ASC")

    result_source = {
        "id": "pilot-results",
        "label": "TUAC ARC pilot summary",
        "path": "summary.json",
    }
    arc_source = {
        "id": "arc-dataset",
        "label": "allenai/ai2_arc dataset card",
        "href": "https://huggingface.co/datasets/allenai/ai2_arc",
    }
    student_source = {
        "id": "student-model",
        "label": "Qwen2.5-0.5B-Instruct model card",
        "href": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
    }
    teacher_source = {
        "id": "teacher-model",
        "label": "Qwen2.5-1.5B-Instruct model card",
        "href": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
    }
    chart_source = {
        "id": "accuracy-chart-results",
        "label": "Reviewed policy accuracy rows",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": chart_sql,
            "description": "Materialize the reviewed held-out accuracy rows used in the policy comparison chart.",
            "executed_at": generated,
            "filters": ["ARC-Challenge validation", "64 examples", "Three random-policy seeds where shown"],
            "metric_definitions": ["accuracy = correct multiple-choice predictions / 64"],
        },
    }
    table_source = {
        "id": "policy-table-results",
        "label": "Reviewed policy metric rows",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": table_sql,
            "description": "Materialize the reviewed held-out policy metric rows shown in the exact-value table.",
            "executed_at": generated,
            "filters": ["ARC-Challenge validation", "64 examples", "Selected primary policies and ablations"],
            "metric_definitions": [
                "accuracy = correct predictions / 64",
                "ECE uses 15 equal-width confidence bins",
                "error AUROC uses normalized predictive entropy to detect incorrect predictions",
                "UQ Spearman compares quantized and FP-student entropy ranks",
            ],
        },
    }

    title = "Teacher-Uncertainty-Aware Compression: ARC Pilot"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Technical pilot of teacher-aware mixed-precision fake quantization on ARC-Challenge.",
        "generatedAt": generated,
        "sources": [result_source, arc_source, student_source, teacher_source, chart_source, table_source],
        "charts": [
            {
                "id": "accuracy-by-budget",
                "title": "Held-out accuracy by quantization policy and bit budget",
                "subtitle": "ARC-Challenge validation, n=64; grouped bars show observed accuracy, not a population estimate.",
                "showDescription": True,
                "intent": "comparison",
                "question": "Which policy preserves held-out task accuracy at matched average bit budgets?",
                "rationale": "Grouped bars support direct policy comparison at the two non-degenerate mixed budgets and the 3-bit floor.",
                "comparisonContext": {
                    "baseline": "FP student accuracy: 31.25%",
                    "denominator": "64 held-out validation examples",
                    "grain": "policy-budget result",
                    "unit": "accuracy rate",
                },
                "type": "bar",
                "dataset": "accuracy_results",
                "sourceId": "accuracy-chart-results",
                "encodings": {
                    "x": {"field": "budget_label", "type": "ordinal", "label": "Average bits per matrix weight"},
                    "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                    "color": {"field": "method", "type": "nominal", "label": "Policy"},
                    "tooltip": [
                        {"field": "method", "type": "text", "label": "Policy"},
                        {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                    ],
                },
                "palette": {"kind": "categorical", "name": "tuac-policies"},
                "settings": {"groupMode": "grouped", "orientation": "vertical", "showValues": True, "sort": "custom"},
                "legend": {"position": "bottom", "sort": "labelAsc", "title": "Policy"},
                "labels": {"values": "all"},
                "layout": "full",
                "maxRows": 20,
                "surface": {"surface": "explorer", "viewMode": "both"},
            }
        ],
        "tables": [
            {
                "id": "metrics-table",
                "title": "Held-out policy metrics",
                "subtitle": "Observed metrics on 64 ARC-Challenge validation examples; lower is better for ECE, Brier, and AURC.",
                "showDescription": True,
                "dataset": "policy_metrics",
                "sourceId": "policy-table-results",
                "defaultSort": {"field": "budget", "direction": "desc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "method", "label": "Policy", "type": "text"},
                    {"field": "budget", "label": "Avg bits", "format": "number"},
                    {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                    {"field": "ece", "label": "ECE", "format": "number"},
                    {"field": "brier", "label": "Brier", "format": "number"},
                    {"field": "error_auroc", "label": "Error AUROC", "format": "number"},
                    {"field": "uq_spearman", "label": "UQ Spearman", "format": "number"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical-summary",
                "type": "markdown",
                "sourceId": "pilot-results",
                "body": (
                    "## The proposed confident-disagreement score was not supported in this pilot\n\n"
                    "On 64 held-out ARC-Challenge validation questions, the proposed combined policy reached **32.8% at 4.0 bits** and **28.1% at 3.5 bits**. "
                    "Student-only sensitivity reached **35.9%** and **29.7%**, respectively. The paired differences were small and uncertain: −3.1 percentage points at 4.0 bits "
                    "(bootstrap 95% CI −12.5 to +6.3) and −1.6 points at 3.5 bits (−10.9 to +7.8).\n\n"
                    "The strongest ablation was the opposite weighting direction—**teacher uncertainty**—at **40.6% (4.0 bits)** and **34.4% (3.5 bits)**. "
                    "Uniform 4-bit also reached 40.6%. With only 32 calibration and 64 evaluation examples, these are directional pilot findings, not evidence of superiority."
                ),
            },
            {
                "id": "key-finding",
                "type": "markdown",
                "sourceId": "pilot-results",
                "body": (
                    "## Policy choice mattered, but not in the hypothesized direction\n\n"
                    "The chart compares observed held-out accuracy at each feasible budget. At 4.0 bits, both teacher-uncertainty and uniform quantization led the proposed policy by 7.8 points. "
                    "At 3.5 bits, teacher-uncertainty led the proposed policy by 6.3 points. Random mixed precision averaged 27.1% at 4.0 bits and 30.7% at 3.5 bits across three seeds, showing substantial policy variance."
                ),
            },
            {"id": "accuracy-chart", "type": "chart", "chartId": "accuracy-by-budget", "layout": "full"},
            {
                "id": "paired-evidence",
                "type": "markdown",
                "sourceId": "pilot-results",
                "body": (
                    "## The sample is too small to distinguish close policies reliably\n\n"
                    "Combined versus student-only had 10 discordant predictions at 4.0 bits (4 helped, 6 hurt; exact McNemar p=0.754) and 9 at 3.5 bits (4 helped, 5 hurt; p=1.000). "
                    "Teacher-uncertainty versus combined improved observed accuracy by 7.8 points at 4.0 bits, but its paired bootstrap interval still crossed zero (−3.1 to +18.8 points). "
                    "Exact metrics are shown below; Wilson intervals for individual 64-example accuracies are roughly 22–24 points wide."
                ),
            },
            {"id": "metrics-table-block", "type": "table", "tableId": "metrics-table", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "pilot-results",
                "body": (
                    "## Scope and metric definitions\n\n"
                    "**Models.** Qwen2.5-1.5B-Instruct was the offline teacher and Qwen2.5-0.5B-Instruct the student. On the held-out split, teacher accuracy was 53.1% versus 31.3% for the FP student, confirming a real capability gap.\n\n"
                    "**Data.** Allocation used 32 deterministic ARC-Challenge training examples. Evaluation used 64 disjoint ARC-Challenge validation examples. Accuracy is multiple-choice exact correctness; ECE uses 15 bins; error AUROC uses normalized predictive entropy to detect mistakes; UQ Spearman compares quantized and FP-student entropy rankings.\n\n"
                    "**Budget.** Average bits are parameter-weighted over fake-quantized transformer-block matrix weights. Embeddings, the LM head, biases, scale metadata, and real packing overhead are outside this budget."
                ),
            },
            {
                "id": "method",
                "type": "markdown",
                "body": (
                    "## Experimental design and implementation\n\n"
                    "For every calibration example, the runner computed teacher and FP-student choice distributions. Each of 24 student blocks was then fake-quantized alone to 4 bits (symmetric, group size 128), and its output KL perturbation was measured. "
                    "The proposed importance was teacher confidence × KL(teacher || student) × block perturbation; the student-only baseline omitted teacher weights. A multiple-choice knapsack assigned widths from {3, 4, 8} under exact parameter-weighted caps.\n\n"
                    "The same cached student snapshot and scoring code evaluated FP, uniform, three random mixed-precision seeds, all signal ablations, student-only, and proposed policies. Train examples alone determined policies; validation labels were read only after allocation."
                ),
            },
            {
                "id": "diagnostics",
                "type": "markdown",
                "sourceId": "pilot-results",
                "body": (
                    "## High ranking overlap limits the teacher signal's novelty\n\n"
                    "Student-only and combined block importance had **Spearman ρ=0.947**. Their allocations differed in only **5 of 24 blocks** at both 4.0 and 3.5 bits. "
                    "Combined and disagreement-only produced identical allocations, so confidence multiplication did not change the selected policy. At 3.0 bits every method collapsed to 24 three-bit blocks because 3 bits was the minimum choice; that operating point cannot test allocation quality."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "pilot-results",
                "body": (
                    "## Limitations prevent a paper-level conclusion\n\n"
                    "This run is **shareable only as a pilot with caveats**. The calibration set is 16–31× smaller than the planned minimum, the evaluation set has only 64 questions, and only ARC-Challenge was tested. "
                    "The teacher exceeded the student by just 3.1 points on the calibration subset despite a 21.9-point held-out gap, making calibration weights unstable. The unexpected gains from uniform 4-bit and uniform 3-bit over FP are plausible sample/noise effects and should not be interpreted as quantization improving the underlying model.\n\n"
                    "Fake quantization retains floating-point storage and does not establish packed size, latency, peak-memory, or throughput gains. GPTQ, AWQ, GSM8K, MMLU-Pro, GPQA, and an existing mixed-precision backend were not run."
                ),
            },
            {
                "id": "next-steps",
                "type": "markdown",
                "body": (
                    "## Recommended next experiment\n\n"
                    "1. Increase teacher-aware calibration to at least 500 stratified train/development examples and evaluate the complete 299-example ARC validation split.\n"
                    "2. Add a candidate width below 3 bits or remove the 3.0-bit point; the current floor makes all policies identical there.\n"
                    "3. Measure block perturbation separately at 3, 4, and 8 bits instead of extrapolating a 4-bit probe with a theoretical error curve.\n"
                    "4. Run MMLU-Pro and GSM8K, plus true GPTQ/AWQ and packed-kernel baselines at matched whole-model storage.\n"
                    "5. Treat teacher-uncertainty weighting as a competing hypothesis, preregister both score directions, and repeat across calibration seeds."
                ),
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "body": (
                    "## Questions the pilot leaves open\n\n"
                    "Does teacher confidence become useful only when calibration examples include enough cases where the teacher is confidently correct and the student is wrong? "
                    "Does uncertainty weighting protect broadly fragile blocks while the proposed score overconcentrates on a few high-weight examples? "
                    "And do the results persist when per-width empirical errors replace the current single-probe approximation?"
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
            "datasets": {"accuracy_results": chart_rows, "policy_metrics": rows},
        },
        "sources": [result_source, arc_source, student_source, teacher_source, chart_source, table_source],
    }
    (DEST / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
