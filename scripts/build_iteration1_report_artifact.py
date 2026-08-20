"""Build the canonical portable technical report artifact for Iteration 1."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("runs/iter1_arc")
DEST = ROOT / "report"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.1f} pp"


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
    sql = f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed"
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": sql,
            "description": f"Materialize the reviewed rows for {label.lower()}.",
            "executed_at": generated,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def main() -> int:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    policies = json.loads((ROOT / "policies.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "summary.json", DEST / "summary.json")

    result_source = {
        "id": "iteration-summary",
        "label": "TUAC Iteration 1 reviewed summary",
        "path": "summary.json",
    }
    dataset_source = {
        "id": "arc-dataset",
        "label": "allenai/ai2_arc dataset card",
        "href": "https://huggingface.co/datasets/allenai/ai2_arc",
    }
    teacher_source = {
        "id": "teacher-model",
        "label": "Qwen2.5-1.5B-Instruct model card",
        "href": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
    }
    student_source = {
        "id": "student-model",
        "label": "Qwen2.5-0.5B-Instruct model card",
        "href": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
    }
    proposed = summary["teacher_uncertainty_vs_student"]
    random = summary["teacher_uncertainty_vs_random"]
    seed_rows = summary["seed_diagnostics"]
    mean_rank_rho = sum(row["uncertainty_vs_student_rho"] for row in seed_rows) / len(seed_rows)
    rank_min = min(row["uncertainty_vs_student_rho"] for row in seed_rows)
    rank_max = max(row["uncertainty_vs_student_rho"] for row in seed_rows)
    interaction_values = [row["high_minus_low_4bit_kl"] for row in summary["layer_interaction_rows"]]
    negative_layers = sum(value < 0 for value in interaction_values)

    chart4_rows = [
        row
        for row in summary["accuracy_rows"]
        if row["family"] in {"uniform", "student_only", "teacher_confidence", "teacher_uncertainty", "oracle"}
    ]
    entropy_source = query_source(
        "entropy-perturbation-query",
        "Teacher entropy and 4-bit perturbation rows",
        summary["relation_rows"],
        ["example_id", "teacher_entropy", "mean_4bit_kl", "teacher_correct", "student_correct", "oracle_case"],
        generated,
        ["ARC-Challenge train pool", "750 examples", "mean over 24 one-block 4-bit probes"],
        ["teacher_entropy = normalized categorical entropy", "mean_4bit_kl = mean KL(student FP || one-block 4-bit) over 24 blocks"],
    )
    interaction_source = query_source(
        "layer-interaction-query",
        "Layer interaction rows",
        summary["layer_interaction_rows"],
        ["layer", "layer_index", "high_minus_low_4bit_kl", "high_minus_low_std", "entropy_perturbation_spearman"],
        generated,
        ["five stratified 500-example calibration subsets", "4-bit probes", "upper versus lower teacher-entropy quartile"],
        ["interaction = mean KL in high-entropy quartile minus mean KL in low-entropy quartile"],
    )
    importance_source = query_source(
        "importance-comparison-query",
        "Layer importance comparison rows",
        summary["importance_rows"],
        ["layer", "layer_index", "student_importance", "teacher_uncertainty_importance"],
        generated,
        ["five stratified 500-example calibration subsets", "4-bit empirical importance"],
        ["student_importance = mean 4-bit KL", "teacher_uncertainty_importance = base x (1 + relative entropy interaction)"],
    )
    frontier_source = query_source(
        "accuracy-frontier-query",
        "Held-out accuracy frontier rows",
        chart4_rows,
        ["method", "family", "average_bits", "accuracy", "accuracy_std"],
        generated,
        ["ARC-Challenge validation", "299 examples", "means across five calibration seeds where applicable"],
        ["accuracy = correct multiple-choice predictions / 299", "accuracy_std = sample standard deviation across calibration seeds"],
    )
    metrics_source = query_source(
        "policy-metrics-query",
        "Held-out policy metric rows",
        summary["metric_rows"],
        ["method", "family", "average_bits", "accuracy_mean", "accuracy_std", "ece_mean", "brier_mean", "error_auroc_mean", "aurc_mean", "uq_spearman_mean"],
        generated,
        ["ARC-Challenge validation", "299 examples", "means across five calibration seeds or random seeds where applicable"],
        ["ECE uses 15 equal-width bins", "error AUROC uses normalized predictive entropy", "UQ Spearman compares quantized and FP-student entropy ranks"],
    )
    sources = [
        result_source,
        dataset_source,
        teacher_source,
        student_source,
        entropy_source,
        interaction_source,
        importance_source,
        frontier_source,
        metrics_source,
    ]
    title = "Teacher Uncertainty and Quantization Vulnerability: Iteration 1"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Five-seed ARC-Challenge test of teacher-aware mixed-precision fake quantization.",
        "generatedAt": generated,
        "sources": sources,
        "charts": [
            {
                "id": "entropy-perturbation",
                "title": "Teacher entropy versus mean 4-bit quantization perturbation",
                "subtitle": "ARC-Challenge training pool, n=750; each point is one example averaged over 24 one-block probes.",
                "showDescription": True,
                "intent": "relationship",
                "question": "Are teacher-uncertain examples more vulnerable to one-layer 4-bit quantization?",
                "rationale": "A scatter plot preserves the example-level relationship and reveals dispersion and outliers.",
                "comparisonContext": {"grain": "calibration example", "unit": "KL divergence", "denominator": "750 examples"},
                "type": "scatter",
                "dataset": "entropy_perturbation",
                "sourceId": "entropy-perturbation-query",
                "encodings": {
                    "x": {"field": "teacher_entropy", "type": "quantitative", "format": "number", "label": "Normalized teacher entropy"},
                    "y": {"field": "mean_4bit_kl", "type": "quantitative", "format": "number", "label": "Mean KL perturbation"},
                    "color": {"field": "oracle_case", "type": "nominal", "label": "Teacher correct / student wrong"},
                    "tooltip": [
                        {"field": "example_id", "type": "text", "label": "Example"},
                        {"field": "teacher_entropy", "type": "quantitative", "format": "number", "label": "Teacher entropy"},
                        {"field": "mean_4bit_kl", "type": "quantitative", "format": "number", "label": "Mean 4-bit KL"},
                    ],
                },
                "palette": {"kind": "categorical", "name": "oracle-status"},
                "legend": {"position": "bottom", "title": "Oracle diagnostic case"},
                "layout": "full",
                "maxRows": 750,
                "surface": {"surface": "explorer", "viewMode": "both"},
            },
            {
                "id": "layer-interaction",
                "title": "Teacher-uncertainty interaction by transformer block",
                "subtitle": "Mean across five 500-example calibration subsets; high-entropy minus low-entropy mean 4-bit KL.",
                "showDescription": True,
                "intent": "comparison",
                "question": "Does teacher uncertainty reveal a structured subset of fragile blocks?",
                "rationale": "Ordered bars expose the sign and magnitude of the interaction for all 24 blocks.",
                "comparisonContext": {"grain": "transformer block", "unit": "KL difference", "denominator": "five calibration seeds"},
                "type": "bar",
                "dataset": "layer_interaction",
                "sourceId": "layer-interaction-query",
                "encodings": {
                    "x": {"field": "layer", "type": "ordinal", "label": "Transformer block"},
                    "y": {"field": "high_minus_low_4bit_kl", "type": "quantitative", "format": "number", "label": "High minus low entropy KL"},
                    "tooltip": [
                        {"field": "layer", "type": "text", "label": "Block"},
                        {"field": "high_minus_low_4bit_kl", "type": "quantitative", "format": "number", "label": "Interaction"},
                        {"field": "entropy_perturbation_spearman", "type": "quantitative", "format": "number", "label": "Entropy-KL Spearman"},
                    ],
                },
                "palette": {"kind": "sequential", "name": "blue"},
                "settings": {"orientation": "vertical", "showValues": False, "sort": "custom"},
                "labels": {"values": "none"},
                "layout": "full",
                "maxRows": 24,
                "surface": {"surface": "explorer", "viewMode": "both"},
            },
            {
                "id": "importance-comparison",
                "title": "Student-only versus teacher-uncertainty layer importance",
                "subtitle": "4-bit empirical importance averaged across five calibration seeds; one point per transformer block.",
                "showDescription": True,
                "intent": "relationship",
                "question": "Does the teacher interaction produce a different layer ranking?",
                "rationale": "A layer-level scatter directly compares both importance scores at the same observation grain.",
                "comparisonContext": {"grain": "transformer block", "unit": "empirical error score", "denominator": "24 blocks"},
                "type": "scatter",
                "dataset": "importance_comparison",
                "sourceId": "importance-comparison-query",
                "encodings": {
                    "x": {"field": "student_importance", "type": "quantitative", "format": "number", "label": "Student-only importance"},
                    "y": {"field": "teacher_uncertainty_importance", "type": "quantitative", "format": "number", "label": "Teacher-uncertainty importance"},
                    "label": {"field": "layer", "type": "nominal", "label": "Block"},
                    "tooltip": [
                        {"field": "layer", "type": "text", "label": "Block"},
                        {"field": "student_importance", "type": "quantitative", "format": "number", "label": "Student-only"},
                        {"field": "teacher_uncertainty_importance", "type": "quantitative", "format": "number", "label": "Teacher uncertainty"},
                    ],
                },
                "palette": {"kind": "sequential", "name": "blue"},
                "layout": "full",
                "maxRows": 24,
                "surface": {"surface": "explorer", "viewMode": "both"},
            },
            {
                "id": "accuracy-frontier",
                "title": "Held-out accuracy by policy and average bit budget",
                "subtitle": "Full ARC-Challenge validation, n=299; means across five calibration seeds where applicable.",
                "showDescription": True,
                "intent": "trend",
                "question": "Does teacher-derived allocation improve the accuracy-compression frontier?",
                "rationale": "An ordered-axis line comparison shows the three matched storage operating points.",
                "comparisonContext": {"grain": "policy-budget result", "unit": "accuracy rate", "denominator": "299 validation examples"},
                "type": "line",
                "dataset": "accuracy_frontier",
                "sourceId": "accuracy-frontier-query",
                "encodings": {
                    "x": {"field": "average_bits", "type": "quantitative", "format": "number", "label": "Average bits per block weight"},
                    "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                    "color": {"field": "method", "type": "nominal", "label": "Policy"},
                    "tooltip": [
                        {"field": "method", "type": "text", "label": "Policy"},
                        {"field": "average_bits", "type": "quantitative", "format": "number", "label": "Average bits"},
                        {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                        {"field": "accuracy_std", "type": "quantitative", "format": "percent", "label": "Seed SD"},
                    ],
                },
                "palette": {"kind": "categorical", "name": "policy-frontier"},
                "settings": {"showPoints": True, "curve": "linear", "sort": "xAsc"},
                "legend": {"position": "bottom", "sort": "labelAsc", "title": "Policy"},
                "layout": "full",
                "maxRows": 20,
                "surface": {"surface": "explorer", "viewMode": "both"},
            },
        ],
        "tables": [
            {
                "id": "policy-metrics",
                "title": "Held-out policy metrics",
                "subtitle": "Means across five calibration seeds; lower is better for ECE, Brier, and AURC.",
                "showDescription": True,
                "dataset": "policy_metrics",
                "sourceId": "policy-metrics-query",
                "defaultSort": {"field": "average_bits", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "method", "label": "Policy", "type": "text"},
                    {"field": "average_bits", "label": "Avg bits", "format": "number"},
                    {"field": "accuracy_mean", "label": "Accuracy", "format": "percent"},
                    {"field": "accuracy_std", "label": "Seed SD", "format": "percent"},
                    {"field": "ece_mean", "label": "ECE", "format": "number"},
                    {"field": "brier_mean", "label": "Brier", "format": "number"},
                    {"field": "error_auroc_mean", "label": "Error AUROC", "format": "number"},
                    {"field": "aurc_mean", "label": "AURC", "format": "number"},
                    {"field": "uq_spearman_mean", "label": "FP-UQ rho", "format": "number"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical-summary",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## Technical summary: stop teacher-aware bit allocation\n\n"
                    "**Decision: NO-GO.** Teacher uncertainty is associated with quantization vulnerability, but in the opposite direction from the pilot: normalized teacher entropy and mean example-level 4-bit perturbation have **Spearman rho="
                    f"{summary['global_teacher_entropy_vs_mean_4bit_perturbation_spearman']:.3f}**. The teacher-uncertainty interaction changes the 4-bit layer ranking (mean rho versus student-only **{mean_rank_rho:.3f}**), yet it selects the same all-4-bit policy at the 4.0-bit budget.\n\n"
                    f"At 3.5 and 3.0 bits, teacher uncertainty improves mean accuracy over student-only by **{pp(proposed['3.5']['mean_accuracy_difference'])}** and **{pp(proposed['3.0']['mean_accuracy_difference'])}**, but both paired bootstrap intervals cross zero. The oracle policy is no better: it trails student-only at both mixed budgets. These differences are within random-allocation variance, so the mechanism fails the preregistered go/no-go criteria."
                ),
            },
            {
                "id": "rq1",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## High teacher uncertainty marks less, not more, quantization-sensitive examples\n\n"
                    f"Across 750 calibration-pool examples, teacher entropy has a moderate negative relationship with mean one-block 4-bit KL perturbation (rho={summary['global_teacher_entropy_vs_mean_4bit_perturbation_spearman']:.3f}). "
                    "The point cloud is heterogeneous, but the direction is not the pilot's implied 'uncertain means fragile' story. Oracle cases are shown separately so the reader can see that teacher-correct/student-wrong examples do not form a simple high-entropy vulnerability cluster."
                ),
            },
            {"id": "entropy-chart", "type": "chart", "chartId": "entropy-perturbation", "layout": "full"},
            {
                "id": "rq1-layers",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## The entropy interaction is broad rather than confined to a few fragile blocks\n\n"
                    f"High-entropy examples have lower mean perturbation than low-entropy examples in **{negative_layers} of 24 blocks** at 4 bits. The interaction therefore has structure, but its sign is broadly negative rather than a small positive subset that would clearly deserve extra precision."
                ),
            },
            {"id": "interaction-chart", "type": "chart", "chartId": "layer-interaction", "layout": "full"},
            {
                "id": "rq3",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## The teacher ranking differs, but budget constraints erase much of the difference\n\n"
                    f"Teacher-uncertainty versus student-only ranking rho ranges from **{rank_min:.3f} to {rank_max:.3f}** across the five calibration seeds, satisfying the plan's rho<0.9 novelty threshold. The ranking is reproducible across seeds (minimum pairwise rho **{policies['stability']['teacher_uncertainty']['pairwise_ranking_spearman_min']:.3f}**). "
                    "However, all five 4.0-bit allocations are identical to student-only; only 4-6 of 24 assignments change at 3.5 bits. Ranking novelty does not automatically translate into a materially different deployable policy."
                ),
            },
            {"id": "importance-chart", "type": "chart", "chartId": "importance-comparison", "layout": "full"},
            {
                "id": "outcome",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## Held-out gains are small, uncertain, and matched by random policies\n\n"
                    f"On all 299 ARC-Challenge validation examples, the FP student reaches **{pct(summary['reference_metrics']['full_precision_student']['accuracy'])}** and the teacher **{pct(summary['reference_metrics']['teacher']['accuracy'])}**. At 3.5 bits, teacher uncertainty averages **{pct(next(row['accuracy'] for row in chart4_rows if row['family']=='teacher_uncertainty' and row['average_bits']==3.5))}**, versus **{pct(next(row['accuracy'] for row in chart4_rows if row['family']=='student_only' and row['average_bits']==3.5))}** for student-only; its paired 95% CI is {pp(proposed['3.5']['paired_bootstrap_95_ci'][0])} to {pp(proposed['3.5']['paired_bootstrap_95_ci'][1])}. "
                    f"At 3.0 bits the difference is again {pp(proposed['3.0']['mean_accuracy_difference'])} ({pp(proposed['3.0']['paired_bootstrap_95_ci'][0])} to {pp(proposed['3.0']['paired_bootstrap_95_ci'][1])}). Proposed-versus-random intervals also cross zero at both mixed budgets."
                ),
            },
            {"id": "frontier-chart", "type": "chart", "chartId": "accuracy-frontier", "layout": "full"},
            {
                "id": "metrics-explanation",
                "type": "markdown",
                "body": (
                    "## Reliability metrics do not rescue the teacher policy\n\n"
                    "The exact table includes ECE, Brier score, error-detection AUROC, AURC, and FP-to-quantized uncertainty-rank preservation. No teacher-aware method shows a consistent reliability advantage at matched accuracy across both mixed budgets. Seed standard deviations quantify calibration-subset sensitivity; uniform quantization is available only at 4 bits because the runtime-limited candidate set is {2, 4, 8}."
                ),
            },
            {"id": "metrics-table-block", "type": "table", "tableId": "policy-metrics", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## Scope, data, and metric definitions\n\n"
                    "**Data and splits.** A deterministic 750-example ARC-Challenge training pool supplies five independently seeded, stratified 500-example calibration subsets. The fixed held-out set is the complete 299-example validation split, with disjoint IDs. Every seed contains 81 teacher-correct/student-wrong oracle cases; calibration accuracy is 47.0% for the teacher and 36.6% for the student.\n\n"
                    "**Outcomes.** Accuracy is exact multiple-choice correctness. ECE uses 15 equal-width confidence bins; Brier is multiclass squared probability error; error AUROC uses normalized predictive entropy to detect mistakes; AURC measures risk under uncertainty-based coverage; FP-UQ rho is the Spearman rank preservation of student predictive entropy.\n\n"
                    "**Budget.** Average bits are parameter-weighted over transformer-block matrix weights only. Embeddings, LM head, biases, scales, packing metadata, and actual kernel storage are outside the budget."
                ),
            },
            {
                "id": "method",
                "type": "markdown",
                "body": (
                    "## Experimental design and model specification\n\n"
                    "Qwen2.5-1.5B-Instruct is the offline teacher and Qwen2.5-0.5B-Instruct the student. For every calibration-pool example, the runner caches teacher and FP-student choice probabilities, then fake-quantizes each of 24 student blocks alone at 2, 4, and 8 bits using symmetric group-size-128 weights. The measured error is KL(student FP || one-block quantized).\n\n"
                    "Generic sensitivity is each block/bit's mean KL. Teacher confidence and uncertainty use a bounded relative interaction: base x [1 +/- (high-low)/(high+low)], where high and low are the upper and lower teacher-entropy quartiles. Disagreement uses the same formulation with disagreement quartiles. The diagnostic oracle uses mean perturbation on teacher-correct/student-wrong examples. An exact multiple-choice knapsack assigns {2, 4, 8} bits under 4.0, 3.5, and 3.0 parameter-weighted budgets. Validation labels are used only after all policies are frozen."
                ),
            },
            {
                "id": "robustness",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## Robustness checks support the negative conclusion\n\n"
                    "The calibration teacher-student gap is +10.4 points in every stratified subset, so the test does not repeat the pilot's unrepresentative-gap failure. Each candidate precision is probed empirically rather than extrapolated. Five calibration seeds, five exactly matched random allocations per budget, 5,000 paired bootstrap replicates, and per-seed exact McNemar tests are saved. The decisive negative check is the oracle: it is equal to student-only at 4 bits and worse on average at 3.5 and 3.0 bits."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "iteration-summary",
                "body": (
                    "## Limitations bound the claim to layer-allocation signal\n\n"
                    "This is a strong ARC diagnostic, not a universal compression result. It uses one model family, one multiple-choice benchmark, and fake quantization. The five calibration subsets overlap because they are sampled from a 750-example pool, so seed variance measures allocation sensitivity rather than five independent datasets. The {2, 4, 8} runtime-limited set omits a uniform 3-bit baseline. No packed file size, peak memory, latency, throughput, GPTQ, or AWQ claim is supported."
                ),
            },
            {
                "id": "next-steps",
                "type": "markdown",
                "body": (
                    "## Recommended next step: pivot to delegation drift\n\n"
                    "Stop scaling teacher-aware bit allocation to additional benchmarks or packed backends. The oracle failure means a better entropy proxy alone is unlikely to rescue this allocation mechanism. Redirect the empirical work to **quantization-induced uncertainty distortion and delegation drift in SLM-to-LLM cascades**, where the current results already show large degradation in FP-to-quantized uncertainty-rank preservation at aggressive budgets.\n\n"
                    "1. Define cascade routing decisions and delegation-cost metrics before choosing a quantizer.\n"
                    "2. Measure how 2/4/8-bit perturbations change error-detection AUROC, AURC, and routing thresholds.\n"
                    "3. Compare task accuracy at matched delegation rate and total compute, then add real packed quantization only after the routing signal is established."
                ),
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "Which blocks drive the largest changes in delegation decisions rather than task accuracy? Are routing failures concentrated near confidence thresholds? Does recalibration after quantization recover delegation quality without changing the compressed weights? These questions follow directly from the observed uncertainty distortion and do not require a teacher-aware allocation signal to be useful."
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
                "entropy_perturbation": summary["relation_rows"],
                "layer_interaction": summary["layer_interaction_rows"],
                "importance_comparison": summary["importance_rows"],
                "accuracy_frontier": chart4_rows,
                "policy_metrics": summary["metric_rows"],
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
