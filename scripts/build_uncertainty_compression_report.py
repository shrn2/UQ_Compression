"""Build the canonical portable report for the uncertainty-compression program."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("runs/iter11_upgd")
DEST = ROOT / "report"


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def query_source(source_id: str, label: str, rows: list[dict], columns: list[str], generated: str,
                 filters: list[str], definitions: list[str]) -> dict:
    values = ",\n    ".join(
        "(" + ", ".join(literal(row.get(column)) for column in columns) + ")" for row in rows
    )
    return {"id": source_id, "label": label, "query": {
        "engine": "SQLite", "language": "sql",
        "sql": f"WITH reviewed({', '.join(columns)}) AS (\n  VALUES\n    {values}\n)\nSELECT * FROM reviewed",
        "description": f"Materialize reviewed {label.lower()} rows.", "executed_at": generated,
        "filters": filters, "metric_definitions": definitions,
    }}


def row_lookup(analysis: dict) -> dict[str, dict]:
    return {row["condition"]: row for row in analysis["metric_rows"]}


def main() -> int:
    development = json.loads((ROOT / "development_analysis.json").read_text(encoding="utf-8"))
    external = json.loads((ROOT / "mmlu_pro_analysis.json").read_text(encoding="utf-8"))
    training = json.loads((ROOT / "training.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "validation.json").read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).isoformat()
    DEST.mkdir(parents=True, exist_ok=True)
    for filename in ("development_analysis.json", "mmlu_pro_analysis.json", "training.json", "validation.json"):
        shutil.copyfile(ROOT / filename, DEST / filename)

    dev = row_lookup(development)
    ext = row_lookup(external)
    accuracy_rows = []
    metric_rows = []
    for dataset, lookup, n in (("MMLU auxiliary validation", dev, 600), ("MMLU-Pro exploratory", ext, 1000)):
        for condition, label in (("dense", "Dense"), ("baseline_aggressive", "One-shot baseline"),
                                 ("upgd_aggressive", "UPGD")):
            accuracy_rows.append({"dataset": dataset, "condition": label,
                                  "accuracy": lookup[condition]["accuracy"], "examples": n})
        for condition, label in (("baseline_aggressive", "One-shot baseline"),
                                 ("upgd_aggressive", "UPGD")):
            metric_rows.append({
                "dataset": dataset, "condition": label, "examples": n,
                "accuracy": lookup[condition]["accuracy"],
                "mean_kl_from_dense": lookup[condition]["mean_kl_from_dense"],
                "prediction_agreement_with_dense": lookup[condition]["prediction_agreement_with_dense"],
                "entropy_spearman_with_dense": lookup[condition]["entropy_spearman_with_dense"],
                "aurc": lookup[condition]["aurc"], "ece": lookup[condition]["ece"],
            })

    fidelity_rows = []
    for metric, field in (("Dense prediction agreement", "prediction_agreement_with_dense"),
                          ("Dense entropy-rank Spearman", "entropy_spearman_with_dense")):
        for condition, label in (("baseline_aggressive", "One-shot baseline"),
                                 ("upgd_aggressive", "UPGD")):
            fidelity_rows.append({"metric": metric, "condition": label, "value": dev[condition][field],
                                  "examples": 600})

    stratum_rows = []
    for dataset, analysis in (("MMLU auxiliary validation", development), ("MMLU-Pro exploratory", external)):
        by_key = {(row["condition"], row["stratum"]): row for row in analysis["stratum_rows"]}
        for stratum, label in enumerate(("Low dense entropy", "Middle dense entropy", "High dense entropy")):
            difference = (by_key[("upgd_aggressive", stratum)]["accuracy"] -
                          by_key[("baseline_aggressive", stratum)]["accuracy"])
            stratum_rows.append({"dataset": dataset, "stratum": label, "accuracy_difference": difference,
                                 "examples": by_key[("upgd_aggressive", stratum)]["examples"]})

    history_rows = [
        {"iteration": 1, "approach": "Teacher-aware mixed precision", "result": "NO-GO", "implication": "Teacher entropy was not a compression-vulnerability proxy."},
        {"iteration": 2, "approach": "Trust-aware layer precision", "result": "NO-GO", "implication": "Trust and capability rankings were nearly collinear."},
        {"iteration": 3, "approach": "Quantized uncertainty routing", "result": "Mixed", "implication": "Compression changed rankings, but routing-only regret was not established."},
        {"iteration": 4, "approach": "Cascade-aware local allocation", "result": "NO-GO", "implication": "Local cascade sensitivity did not compose into a better model."},
        {"iteration": 5, "approach": "Constrained joint policy search", "result": "NO-GO", "implication": "Calibration headroom did not survive held-out evaluation."},
        {"iteration": 6, "approach": "Progressive structured pruning", "result": "NO-GO", "implication": "A compute hierarchy existed, but uncertainty did not route it reliably."},
        {"iteration": 7, "approach": "Counterfactual capacity routing", "result": "NO-GO", "implication": "Sparse hidden states underperformed generic uncertainty."},
        {"iteration": 8, "approach": "Capacity-contrast distillation", "result": "NO-GO", "implication": "Disagreement was learnable; nonlinear modeling added no value."},
        {"iteration": 9, "approach": "Frozen capacity-contrast routing", "result": "NO-GO", "implication": "Ranking transfer did not improve matched-compute routing."},
        {"iteration": 10, "approach": "Uncertainty-robust local pruning", "result": "NO-GO", "implication": "Entropy strata barely changed masks and reduced accuracy."},
        {"iteration": 11, "approach": "Joint stochastic gate distillation", "result": "Promising / gate fail", "implication": "Aggregate pruning frontier improved; conditional robustness remains unproven."},
    ]
    paired_rows = [
        {"dataset": "MMLU auxiliary validation", "examples": 600,
         "accuracy_difference": development["paired_comparison"]["difference"],
         "ci_low": development["paired_comparison"]["bootstrap_95_ci"][0],
         "ci_high": development["paired_comparison"]["bootstrap_95_ci"][1],
         "replicates": development["paired_comparison"]["replicates"]},
        {"dataset": "MMLU-Pro exploratory", "examples": 1000,
         "accuracy_difference": external["paired_comparison"]["difference"],
         "ci_low": external["paired_comparison"]["bootstrap_95_ci"][0],
         "ci_high": external["paired_comparison"]["bootstrap_95_ci"][1],
         "replicates": external["paired_comparison"]["replicates"]},
    ]

    sources = [
        {"id": "development-json", "label": "UPGD MMLU auxiliary development analysis", "path": "development_analysis.json"},
        {"id": "external-json", "label": "UPGD MMLU-Pro exploratory analysis", "path": "mmlu_pro_analysis.json"},
        {"id": "training-json", "label": "Frozen UPGD training record", "path": "training.json"},
        {"id": "validation-json", "label": "Independent Iteration 11 validation", "path": "validation.json"},
        query_source("accuracy-query", "dense, one-shot, and UPGD accuracy", accuracy_rows,
                     ["dataset", "condition", "accuracy", "examples"], generated,
                     ["same examples within dataset", "50% FFN pruning for compressed conditions"],
                     ["accuracy is exact-match multiple-choice accuracy"]),
        query_source("metric-query", "compressed-model metric comparison", metric_rows,
                     ["dataset", "condition", "examples", "accuracy", "mean_kl_from_dense",
                      "prediction_agreement_with_dense", "entropy_spearman_with_dense", "aurc", "ece"], generated,
                     ["physical 50% FFN pruning", "one frozen UPGD run"],
                     ["KL is dense-to-compressed answer-distribution divergence", "lower AURC and ECE are better"]),
        query_source("fidelity-query", "development uncertainty fidelity", fidelity_rows,
                     ["metric", "condition", "value", "examples"], generated,
                     ["600 fixed MMLU auxiliary validation examples"],
                     ["agreement compares argmax predictions", "Spearman compares normalized answer-entropy ranks"]),
        query_source("stratum-query", "UPGD accuracy advantage by dense-entropy stratum", stratum_rows,
                     ["dataset", "stratum", "accuracy_difference", "examples"], generated,
                     ["three equal-frequency strata defined independently within each dataset"],
                     ["difference is UPGD accuracy minus one-shot baseline accuracy"]),
        query_source("history-query", "eleven-iteration evidence map", history_rows,
                     ["iteration", "approach", "result", "implication"], generated,
                     ["repository Iterations 1 through 11"],
                     ["result labels follow each iteration's frozen gate"]),
        query_source("paired-query", "paired UPGD accuracy comparisons", paired_rows,
                     ["dataset", "examples", "accuracy_difference", "ci_low", "ci_high", "replicates"], generated,
                     ["UPGD and one-shot baseline scored on identical examples", "5,000 paired bootstrap replicates"],
                     ["difference is UPGD accuracy minus one-shot baseline accuracy"]),
    ]

    charts = [
        {"id": "accuracy", "title": "Accuracy at 50% FFN pruning",
         "subtitle": "Physical pruning on fixed 600-example development and 1,000-example exploratory cohorts.",
         "showDescription": True, "intent": "comparison",
         "question": "Does joint gate learning improve aggregate accuracy over one-shot pruning?",
         "rationale": "Grouped bars compare the three model conditions on the same examples within each dataset.",
         "comparisonContext": {"grain": "dataset and model condition", "unit": "accuracy", "denominator": "600 or 1,000 examples"},
         "type": "bar", "dataset": "accuracy", "sourceId": "accuracy-query",
         "encodings": {"x": {"field": "dataset", "type": "nominal", "label": "Evaluation cohort"},
                       "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                       "color": {"field": "condition", "type": "nominal", "label": "Model"},
                       "tooltip": [{"field": "condition", "type": "text", "label": "Model"},
                                   {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                                   {"field": "examples", "type": "quantitative", "format": "number", "label": "Examples"}]},
         "palette": {"kind": "categorical", "name": "model-comparison"},
         "legend": {"position": "bottom", "title": "Model"}, "layout": "full", "maxRows": 6,
         "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "fidelity", "title": "Development uncertainty fidelity",
         "subtitle": "Two same-scale proportions on the fixed 600-example MMLU auxiliary validation split.",
         "showDescription": True, "intent": "comparison",
         "question": "Does UPGD better preserve dense predictions and uncertainty ordering?",
         "rationale": "Grouped bars compare two bounded fidelity measures without mixing incompatible units.",
         "comparisonContext": {"grain": "metric and compressed model", "unit": "proportion", "denominator": "600 examples"},
         "type": "bar", "dataset": "fidelity", "sourceId": "fidelity-query",
         "encodings": {"x": {"field": "metric", "type": "nominal", "label": "Fidelity measure"},
                       "y": {"field": "value", "type": "quantitative", "format": "number", "label": "Value"},
                       "color": {"field": "condition", "type": "nominal", "label": "Model"},
                       "tooltip": [{"field": "condition", "type": "text", "label": "Model"},
                                   {"field": "value", "type": "quantitative", "format": "number", "label": "Value"}]},
         "palette": {"kind": "categorical", "name": "fidelity-comparison"},
         "legend": {"position": "bottom", "title": "Model"}, "layout": "full", "maxRows": 4,
         "surface": {"surface": "explorer", "viewMode": "both"}},
        {"id": "strata", "title": "UPGD accuracy difference by dense-entropy stratum",
         "subtitle": "Signed percentage-point effect versus one-shot pruning; strata are dataset-specific thirds.",
         "showDescription": True, "intent": "comparison",
         "question": "Is the aggregate gain robust across predictive-uncertainty strata?",
         "rationale": "Signed grouped bars expose losses and the cross-dataset reversal around a common zero reference.",
         "comparisonContext": {"grain": "dataset and dense-entropy stratum", "unit": "accuracy difference", "denominator": "about one third of each cohort"},
         "type": "bar", "dataset": "strata", "sourceId": "stratum-query",
         "encodings": {"x": {"field": "stratum", "type": "nominal", "label": "Dense predictive-entropy stratum"},
                       "y": {"field": "accuracy_difference", "type": "quantitative", "format": "percent", "label": "UPGD minus baseline accuracy"},
                       "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                       "tooltip": [{"field": "dataset", "type": "text", "label": "Dataset"},
                                   {"field": "accuracy_difference", "type": "quantitative", "format": "percent", "label": "Accuracy difference"},
                                   {"field": "examples", "type": "quantitative", "format": "number", "label": "Examples"}]},
         "palette": {"kind": "categorical", "name": "dataset-comparison"},
         "legend": {"position": "bottom", "title": "Dataset"}, "layout": "full", "maxRows": 6,
         "surface": {"surface": "explorer", "viewMode": "both"}},
    ]

    tables = [
        {"id": "metrics", "title": "Compressed-model metrics",
         "subtitle": "Exact metrics for both 50%-pruned models; lower KL, AURC, and ECE are better.",
         "showDescription": True, "dataset": "metrics", "sourceId": "metric-query", "density": "compact", "layout": "full",
         "columns": [{"field": "dataset", "label": "Dataset"}, {"field": "condition", "label": "Model"},
                     {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                     {"field": "mean_kl_from_dense", "label": "KL from dense", "format": "number"},
                     {"field": "prediction_agreement_with_dense", "label": "Dense agreement", "format": "percent"},
                     {"field": "entropy_spearman_with_dense", "label": "Entropy rho", "format": "number"},
                     {"field": "aurc", "label": "AURC", "format": "number"},
                     {"field": "ece", "label": "ECE", "format": "number"}]},
        {"id": "history", "title": "Evidence map across eleven iterations",
         "subtitle": "Each row records the frozen-gate outcome and the design implication used by the next iteration.",
         "showDescription": True, "dataset": "history", "sourceId": "history-query", "density": "compact", "layout": "full",
         "columns": [{"field": "iteration", "label": "Iter."}, {"field": "approach", "label": "Approach"},
                     {"field": "result", "label": "Result"}, {"field": "implication", "label": "Implication"}]},
    ]

    dev_diff = development["paired_comparison"]
    ext_diff = external["paired_comparison"]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Uncertainty-Preserving Model Compression: Evidence and Method"},
        {"id": "summary", "type": "markdown", "sourceId": "paired-query", "body":
         "## Technical summary\n\n"
         "The strongest method in this repository is **Uncertainty-Preserving Gate Distillation (UPGD)**: learn exact-budget FFN channel gates jointly through the complete frozen model, using dense answer distributions, entropy, margin, stochastic subnetworks, and a smooth worst-stratum loss; then export one deterministic physically pruned model. At 50% FFN channel pruning, UPGD improves the matched one-shot baseline by "
         f"{dev_diff['difference']*100:.1f} points on 600 held-out MMLU auxiliary examples (95% paired bootstrap interval [{dev_diff['bootstrap_95_ci'][0]*100:.1f}, {dev_diff['bootstrap_95_ci'][1]*100:.1f}]). The 1,000-example MMLU-Pro transfer is directionally positive at {ext_diff['difference']*100:+.1f} points but unresolved ([{ext_diff['bootstrap_95_ci'][0]*100:.1f}, {ext_diff['bootstrap_95_ci'][1]*100:.1f}]). The conservative uncertainty-robustness gate fails because one entropy stratum loses in each dataset and the affected stratum reverses. The warranted claim is a promising aggregate compression improvement, not proven conditional robustness or deployment speed."},
        {"id": "accuracy-text", "type": "markdown", "sourceId": "development-json", "body":
         "## Key finding: joint optimization materially beats one-shot pruning\n\n"
         "On development data, UPGD reaches 41.83% accuracy versus 34.33% for the matched baseline, while the dense model reaches 68.17%. On MMLU-Pro, the corresponding figures are 13.20%, 11.30%, and 26.70%. UPGD therefore recovers meaningful capability at an identical physical channel budget, although the external interval still includes zero."},
        {"id": "accuracy-chart", "type": "chart", "chartId": "accuracy", "layout": "full"},
        {"id": "fidelity-text", "type": "markdown", "sourceId": "development-json", "body":
         "## Key finding: dense behavior is substantially better preserved in development\n\n"
         "Relative to one-shot pruning, UPGD reduces dense-to-compressed KL from 1.898 to 1.017, increases dense prediction agreement from 34.33% to 48.00%, improves dense entropy-rank Spearman from 0.158 to 0.395, reduces AURC from 0.617 to 0.439, and reduces ECE from 0.396 to 0.257. On MMLU-Pro, KL and ECE improve but entropy-rank correlation is slightly worse, so uncertainty preservation is not uniformly portable."},
        {"id": "fidelity-chart", "type": "chart", "chartId": "fidelity", "layout": "full"},
        {"id": "strata-text", "type": "markdown", "sourceId": "external-json", "body":
         "## Key caveat: uncertainty-stratum effects reverse across datasets\n\n"
         "UPGD gains 18.0 and 7.5 points in the low- and middle-entropy development strata but loses 3.0 points in the high-entropy stratum. On MMLU-Pro it loses 2.40 points in the low-entropy stratum and gains 1.80 and 6.31 points in the middle and high strata. This is direct evidence against claiming that the current robust term reliably protects a particular uncertainty region."},
        {"id": "strata-chart", "type": "chart", "chartId": "strata", "layout": "full"},
        {"id": "metric-table-block", "type": "table", "tableId": "metrics", "layout": "full"},
        {"id": "history-text", "type": "markdown", "sourceId": "history-query", "body":
         "## What the full experiment history establishes\n\n"
         "Iterations 1-10 repeatedly show that predictive uncertainty is not a reliable local importance weight or a dependable per-example densification rule. Teacher uncertainty can be anticorrelated with perturbation, trust and capability layer rankings collapse together, local effects are non-additive, and learnable capacity disagreement does not necessarily produce useful routing. Iteration 11 succeeds where those methods fail because it changes the optimization level: all channel decisions are trained jointly through the whole frozen network, and uncertainty is a dense-behavior target rather than a rule that simply allocates more resources to uncertain cases."},
        {"id": "history-table-block", "type": "table", "tableId": "history", "layout": "full"},
        {"id": "literature", "type": "markdown", "body":
         "## Position relative to current literature\n\n"
         "The design combines ideas that the literature usually treats separately. [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SparseGPT](https://arxiv.org/abs/2301.00774), and [Wanda](https://arxiv.org/abs/2306.11695) primarily preserve outputs or activations under local or approximate reconstruction criteria. [Self-calibration](https://arxiv.org/abs/2410.17170) and [calibration-data curation](https://arxiv.org/abs/2510.10618) make the calibration signal itself more task-relevant. [BayesQ](https://arxiv.org/abs/2511.08821) treats weight uncertainty as a precision-allocation signal, while [UNComp](https://arxiv.org/abs/2410.03090) uses representation entropy for runtime state compression. A recent [compression-and-conformal benchmark](https://arxiv.org/abs/2606.01850) reports that accuracy and uncertainty can decouple after compression. UPGD's distinct hypothesis is that predictive uncertainty should be preserved as part of a joint exact-budget distillation objective, not interpreted monotonically as importance."},
        {"id": "scope", "type": "markdown", "sourceId": "training-json", "body":
         "## Scope, data, and metric definitions\n\n"
         "The model is Qwen2.5-1.5B-Instruct. Training uses fixed 2,400 MMLU auxiliary training examples; development uses 600 disjoint fixed validation examples; external exploration uses 1,000 MMLU-Pro examples. Every compressed model retains exactly 4,480 of 8,960 FFN channels in each of 28 transformer layers. Accuracy is exact-choice accuracy. KL compares dense and compressed answer distributions. Entropy Spearman compares normalized predictive-entropy ranks. AURC measures risk-coverage quality; ECE measures calibration error. Split-conformal LAC diagnostics use one deterministic calibration/evaluation split and establish marginal, not conditional, coverage."},
        {"id": "method", "type": "markdown", "sourceId": "training-json", "body":
         "## Methodology\n\n"
         "For each layer and FFN channel, UPGD learns one gate logit. Each forward pass adds annealed logistic noise and selects the exact top 4,480 channels. A binary forward mask enforces the deployment budget; a sigmoid straight-through relaxation supplies gradients. All model weights remain frozen. The per-example objective is dense-to-gated KL plus squared entropy error plus 0.5 times squared margin error. Batches contain one prompt from each of three dense-entropy strata, and the mean loss is augmented by 0.5 times a smooth worst-stratum loss. Temperature anneals from 0.5 to 0.1 and noise from 0.5 to 0.05 over 600 Adam steps. Export removes the unselected `gate_proj`, `up_proj`, and `down_proj` channels; gates and the dense model are not needed at inference."},
        {"id": "resources", "type": "markdown", "sourceId": "training-json", "body":
         "## Resource fit\n\n"
         f"Only 250,880 gate logits and their Adam state are optimized. The frozen 1.5B model remains in BF16 and batches contain three prompts. The completed run took {training['training_seconds']/60:.1f} minutes on the available 8 GB GPU and stayed within the 16 GB system-RAM constraint. The exported mask differs materially from the initialization: retained-channel overlap is {training['retained_overlap_with_initial_baseline']:.2%}."},
        {"id": "limitations", "type": "markdown", "sourceId": "validation-json", "body":
         "## Limitations, uncertainty, and robustness checks\n\n"
         f"Independent artifact validation passes {validation['passed']}/{validation['total']} checks and the Python test suite passes 49 tests. Statistical uncertainty remains substantial: only one model family, one gate-training seed, one development split, and one exploratory transfer cohort were run. MMLU-Pro was used earlier in the broader research program, so it is not a clean confirmation set. The 50% masks physically prune FFN channels and imply a transformer matrix-multiply proxy of 0.5588, but no packed checkpoint, peak-memory measurement, latency, throughput, energy, or kernel benchmark was produced. Conformal intervals are marginal and use one split. The method should not yet be represented as robust to uncertainty strata or as deployment-ready."},
        {"id": "next", "type": "markdown", "body":
         "## Recommended next steps\n\n"
         "1. Freeze the current UPGD implementation and evaluate at least three training seeds on a fresh benchmark not used anywhere in Iterations 1-11.\n2. Ablate the uncertainty terms against KL-only joint gates; the current experiment proves joint optimization helps, but does not isolate which part of the objective causes the gain.\n3. Replace the smooth worst-stratum penalty with a held-out dual update or distributionally robust validation procedure; the present penalty did not deliver stable conditional protection.\n4. Export a packed/pruned checkpoint and benchmark peak VRAM, tokens per second, and end-to-end latency against dense and one-shot masks.\n5. Only after those checks, test a larger model or combine UPGD with quantization."},
        {"id": "questions", "type": "markdown", "body":
         "## Further questions\n\n"
         "How much of the +7.5-point development gain comes from joint gate optimization versus entropy and margin matching? Does the learned mask transfer across tasks because it preserves general computation, or because MMLU auxiliary examples match the evaluation format? Can uncertainty robustness be measured with enough per-stratum power without turning the calibration objective into a brittle worst-case optimizer? Does a physically packed implementation preserve the analytical 44.1% transformer-matrix reduction in real latency and memory?"},
    ]

    manifest = {"version": 1, "surface": "report",
                "title": "Uncertainty-Preserving Model Compression: Evidence and Method",
                "description": "Eleven-iteration synthesis and physical-pruning evaluation of UPGD.",
                "generatedAt": generated, "sources": sources, "charts": charts, "tables": tables, "blocks": blocks}
    artifact = {"surface": "report", "manifest": manifest,
                "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {
                    "accuracy": accuracy_rows, "metrics": metric_rows, "fidelity": fidelity_rows,
                    "strata": stratum_rows, "history": history_rows,
                }}, "sources": sources}
    (DEST / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    chart_map = [
        {"section": "Aggregate accuracy", "question": "Does joint gating improve accuracy?", "family": "comparison", "type": "grouped bar", "fields": ["dataset", "condition", "accuracy"], "claim": "UPGD gains on both cohorts, with uncertainty resolved only on development.", "palette": "relaxed multi-category", "delivery": "artifact chart accuracy"},
        {"section": "Uncertainty fidelity", "question": "Is dense behavior better preserved?", "family": "comparison", "type": "grouped bar", "fields": ["metric", "condition", "value"], "claim": "UPGD improves dense agreement and entropy rank on development.", "palette": "hard two-root cap", "delivery": "artifact chart fidelity"},
        {"section": "Conditional robustness", "question": "Are gains stable across entropy strata?", "family": "uncertainty and benchmark", "type": "signed grouped bar", "fields": ["stratum", "dataset", "accuracy_difference"], "claim": "The losing stratum reverses across datasets.", "palette": "hard two-root cap", "delivery": "artifact chart strata"},
    ]
    (DEST / "chart_map.json").write_text(json.dumps(chart_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DEST / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
