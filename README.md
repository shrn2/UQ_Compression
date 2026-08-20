# TUAC

TUAC is a runnable minimum implementation of the method in [plan.md](plan.md): it uses a
strong teacher's confidence and teacher–student disagreement to decide which student
transformer blocks should retain higher weight precision.

The implementation deliberately separates expensive model inference from cheap policy
analysis:

```text
JSONL calibration examples
        │
        ├── teacher probabilities (teacher is then unloaded)
        └── full-precision student probabilities
                    │
                    └── one-block-at-a-time fake quantization
                                  │
                                  ▼
                         calibration.npz
                                  │
                    five signal ablations + budgets
                                  │
                                  ▼
                              report.json
                                  │
                         apply one selected policy
                                  ▼
                       probabilities + reliability metrics
```

## What is implemented

- Normalized teacher entropy and confidence.
- `KL(teacher || student)` disagreement.
- All five planned weighting modes: student-only, confidence-only,
  uncertainty-only, disagreement-only, and the proposed combined score.
- Layer sensitivity from `KL(student_fp || student_one_block_quantized)`.
- Symmetric per-group fake weight quantization for transformer blocks.
- Exact multiple-choice-knapsack allocation for `{3, 4, 8}` or custom bit sets,
  weighted by real per-block parameter counts.
- Five reproducible random mixed-precision policies per budget and uniform policies
  wherever the target is an available uniform width.
- Accuracy, ECE, Brier score, error-detection AUROC, AURC, uncertainty distortion,
  and Spearman uncertainty preservation.
- Cached, portable `.npz` artifacts and JSON policy reports.
- An evaluation command that applies a selected mixed-precision policy.

The included adapter targets Hugging Face causal language models and evaluates
multiple-choice tasks by normalized continuation likelihood. Mixed datasets may have
different numbers of choices; zero padding is excluded from normalized entropy.

## Installation

Python 3.10+ is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[hf,test]"
pytest
```

Large models may also require authentication with the Hugging Face Hub and enough CPU/GPU
memory. The teacher and student are loaded sequentially, so they do not need to coexist in
memory.

## HPC workflow

The repository is designed to keep source code and experiment definitions in Git while
keeping model weights, datasets, checkpoints, and large `.npz`/prediction artifacts in HPC
scratch or object storage. See [hpc/README.md](hpc/README.md) for Slurm setup and
[configs/iter16_uncertainty.yaml](configs/iter16_uncertainty.yaml) for the reproducible
UPGD lambda sweep. The CI workflow runs compile and unit checks; GPU sweeps should run
through the provided Slurm array script.

## Data format

Calibration and multiple-choice evaluation data use one JSON object per line:

```json
{"id":"arc-1","dataset":"arc","prompt":"Question: ...\nAnswer:","choices":[" A"," B"," C"," D"],"answer":2}
```

`answer` is a zero-based choice index and is optional during calibration, but required for
task/reliability metrics. Choice strings should include any intended leading separator.
Only train/development examples should be used for `collect`; keep test data for
`evaluate`. A tiny schema example is in
[examples/calibration.sample.jsonl](examples/calibration.sample.jsonl).

## Run the pipeline

Collect teacher/student signals and 4-bit block probes:

```powershell
tuac collect `
  --data data/calibration.jsonl `
  --teacher Qwen/Qwen2.5-7B-Instruct `
  --student Qwen/Qwen2.5-1.5B-Instruct `
  --probe-bits 4 `
  --output runs/calibration.npz
```

This is the expensive step: it performs one complete student scoring pass per transformer
block. The teacher is called only once. Use `--limit 20` for a smoke test.

Generate all planned signal ablations at three memory budgets:

```powershell
tuac analyze `
  --artifact runs/calibration.npz `
  --budgets 4.0,3.5,3.0 `
  --bits 3,4,8 `
  --output runs/policies.json
```

Apply the proposed 3.5-bit policy to held-out data:

```powershell
tuac evaluate `
  --artifact runs/calibration.npz `
  --report runs/policies.json `
  --data data/evaluation.jsonl `
  --student Qwen/Qwen2.5-1.5B-Instruct `
  --mode combined `
  --budget 3.5 `
  --output runs/combined-3.5-eval.npz
```

Evaluate FP, uniform, three random seeds, student-only, and proposed policies in
one model load:

```powershell
tuac benchmark `
  --artifact runs/calibration.npz `
  --report runs/policies.json `
  --data data/evaluation.jsonl `
  --student Qwen/Qwen2.5-1.5B-Instruct `
  --output-dir runs/benchmark
```

Inspect the full-precision and individual block-probe metrics on a labeled calibration
artifact:

```powershell
tuac metrics --artifact runs/calibration.npz
```

The values in [configs/mvp.yaml](configs/mvp.yaml) document the intended first experiment;
CLI arguments remain explicit so every run is visible and reproducible in shell history.

## Iteration 1: empirical teacher/quantization interaction

Iteration 1 adds a resumable multi-bit experiment that directly probes every candidate
precision, constructs five stratified calibration policies, and deduplicates identical
held-out allocations before inference:

```powershell
tuac collect-iteration `
  --data runs/iter1_arc/calibration_pool.jsonl `
  --teacher Qwen/Qwen2.5-1.5B-Instruct `
  --student Qwen/Qwen2.5-0.5B-Instruct `
  --bits 2,4,8 `
  --batch-size 128 `
  --output runs/iter1_arc/calibration.npz

tuac analyze-iteration `
  --artifact runs/iter1_arc/calibration.npz `
  --subset-size 500 `
  --calibration-seeds 11,23,37,53,71 `
  --budgets 4.0,3.5,3.0 `
  --output runs/iter1_arc/policies.json

tuac benchmark-iteration `
  --artifact runs/iter1_arc/calibration.npz `
  --report runs/iter1_arc/policies.json `
  --data runs/iter1_arc/evaluation.jsonl `
  --teacher Qwen/Qwen2.5-1.5B-Instruct `
  --student Qwen/Qwen2.5-0.5B-Instruct `
  --batch-size 128 `
  --output-dir runs/iter1_arc/benchmark
```

The scorer left-pads and token-buckets continuations, and uses a model's final-logit-only
projection when supported. This preserves normalized continuation likelihood while avoiding
work on padded prompt positions. Probe checkpoints include the scoring version and batch size,
so interrupted or numerically incompatible runs cannot silently mix outputs.

## Iteration 2: capability-critical versus trust-critical layers

Iteration 2 reuses the full 750-example multi-bit probe cache to compare predictive
sensitivity with uncertainty-ranking sensitivity. It bootstraps examples, reports top-k
overlap and rank displacement, tests entropy against margin uncertainty, and applies a
predeclared gate before any whole-model policy benchmark:

```powershell
tuac analyze-trust `
  --artifact runs/iter1_arc/calibration.npz `
  --bootstrap-replicates 1000 `
  --bootstrap-seed 20260817 `
  --output runs/iter2_arc/layer_analysis.json

# Run these only when layer_analysis.json records gate.passed=true.
tuac build-trust-policies `
  --artifact runs/iter1_arc/calibration.npz `
  --analysis runs/iter2_arc/layer_analysis.json `
  --budgets 4.0,3.5,3.0 `
  --output runs/iter2_arc/policies.json

tuac benchmark-trust `
  --artifact runs/iter1_arc/calibration.npz `
  --report runs/iter2_arc/policies.json `
  --data runs/iter1_arc/evaluation.jsonl `
  --student Qwen/Qwen2.5-0.5B-Instruct `
  --batch-size 128 `
  --output-dir runs/iter2_arc/benchmark
```

The ARC run failed the gate because capability/trust Spearman correlation was 0.959 at
2 bits and 0.967 at 4 bits. Policy evaluation and MMLU-Pro were therefore intentionally
skipped. The complete execution record is in
[runs/iter2_arc/RUN.md](runs/iter2_arc/RUN.md).

## Iteration 3: quantization-induced delegation drift

Iteration 3 tests whether whole-model quantization changes which examples an SLM sends to
a stronger fallback. It includes an FP router-viability gate, uniform Q8/Q4/Q2 fake
quantization, frozen and recalibrated routers, exact matched-rate comparisons, random and
oracle baselines, flip taxonomy, paired uncertainty/cascade statistics, and controls for
unchanged predictions:

```powershell
tuac cascade-gate `
  --artifact runs/iter1_arc/calibration.npz `
  --role teacher `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --output runs/iter3_arc/gate_1.5b.json

tuac collect-cascade `
  --calibration-data runs/iter1_arc/calibration_pool.jsonl `
  --evaluation-data runs/iter1_arc/evaluation.jsonl `
  --slm Qwen/Qwen2.5-1.5B-Instruct `
  --fallback Qwen/Qwen2.5-7B-Instruct `
  --bits 8,4,2 `
  --reuse-calibration-artifact runs/iter1_arc/calibration.npz `
  --reuse-evaluation-probabilities runs/iter1_arc/benchmark/probabilities.npz `
  --reuse-key teacher `
  --output runs/iter3_arc/probabilities.npz

tuac analyze-cascade `
  --artifact runs/iter3_arc/probabilities.npz `
  --bootstrap-replicates 2000 `
  --bootstrap-seed 20260818 `
  --output runs/iter3_arc/analysis.json
```

The 0.5B ARC router failed the AUROC > 0.60 gate, so the preregistered scale-up used a
1.5B SLM and 7B fallback. ARC produced a moderate-positive result: Q4/Q2 cause large
uncertainty-rank and delegation drift, but all routing-only accuracy intervals cross zero
when local/fallback predictions and delegation count are held fixed. A complete
1,000-example MMLU-Pro scale-up with a 3B fallback replicated decision drift; at two Q2
operating points the quantized ranking significantly outperformed the FP ranking, so neither
benchmark establishes harmful routing-only regret. See
[runs/iter3_arc/RUN.md](runs/iter3_arc/RUN.md) and the portable HTML report for the exact
results and claim boundary.

## Iteration 4: cascade-aware compression

Iteration 4 tests whether layer-local cascade-utility sensitivity can choose a better
mixed-precision SLM than conventional output-KL sensitivity. It reuses the Iteration 3
Qwen2.5-1.5B → Qwen2.5-7B cascade, collects 84 one-block ARC probes, freezes exact-budget
3.5- and 4.0-bit policies, and compares CUAQ against KL and ten random allocations:

```powershell
tuac collect-cascade-probes `
  --cascade-artifact runs/iter3_arc/probabilities.npz `
  --calibration-data runs/iter1_arc/calibration_pool.jsonl `
  --slm Qwen/Qwen2.5-1.5B-Instruct `
  --bits 2,4,8 `
  --output runs/iter4_arc/layer_probes.npz

tuac analyze-cascade-compression `
  --artifact runs/iter4_arc/layer_probes.npz `
  --budgets 3.5,4.0 `
  --output runs/iter4_arc/policies.json

tuac benchmark-cascade-compression `
  --data runs/iter1_arc/evaluation.jsonl `
  --cascade-artifact runs/iter3_arc/probabilities.npz `
  --probe-artifact runs/iter4_arc/layer_probes.npz `
  --policies runs/iter4_arc/policies.json `
  --slm Qwen/Qwen2.5-1.5B-Instruct `
  --methods kl,cuaq `
  --output-dir runs/iter4_arc/benchmark
```

The Stage-1 result is **NO-GO**. CUAQ trails KL by 0.53 utility points at 3.5 bits
(95% CI -5.14 to +4.53) and 13.31 points at 4.0 bits (-18.22 to -8.00). Under the
predeclared gate, DBAQ held-out evaluation, CARQ, MMLU-Pro, and real quantization are
intentionally skipped. See [runs/iter4_arc/RUN.md](runs/iter4_arc/RUN.md).

## Iteration 5: capability-constrained cascade quantization

Iteration 5 implements C3Q: generate exact-budget neighbors around KL, retain only
capability-safe configurations, and select directly on whole-model cascade utility. The
CLI supports literal exact evaluation of every candidate; the completed 8GB-GPU run used
the cached layer probes to screen 100 candidates per budget and exact-scored six frozen
finalists per budget:

```powershell
tuac search-c3q `
  --calibration-data runs/iter1_arc/calibration_pool.jsonl `
  --cascade-artifact runs/iter3_arc/probabilities.npz `
  --probe-artifact runs/iter4_arc/layer_probes.npz `
  --policies runs/iter4_arc/policies.json `
  --slm Qwen/Qwen2.5-1.5B-Instruct `
  --output-dir runs/iter5_arc/search `
  --candidates-per-budget 100 `
  --random-candidates 15 `
  --generation-seeds 11,23,37 `
  --screen-with-layer-probes `
  --exact-finalists 6

tuac benchmark-c3q `
  --evaluation-data runs/iter1_arc/evaluation.jsonl `
  --cascade-artifact runs/iter3_arc/probabilities.npz `
  --probe-artifact runs/iter4_arc/layer_probes.npz `
  --search-results runs/iter5_arc/search/results.json `
  --slm Qwen/Qwen2.5-1.5B-Instruct `
  --output-dir runs/iter5_arc/benchmark
```

The result is **NO-GO**. The 3.5-bit safe set shows calibration headroom, but C3Q loses
2.03 held-out mean-utility points to KL (95% CI -6.73 to +2.88) while standalone accuracy
stays within 0.33 points. At 4.0 bits, no alternative exact finalist survives the joint
guardrail. The ARC gate therefore blocks MMLU-Pro and packed quantization. See
[runs/iter5_arc/RUN.md](runs/iter5_arc/RUN.md) for the execution adaptation and full claim
boundary.

## Iteration 6: uncertainty-guided progressive densification

Iteration 6 physically removes nested FFN channels from Qwen2.5-1.5B-Instruct, learns
which aggressive-sparse errors dense inference can correct, and tests calibration-frozen
two-level routing on held-out ARC:

```powershell
tuac collect-sparse-hierarchy `
  --calibration-data runs/iter1_arc/calibration_pool.jsonl `
  --cascade-artifact runs/iter3_arc/probabilities.npz `
  --slm Qwen/Qwen2.5-1.5B-Instruct `
  --output runs/iter6_arc/calibration.npz `
  --masks-output runs/iter6_arc/masks.npz `
  --importance-output runs/iter6_arc/importance.npz

tuac analyze-densification `
  --artifact runs/iter6_arc/calibration.npz `
  --output runs/iter6_arc/analysis.json `
  --probe-output runs/iter6_arc/probe.npz

tuac score-sparse-hierarchy `
  --data runs/iter1_arc/evaluation.jsonl `
  --cascade-artifact runs/iter3_arc/probabilities.npz `
  --masks runs/iter6_arc/masks.npz `
  --slm Qwen/Qwen2.5-1.5B-Instruct `
  --output runs/iter6_arc/evaluation.npz

tuac benchmark-densification `
  --analysis runs/iter6_arc/analysis.json `
  --probe runs/iter6_arc/probe.npz `
  --evaluation-artifact runs/iter6_arc/evaluation.npz `
  --output-dir runs/iter6_arc/benchmark
```

The result is **NO-GO**. Fifty-percent FFN pruning reduces the analytical transformer
matrix-compute proxy to 0.559 and leaves 31.7% calibration accuracy. A cheap uncertainty
signal authorizes the one held-out evaluation, but the uplift probe does not beat generic
uncertainty on calibration and does not statistically dominate entropy on held-out ARC.
The failed two-level gate blocks three-level routing, MMLU-Pro, and runtime claims. See
[runs/iter6_arc/RUN.md](runs/iter6_arc/RUN.md).

## Iteration 7: counterfactual capacity routing

Iteration 7 asks whether sparse residual-stream states predict the signed value of restoring
dense capacity. It reuses the Iteration 6 channel ordering, adds a physical S40 variant,
chooses the sparse start from oracle efficiency, and evaluates every learned score with
five-fold out-of-fold predictions and exact matched-rate routing:

```powershell
tuac collect-capacity-hierarchy --iteration6-artifact runs/iter6_arc/calibration.npz `
  --importance runs/iter6_arc/importance.npz --data runs/iter1_arc/calibration_pool.jsonl `
  --slm Qwen/Qwen2.5-1.5B-Instruct --output runs/iter7_arc/hierarchy.npz `
  --masks-output runs/iter7_arc/masks.npz
tuac analyze-capacity-headroom --hierarchy runs/iter7_arc/hierarchy.npz `
  --output runs/iter7_arc/oracle_analysis.json
tuac collect-capacity-representations --hierarchy runs/iter7_arc/hierarchy.npz `
  --oracle-analysis runs/iter7_arc/oracle_analysis.json --masks runs/iter7_arc/masks.npz `
  --data runs/iter1_arc/calibration_pool.jsonl --slm Qwen/Qwen2.5-1.5B-Instruct `
  --output runs/iter7_arc/representations.npz
tuac analyze-counterfactual-routing --hierarchy runs/iter7_arc/hierarchy.npz `
  --oracle-analysis runs/iter7_arc/oracle_analysis.json `
  --representations runs/iter7_arc/representations.npz `
  --output runs/iter7_arc/development_analysis.json --probe-output runs/iter7_arc/probes.npz
```

The result is **NO-GO**. S50 has substantial oracle headroom and actual S50-S25
prediction disagreement reproduces AUROC 0.709, but the best hidden signed-advantage probe
has AUROC 0.587 versus 0.632 for ordinary uncertainty and recovers only 33.1% of oracle
gain at a 30% dense rate. ARC validation is not reused, and the failed development gate
blocks MMLU-Pro and runtime work. See [runs/iter7_arc/RUN.md](runs/iter7_arc/RUN.md).

## Iteration 8: capacity-contrast distillation

Iteration 8 treats paired S50/S25 behavior as privileged training supervision. It caches
four S50 residual layers and answer probabilities on 3,000 MMLU auxiliary-train prompts,
constructs hard and soft capacity-contrast targets, and compares a matched linear router
with hard, soft, and multi-task nonlinear CCD routers on a fixed held-out split:

```powershell
python scripts/prepare_ccd_mmlu.py --count 3000 `
  --output runs/iter8_arc/ccd_training.jsonl
tuac collect-capacity-contrast --data runs/iter8_arc/ccd_training.jsonl `
  --masks runs/iter7_arc/masks.npz --slm Qwen/Qwen2.5-1.5B-Instruct `
  --output runs/iter8_arc/paired_capacity.npz --layer-indices 6,13,20,27
tuac analyze-capacity-contrast --contrast-artifact runs/iter8_arc/paired_capacity.npz `
  --arc-hierarchy runs/iter7_arc/hierarchy.npz `
  --arc-representations runs/iter7_arc/representations.npz `
  --iteration7-probes runs/iter7_arc/probes.npz `
  --output runs/iter8_arc/analysis.json --scores-output runs/iter8_arc/scores.npz `
  --models-output runs/iter8_arc/routers.pt
```

The result is **STOP at Gate 1**. CCD-Hard clears the absolute distillation heuristic with
0.6767 held-out disagreement AUROC, but the matched linear router reaches 0.6777. The
nonlinear architecture therefore does not provide the required clear improvement. In
accordance with the plan, ARC advantage, matched-compute routing, MMLU-Pro, and runtime
evaluation were not run. See [runs/iter8_arc/RUN.md](runs/iter8_arc/RUN.md) and the
[portable technical report](runs/iter8_arc/report/report.html).

## Iteration 9: frozen capacity-contrast routing

Iteration 9 freezes the successful Iteration 8 linear disagreement router before reading
fresh task outcomes, removes training/evaluation duplicates, and evaluates exact S50-to-
dense routing on 1,000 category-stratified MMLU-Pro examples:

```powershell
tuac freeze-ccr --paired-artifact runs/iter8_arc/paired_capacity.npz `
  --iteration8-models runs/iter8_arc/routers.pt `
  --iteration8-scores runs/iter8_arc/scores.npz `
  --iteration8-analysis runs/iter8_arc/analysis.json `
  --output runs/iter9_mmlu_pro/frozen_ccr.npz
tuac prepare-ccr-evaluation --training-data runs/iter8_arc/ccd_training.jsonl `
  --evaluation-data runs/iter3_mmlu_pro/evaluation.jsonl `
  --output runs/iter9_mmlu_pro/evaluation.jsonl `
  --manifest runs/iter9_mmlu_pro/evaluation_manifest.json
tuac collect-ccr-evaluation --data runs/iter9_mmlu_pro/evaluation.jsonl `
  --masks runs/iter7_arc/masks.npz --frozen-router runs/iter9_mmlu_pro/frozen_ccr.npz `
  --slm Qwen/Qwen2.5-1.5B-Instruct --output runs/iter9_mmlu_pro/evaluation.npz
tuac analyze-ccr-system --evaluation-artifact runs/iter9_mmlu_pro/evaluation.npz `
  --paired-artifact runs/iter8_arc/paired_capacity.npz `
  --iteration8-scores runs/iter8_arc/scores.npz `
  --output runs/iter9_mmlu_pro/analysis.json --scores-output runs/iter9_mmlu_pro/scores.npz
```

The result is **NO-GO**. CCR slightly exceeds entropy in positive-advantage AUROC
(0.5664 versus 0.5633), but has lower AUPRC and trails uncertainty at both primary exact
dense rates. CCR accuracy is 16.3% at 20% dense execution versus 16.7% for entropy, and
17.7% at 30% versus 18.3% for max-probability uncertainty. The failed system gate blocks
feature/data ablations, S40, and runtime work. See
[runs/iter9_mmlu_pro/RUN.md](runs/iter9_mmlu_pro/RUN.md) and the
[portable technical report](runs/iter9_mmlu_pro/report/report.html).

## Iteration 10: uncertainty-robust structured allocation

Iteration 10 uses dense predictive-entropy strata as robust calibration environments for
one static physically pruned model. URSA constructs nested 25% and 50% FFN-channel masks
by minimizing a smooth worst-stratum omitted-importance objective and compares them with a
matched global mean-activation baseline.

The result is **NO-GO**. URSA changes only 0.62% of retained channels at 50% pruning and
scores 33.33% versus 34.33% for the baseline on 600 held-out MMLU auxiliary examples. It
also loses in every entropy stratum. This rules out uncertainty-stratified reweighting of
the tested local importance score and motivates joint whole-model optimization. See
[runs/iter10_ursa/RUN.md](runs/iter10_ursa/RUN.md).

## Iteration 11: uncertainty-preserving gate distillation

Iteration 11 implements UPGD, which jointly learns exact-top-k FFN channel gates through
the complete frozen Qwen2.5-1.5B-Instruct model. The loss distills dense answer
distributions, entropy, and answer margin while training over stochastic subnetworks and
three dense-entropy strata. Inference uses one deterministic physical mask with no router,
gate parameters, teacher, or second pass.

At 50% FFN pruning, UPGD reaches 41.83% versus 34.33% for the matched one-shot baseline on
the fixed 600-example validation split: +7.50 percentage points with a paired bootstrap
95% interval of [+2.50, +12.50]. On an exploratory 1,000-example MMLU-Pro transfer it
reaches 13.20% versus 11.30%, a directionally positive but unresolved +1.90 points
[-0.60, +4.30]. The conservative gate remains **NO-GO** because the per-entropy-stratum
effect is not stable across datasets. The warranted conclusion is that joint gate learning
is a promising aggregate compression method, while uncertainty-conditional robustness is
not yet established. See [runs/iter11_upgd/RUN.md](runs/iter11_upgd/RUN.md) and the
[technical synthesis](runs/iter11_upgd/report/report.html).

## Iteration 12: larger-model 4-bit transfer

Iteration 12 validates UPGD on `Qwen/Qwen2.5-3B-Instruct` loaded with bitsandbytes NF4
4-bit quantization. The model has 36 layers and retains exactly 5,504 of 11,008 FFN
channels per layer. Because quantized `Linear4bit` modules cannot be physically shrunk in
place, evaluation applies exact logical channel masks immediately before `down_proj`; this
tests quantized quality, not packed sparse-kernel speed.

UPGD improves ARC-Challenge accuracy from 26.09% to 31.77% (+5.69 points, paired 95% CI
[+0.67, +10.70]) and passes the conservative gate. It loses on the 1,000-example MMLU-Pro
transfer, 13.50% to 11.00% (-2.50 points, CI [-5.30, +0.30]), failing every uncertainty
stratum. The larger-model result is therefore mixed: UPGD is feasible in 4-bit NF4 on the
8GB GPU and can improve one held-out task, but cross-dataset transfer is not established.
See [runs/iter12_q4_3b/RUN.md](runs/iter12_q4_3b/RUN.md), the [portable report](runs/iter12_q4_3b/report/report.html), the [validation artifact](runs/iter12_q4_3b/validation.json), and the reproducible scorer [scripts/score_quantized_mask_benchmark.py](scripts/score_quantized_mask_benchmark.py).

For a single method-level view across both model scales and all evaluated datasets—including CommonsenseQA, HellaSwag, OpenBookQA, and Winogrande transfer cohorts—see the [combined UPGD report](runs/iter11_12_upgd/report/report.html).
The report now also includes the Iteration 14 HellaSwag ablation (random controls, one-shot, KL-only, KL+H+M, and full UPGD) for both models; raw artifacts are in [runs/iter14_ablation_hellaswag](runs/iter14_ablation_hellaswag).
It also includes the Iteration 15 rerun of that matrix on the prior MMLU auxiliary and ARC-Challenge cohorts; raw artifacts are in [runs/iter15_ablation_previous](runs/iter15_ablation_previous).

## Interpreting output

`policies.json` records the example weights, teacher entropy, disagreement, block
importance, chosen width per block, actual average bits, bit cap, and objective for every
ablation/budget. It also reports the Spearman correlation between student-only and combined
teacher-aware block rankings.

The allocation budget covers transformer-block weights included in the policy. Embeddings,
the LM head, biases, scales, and packing metadata are not counted. Report this block-weight
average separately from whole-model storage.

## Important scope boundaries

Fake quantization reproduces numerical weight perturbations while retaining floating-point
storage. It supports sensitivity research and accuracy/reliability evaluation, but it does
**not** provide packed model files, reduced peak memory, latency gains, or tokens/second
improvements. Use a backend such as GPTQ/AWQ for those deployment claims, applying the
derived block policy where that backend supports mixed widths.

The MVP covers the primary multiple-choice path (MMLU-Pro and ARC-Challenge). GSM8K exact
match, GPQA-specific formatting, C4-based GPTQ/AWQ calibration, packed-kernel benchmarks,
and third-party mixed-precision baselines remain experiment adapters,
not silently simulated baselines. This keeps comparisons scientifically honest: the
current `student_only` versus `combined` comparison isolates the contribution of the
teacher within exactly the same quantization and allocation machinery.

## Tests

The tests cover scoring semantics, mixed-choice entropy, allocation under equal and unequal
block sizes, reliability metrics (including ties), and artifact round trips:

```powershell
pytest
```
