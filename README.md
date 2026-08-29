# Dual-constrained knowledge distillation under structured compression

This repository has one supported experimental workflow: compare **vanilla
teacher-to-student KL distillation** with **Dual-constrained KD** while learning
exact-budget structured FFN channel gates.

The supported empirical grid is intentionally fixed:

- models: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct;
- datasets: ARC-Challenge, MMLU-AUX, MMLU-Pro, and HellaSwag;
- methods: `kl_only` and `dual_kl`;
- FFN retention: 25%, 50%, and 75%; and
- default training: two exact without-replacement epochs; the reported grid uses
  three gate-training seeds.

Historical quantization, routing, UPGD regularization, UBKL, symmetric-KL, and
other iteration-specific runners are not part of the supported codebase.

## Objectives

Vanilla KL minimizes `E_x[KL(p_teacher || p_student)]`. Dual-KD adds the equality
constraint `E_x[H(p_student) - H(p_teacher)] = 0` and alternates primal descent
on gate parameters with detached dual ascent:

```text
lambda <- lambda + eta_lambda * mean(H_student - H_teacher).
```

The dense model weights stay frozen. Only stochastic exact-top-k gate logits
are optimized; deterministic top-k masks are used for evaluation. Logical masks
do not establish packed-kernel latency or energy savings.

## Results

Full grid: 3 models x 4 datasets x 2 methods x 3 retentions x 3 seeds = 216 masks,
two exact without-replacement epochs, 432 prediction files. The analysis is run once
per seed; the seeds are compared rather than pooled. `validate_results.py` passed for
each seed (12 calibrations, 72 training conditions, 144 prediction files per seed).

**Dual-KD conserves the dense teacher's uncertainty. Its effect on accuracy is not
resolved by three seeds, but it is what makes a dense-calibrated threshold survive
compression.** Each cell is `dual_kl - kl_only`, given as the **mean ± sample
standard deviation over the three seeds**, for three metrics:

- **Δaccuracy** — task accuracy on the held-out split. Positive means Dual-KD wins.
- **ΔH-MAE** — mean absolute error against the dense teacher's predictive entropy.
  Negative means Dual-KD tracks the teacher's uncertainty more closely.
- **ΔD_τ** — escalation disagreement when the dense model's entropy threshold is
  applied unchanged to the compressed model, averaged over the four target rates
  (0.05, 0.10, 0.20, 0.30). Negative means Dual-KD preserves the operating point.

With n=3 the standard deviation is a noisy estimate, so read it as a rough scale for
seed sensitivity rather than a confidence interval.

| model | dataset | retention | Δaccuracy | ΔH-MAE | ΔD_τ |
|---|---|---|---|---|---|
| 1.5B | ARC-Challenge | 0.75 | +0.0034 ± 0.0045 | -0.0039 ± 0.0024 | -0.0085 ± 0.0057 |
| 1.5B | ARC-Challenge | 0.50 | -0.0082 ± 0.0107 | -0.0035 ± 0.0023 | -0.0141 ± 0.0096 |
| 1.5B | ARC-Challenge | 0.25 | +0.0023 ± 0.0005 | -0.0228 ± 0.0025 | -0.1929 ± 0.0290 |
| 1.5B | MMLU-AUX | 0.75 | +0.0017 ± 0.0009 | -0.0741 ± 0.0076 | -0.1258 ± 0.0278 |
| 1.5B | MMLU-AUX | 0.50 | -0.0111 ± 0.0040 | -0.1689 ± 0.0102 | -0.3470 ± 0.0165 |
| 1.5B | MMLU-AUX | 0.25 | -0.0091 ± 0.0099 | -0.3417 ± 0.0075 | -0.6744 ± 0.0009 |
| 1.5B | MMLU-Pro | 0.75 | -0.0051 ± 0.0099 | -0.0226 ± 0.0007 | -0.0614 ± 0.0056 |
| 1.5B | MMLU-Pro | 0.50 | -0.0004 ± 0.0058 | -0.0394 ± 0.0014 | -0.1801 ± 0.0016 |
| 1.5B | MMLU-Pro | 0.25 | +0.0022 ± 0.0010 | -0.0743 ± 0.0099 | -0.3946 ± 0.0099 |
| 1.5B | HellaSwag | 0.75 | -0.0036 ± 0.0014 | -0.0003 ± 0.0001 | -0.0027 ± 0.0021 |
| 1.5B | HellaSwag | 0.50 | -0.0090 ± 0.0015 | -0.0001 ± 0.0001 | -0.0080 ± 0.0022 |
| 1.5B | HellaSwag | 0.25 | -0.0161 ± 0.0070 | +0.0009 ± 0.0023 | -0.0686 ± 0.0028 |
| 3B | ARC-Challenge | 0.75 | +0.0054 ± 0.0160 | -0.0137 ± 0.0010 | -0.0250 ± 0.0060 |
| 3B | ARC-Challenge | 0.50 | +0.0230 ± 0.0045 | -0.0155 ± 0.0014 | -0.0476 ± 0.0078 |
| 3B | ARC-Challenge | 0.25 | -0.0014 ± 0.0202 | -0.0332 ± 0.0152 | -0.2036 ± 0.0693 |
| 3B | MMLU-AUX | 0.75 | -0.0107 ± 0.0055 | -0.1636 ± 0.0066 | -0.2232 ± 0.0117 |
| 3B | MMLU-AUX | 0.50 | -0.0526 ± 0.0195 | -0.3318 ± 0.0133 | -0.4462 ± 0.0136 |
| 3B | MMLU-AUX | 0.25 | -0.0108 ± 0.0153 | -0.6464 ± 0.0067 | -0.6746 ± 0.0004 |
| 3B | MMLU-Pro | 0.75 | +0.0026 ± 0.0074 | -0.0641 ± 0.0038 | -0.1094 ± 0.0074 |
| 3B | MMLU-Pro | 0.50 | -0.0104 ± 0.0014 | -0.1535 ± 0.0049 | -0.3346 ± 0.0050 |
| 3B | MMLU-Pro | 0.25 | +0.0007 ± 0.0052 | -0.3141 ± 0.0101 | -0.6408 ± 0.0095 |
| 3B | HellaSwag | 0.75 | -0.0023 ± 0.0027 | -0.0004 ± 0.0006 | -0.0042 ± 0.0014 |
| 3B | HellaSwag | 0.50 | -0.0042 ± 0.0113 | +0.0001 ± 0.0018 | -0.0127 ± 0.0018 |
| 3B | HellaSwag | 0.25 | -0.0100 ± 0.0849 | -0.0080 ± 0.0142 | -0.1784 ± 0.1234 |
| 7B | ARC-Challenge | 0.75 | -0.0054 ± 0.0027 | -0.0165 ± 0.0018 | -0.0260 ± 0.0043 |
| 7B | ARC-Challenge | 0.50 | +0.0253 ± 0.0137 | -0.0205 ± 0.0030 | -0.0508 ± 0.0097 |
| 7B | ARC-Challenge | 0.25 | -0.0068 ± 0.0173 | -0.0267 ± 0.0039 | -0.1547 ± 0.0117 |
| 7B | MMLU-AUX | 0.75 | -0.0006 ± 0.0025 | -0.1292 ± 0.0119 | -0.1656 ± 0.0144 |
| 7B | MMLU-AUX | 0.50 | -0.0296 ± 0.0086 | -0.2981 ± 0.0109 | -0.3804 ± 0.0084 |
| 7B | MMLU-AUX | 0.25 | -0.1177 ± 0.0079 | -0.4863 ± 0.0009 | -0.5785 ± 0.0047 |
| 7B | MMLU-Pro | 0.75 | -0.0014 ± 0.0030 | -0.0463 ± 0.0037 | -0.0735 ± 0.0070 |
| 7B | MMLU-Pro | 0.50 | -0.0087 ± 0.0090 | -0.0878 ± 0.0079 | -0.1928 ± 0.0162 |
| 7B | MMLU-Pro | 0.25 | -0.0068 ± 0.0504 | -0.2249 ± 0.0544 | -0.4869 ± 0.0462 |
| 7B | HellaSwag | 0.75 | +0.0006 ± 0.0014 | -0.0007 ± 0.0009 | -0.0052 ± 0.0013 |
| 7B | HellaSwag | 0.50 | +0.0015 ± 0.0055 | -0.0004 ± 0.0003 | -0.0087 ± 0.0022 |
| 7B | HellaSwag | 0.25 | -0.0176 ± 0.0118 | -0.0005 ± 0.0023 | -0.0391 ± 0.0060 |

Across the 36 conditions, counting cells whose three-seed mean favours Dual-KD, and
cells whose absolute mean is smaller than its own standard deviation and so is not
resolved by three seeds:

| metric | Dual-KD better | not resolved (\|mean\| < std) |
|---|---|---|
| Δaccuracy | 11/36 | **20/36** |
| ΔH-MAE | 34/36 | 7/36 |
| ΔD_τ | **36/36** | **0/36** |

Per-seed counts (seed 1 / 2 / 3): Dual-KD is better on entropy fidelity in
**30 / 33 / 36** of 36 cells, on accuracy in **8 / 13 / 16**, and on ECE in
**8 / 8 / 9**. Of 72 paired bootstraps on error AUROC/AUPRC per seed, **8 / 6 / 7**
favour Dual-KD significantly and **13 / 11 / 9** favour vanilla KL; the rest cross zero.

The ordering is consistent: D_τ is the most seed-stable result, H-MAE next, and
accuracy is not resolved at all. Every one of the 36 D_τ cells favours Dual-KD and
none is within a standard deviation of zero. By contrast, the largest apparent
accuracy cost in the earlier single-seed table, 3B/HellaSwag at 0.25 retention, was
-0.0884; over three seeds it is **-0.0100 ± 0.0849**. Treat a per-cell accuracy
difference as seed noise unless its mean clears its standard deviation — 7B/MMLU-AUX
at 0.25 retention (-0.1177 ± 0.0079) is the clearest case that does, against
-0.4863 ± 0.0009 on H-MAE and -0.5785 ± 0.0047 on D_τ in the same cell.

The 7 unresolved H-MAE cells are all HellaSwag, where the effect is within ±0.008 of
zero. The 3 largest D_τ standard deviations are also HellaSwag or ARC at 0.25
retention, and even those keep their sign.

### Threshold compatibility

Applying the **dense** model's entropy threshold, unchanged, to the compressed model.
`student_rate` is the fraction the compressed model then escalates; it should equal the
target. This is where the conserved uncertainty pays off. Cells are the mean ± sample
standard deviation over the three seeds.

| target rate | mean abs rate error KL | Dual-KD | mean disagreement KL | Dual-KD |
|---|---|---|---|---|
| 0.05 | 0.2552 ± 0.0058 | **0.0290 ± 0.0002** | 0.2903 ± 0.0056 | **0.0693 ± 0.0000** |
| 0.10 | 0.3024 ± 0.0079 | **0.0475 ± 0.0011** | 0.3539 ± 0.0075 | **0.1204 ± 0.0010** |
| 0.20 | 0.3408 ± 0.0057 | **0.0807 ± 0.0031** | 0.4045 ± 0.0052 | **0.2060 ± 0.0018** |
| 0.30 | 0.3545 ± 0.0056 | **0.1026 ± 0.0041** | 0.4129 ± 0.0046 | **0.2725 ± 0.0024** |

This is the most seed-stable result in the study: the standard deviations are two
orders of magnitude below the effects, where 20 of 36 accuracy cells cannot clear
their own. Per seed (1 / 2 / 3), Dual-KD transfers the dense threshold better in
**143 / 144 / 143** of 144 paired conditions and lowers escalation disagreement in
**142 / 141 / 141**.

Under vanilla KL the dense threshold collapses at aggressive compression: at 25%
retention on MMLU-AUX, against an intended 10% escalation rate, it escalates
essentially every input for 1.5B and 3B (0.9987 to 1.0000 across seeds) and 87-88%
for 7B, because the compressed entropy distribution has shifted wholesale past the
threshold. Dual-KD escalates 0% in the same cells — a miss in the other direction,
but a far smaller one — and keeps the operating point usable at the rates above.

HellaSwag is the control throughout: vanilla KL already reaches 0.02-0.09 entropy error
there, so the constraint has nothing to close and Dual-KD changes little.

Per-cell numbers are in [runs/dual_kd_vs_kl/RUN.md](runs/dual_kd_vs_kl/RUN.md).

## Repository map

```text
configs/experiment.yaml          complete supported grid and hyperparameters
src/dualkd/                      objectives, gate training, scoring, and metrics
scripts/prepare_datasets.py      download and normalize all four datasets
scripts/calibrate.py             dense references and gate initialization
scripts/run_experiment.py        train masks or score held-out partitions
scripts/analyze_results.py       matched metrics and paired bootstrap analysis
scripts/build_report.py          self-contained HTML report
scripts/validate_results.py      end-to-end artifact validation
```

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiment,test]"
pytest -q
```

Hugging Face credentials should be supplied through the shell environment. Do
not store tokens or W&B keys in the repository.

## Run the pipeline

All commands read [configs/experiment.yaml](configs/experiment.yaml). Defaults
write normalized inputs to `runs/dual_kd_vs_kl_inputs/` and learned masks,
predictions, metrics, and reports to `runs/dual_kd_vs_kl/`.

### 1. Prepare datasets

```bash
python scripts/prepare_datasets.py
```

The script verifies declared split sizes and ID disjointness. MMLU-Pro uses a
deterministic category-stratified 80/20 split of its labeled test data;
HellaSwag uses official train and a deterministic 20/80 split of labeled
validation because official test labels are hidden.

### 2. Build dense references

Run one calibration per dataset/model pair:

```bash
python scripts/calibrate.py --model-key 1p5b_bf16 --dataset-key arc_challenge
```

Calibration computes dense probabilities, three equal-frequency teacher-entropy
strata, and activation-times-weight-norm FFN importance for matched gate
initialization.

### 3. Train masks

```bash
python scripts/run_experiment.py --stage train \
  --model-key 1p5b_bf16 --dataset-key arc_challenge \
  --method dual_kl --retention 0.50
```

Omit `--method` and `--retention` to run both methods and all retention levels
for that model/dataset pair. Completed artifacts are reused.

To log training, set `WANDB_PROJECT`; `WANDB_ENTITY`, `WANDB_RUN_GROUP`, and
`WANDB_MODE` are optional. Authentication remains external to the repository.

### 4. Score masks

```bash
python scripts/run_experiment.py --stage score \
  --model-key 1p5b_bf16 --dataset-key arc_challenge
```

One scoring invocation evaluates all six masks on the lambda-selection and
held-out evaluation partitions.

### 5. Analyze, report, and validate

After the complete grid is available:

```bash
python scripts/analyze_results.py
python scripts/build_report.py
python scripts/validate_results.py
```

The main output is `runs/dual_kd_vs_kl/report/report.html`. CSV outputs contain
accuracy, normalized entropy MAE to the dense teacher, margin MAE, ECE, error
AUROC/AUPRC, AURC, and dense-cascade curves. Paired example bootstrap intervals
compare Dual-KD minus KL for error AUROC and AUPRC; they do not quantify
training-seed variability.

## Reusing Iteration 19 artifacts

The cleaned scripts retain the Iteration 19 artifact layout. Existing data and
two-method masks can therefore be analyzed without retraining:

```bash
python scripts/analyze_results.py \
  --root runs/iter19_corrected_arc \
  --source-root runs/iter18_full_cohort \
  --output runs/iter19_corrected_arc/outputs/dual_kd_vs_kl
python scripts/build_report.py \
  --root runs/iter19_corrected_arc \
  --downstream runs/iter19_corrected_arc/outputs/dual_kd_vs_kl \
  --report-dir runs/iter19_corrected_arc/report_dual_kd_vs_kl
python scripts/validate_results.py \
  --root runs/iter19_corrected_arc \
  --source-root runs/iter18_full_cohort \
  --downstream runs/iter19_corrected_arc/outputs/dual_kd_vs_kl \
  --report-dir runs/iter19_corrected_arc/report_dual_kd_vs_kl \
  --receipt runs/iter19_corrected_arc/report_dual_kd_vs_kl/validation.json
```

The LIGHT paper and its independent provenance checks are maintained in the
[standalone paper repository](https://github.com/shrn2/light2026_dualkd).
