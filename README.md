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
per seed; the seeds are compared rather than pooled.

**Dual-KD conserves the dense teacher's uncertainty. Its effect on accuracy is not
resolved by three seeds, but it is what makes a dense-calibrated threshold survive
compression.** Each cell is `dual_kl - kl_only`: negative entropy MAE means better
fidelity, positive accuracy means Dual-KD wins. Entropy MAE is the mean over the
three seeds. Accuracy is shown as the **range** across seeds, because in 15 of 36
accuracy cells the sign is not stable across seeds.

| model | dataset | Δacc .75 | Δacc .50 | Δacc .25 | ΔentMAE .75 | ΔentMAE .50 | ΔentMAE .25 |
|---|---|---|---|---|---|---|---|
| 1.5B | ARC-Challenge | +0.0000 to +0.0085 | -0.0205 to -0.0009 | +0.0017 to +0.0026 | -0.0039 | -0.0035 | -0.0228 |
| 1.5B | MMLU-AUX | +0.0011 to +0.0028 | -0.0154 to -0.0074 | -0.0204 to -0.0028 | -0.0741 | -0.1689 | -0.3417 |
| 1.5B | MMLU-Pro | -0.0162 to +0.0029 | -0.0071 to +0.0033 | +0.0017 to +0.0033 | -0.0226 | -0.0394 | -0.0743 |
| 1.5B | HellaSwag | -0.0047 to -0.0020 | -0.0102 to -0.0073 | -0.0212 to -0.0081 | -0.0003 | -0.0001 | +0.0009 |
| 3B | ARC-Challenge | -0.0128 to +0.0171 | +0.0179 to +0.0265 | -0.0247 to +0.0111 | -0.0137 | -0.0155 | -0.0332 |
| 3B | MMLU-AUX | -0.0154 to -0.0046 | -0.0749 to -0.0390 | -0.0212 to +0.0068 | -0.1636 | -0.3318 | -0.6464 |
| 3B | MMLU-Pro | -0.0017 to +0.0112 | -0.0112 to -0.0087 | -0.0042 to +0.0062 | -0.0641 | -0.1535 | -0.3141 |
| 3B | HellaSwag | -0.0052 to +0.0000 | -0.0132 to +0.0085 | -0.0884 to +0.0802 | -0.0004 | +0.0001 | -0.0080 |
| 7B | ARC-Challenge | -0.0085 to -0.0034 | +0.0154 to +0.0410 | -0.0222 to +0.0119 | -0.0165 | -0.0205 | -0.0267 |
| 7B | MMLU-AUX | -0.0023 to +0.0023 | -0.0391 to -0.0222 | -0.1269 to -0.1127 | -0.1292 | -0.2981 | -0.4863 |
| 7B | MMLU-Pro | -0.0033 to +0.0021 | -0.0191 to -0.0033 | -0.0432 to +0.0507 | -0.0463 | -0.0878 | -0.2249 |
| 7B | HellaSwag | -0.0010 to +0.0017 | -0.0032 to +0.0075 | -0.0291 to -0.0056 | -0.0007 | -0.0004 | -0.0005 |

Counts per seed (seed 1 / 2 / 3). Dual-KD is better on entropy fidelity in
**30 / 33 / 36** of 36 cells, on accuracy in **8 / 13 / 16**, and on ECE in
**8 / 8 / 9**. Of 72 paired bootstraps on error AUROC/AUPRC per seed, **8 / 6 / 7**
favour Dual-KD significantly and **13 / 11 / 9** favour vanilla KL; the rest cross zero.

The entropy-fidelity result is stable: the seven cells whose sign disagrees across
seeds are all HellaSwag, where the effect is within ±0.008 of zero. The accuracy
result is not: the sign flips across seeds in 15 of 36 cells, and the single largest
apparent accuracy cost in the earlier single-seed table (3B/HellaSwag at 0.25
retention, -0.0884) spans -0.0884 to +0.0802 across seeds. Treat per-cell accuracy
differences as seed noise unless the range excludes zero — 7B/MMLU-AUX at 0.25
retention (-0.1269 to -0.1127) is the clearest case that does.

### Threshold compatibility

Applying the **dense** model's entropy threshold, unchanged, to the compressed model.
These threshold numbers are **seed 1 only**; the per-seed threshold sweep has not been rerun.
`student_rate` is the fraction the compressed model then escalates; it should equal the
target. This is where the conserved uncertainty pays off.

| target rate | mean abs rate error KL | Dual-KD | mean disagreement KL | Dual-KD |
|---|---|---|---|---|
| 0.05 | 0.2509 | **0.0292** | 0.2865 | **0.0693** |
| 0.10 | 0.2951 | **0.0487** | 0.3475 | **0.1214** |
| 0.20 | 0.3346 | **0.0842** | 0.3991 | **0.2072** |
| 0.30 | 0.3491 | **0.1072** | 0.4081 | **0.2724** |

Dual-KD transfers the dense threshold better in **143/144** paired conditions and
lowers escalation disagreement in **142/144**. Under vanilla KL the dense threshold
collapses entirely at aggressive compression: at 25% retention on MMLU-AUX it escalates
100% of inputs instead of the intended 10%, because the compressed entropy distribution has
shifted wholesale past the threshold. Dual-KD keeps the operating point usable.

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
