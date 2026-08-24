# Dual-constrained knowledge distillation under structured compression

This repository has one supported experimental workflow: compare **vanilla
teacher-to-student KL distillation** with **Dual-constrained KD** while learning
exact-budget structured FFN channel gates.

The supported empirical grid is intentionally fixed:

- models: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct;
- datasets: ARC-Challenge, MMLU-AUX, MMLU-Pro, and HellaSwag;
- methods: `kl_only` and `dual_kl`;
- FFN retention: 25%, 50%, and 75%; and
- default training: one seed and two exact without-replacement epochs.

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
