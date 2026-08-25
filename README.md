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

## Results

Full grid: 3 models x 4 datasets x 2 methods x 3 retentions = 72 masks, one seed,
two exact without-replacement epochs. `validate_results.py` passed (12 calibrations,
72 training conditions, 144 prediction files).

**Dual-KD conserves the dense teacher's uncertainty and pays for it in accuracy.**
Each cell below is `dual_kl - kl_only`: negative entropy MAE means better fidelity,
positive accuracy means Dual-KD wins.

| model | dataset | Δacc .75 | Δacc .50 | Δacc .25 | ΔentMAE .75 | ΔentMAE .50 | ΔentMAE .25 |
|---|---|---|---|---|---|---|---|
| 1.5B | ARC-Challenge | +0.0017 | -0.0205 | +0.0026 | -0.0015 | -0.0055 | -0.0205 |
| 1.5B | MMLU-AUX | +0.0011 | -0.0154 | -0.0039 | -0.0657 | -0.1779 | -0.3482 |
| 1.5B | MMLU-Pro | -0.0162 | -0.0071 | +0.0033 | -0.0233 | -0.0382 | -0.0653 |
| 1.5B | HellaSwag | -0.0020 | -0.0093 | -0.0212 | -0.0003 | -0.0002 | +0.0031 |
| 3B | ARC-Challenge | +0.0171 | +0.0247 | -0.0247 | -0.0148 | -0.0167 | -0.0181 |
| 3B | MMLU-AUX | -0.0063 | -0.0352 | -0.0160 | -0.1568 | -0.3168 | -0.6505 |
| 3B | MMLU-Pro | -0.0017 | -0.0112 | -0.0042 | -0.0642 | -0.1564 | -0.3066 |
| 3B | HellaSwag | -0.0052 | -0.0132 | -0.0884 | +0.0002 | +0.0001 | +0.0057 |
| 7B | ARC-Challenge | -0.0085 | +0.0196 | -0.0102 | -0.0144 | -0.0230 | -0.0302 |
| 7B | MMLU-AUX | -0.0023 | -0.0274 | -0.1127 | -0.1209 | -0.3102 | -0.4866 |
| 7B | MMLU-Pro | -0.0029 | -0.0191 | -0.0432 | -0.0499 | -0.0896 | -0.1933 |
| 7B | HellaSwag | +0.0017 | -0.0032 | -0.0182 | +0.0003 | -0.0003 | +0.0019 |

Dual-KD is better on entropy fidelity in **30/36** cells, on accuracy in **8/36**,
and on ECE in **8/36**. Of 72 paired bootstraps on error AUROC/AUPRC, 7 favour
Dual-KD significantly and 13 favour vanilla KL; the rest cross zero.

HellaSwag is the control: vanilla KL already reaches 0.02-0.09 entropy error there,
so the constraint has nothing to close and Dual-KD changes nothing.

Per-cell numbers with dense references and bootstrap intervals are in
[runs/dual_kd_vs_kl/RUN.md](runs/dual_kd_vs_kl/RUN.md), which also records the
single-seed limitation and flags 3B/MMLU-AUX as provisional.

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
