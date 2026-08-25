# CLAUDE.md

This file gives Claude Code repository-specific guidance. Keep changes focused on
the supported experimental workflow described below.

## Supported scope

This repository compares exactly two structured-compression objectives:

- `kl_only`: vanilla forward KL, `KL(p_teacher || p_student)`;
- `dual_kl`: forward KL with a dynamically updated entropy-conservation dual
  variable.

The supported datasets are ARC-Challenge, MMLU-AUX, MMLU-Pro, and HellaSwag.
The supported models and retention levels are defined in
`configs/experiment.yaml`. Do not reintroduce historical TUAC, UPGD, UBKL,
symmetric-KL, routing, quantization, or iteration-specific workflows.

Dense pretrained weights remain frozen. Training optimizes structured FFN gate
parameters and, for `dual_kl`, the dual state. Evaluation uses deterministic
exact-top-k masks. Do not describe logical sparsity as measured latency, energy,
or memory savings without packed-model measurements.

## Repository map

- `configs/experiment.yaml`: supported models, datasets, methods, retentions,
  training schedule, and artifact roots.
- `src/dualkd/`: data handling, objectives, gate modeling, training, scoring,
  evaluation, and metrics.
- `scripts/prepare_datasets.py`: normalize and validate the four datasets.
- `scripts/calibrate.py`: compute dense references and gate initialization.
- `scripts/run_experiment.py`: train and score matched conditions.
- `scripts/analyze_results.py`: compute matched metrics and paired analyses.
- `scripts/build_report.py`: generate the HTML results report.
- `scripts/validate_results.py`: validate the complete result bundle.
- `tests/`: unit tests for the supported implementation.

## Development workflow

Install development and model-runtime dependencies with:

```bash
python -m pip install -e ".[experiment,test]"
```

Before handing off a code change, run:

```bash
python -m pytest -q
python -m compileall -q src/dualkd scripts
git diff --check
```

The GPU smoke test additionally requires prepared data, a calibration artifact,
and access to the configured Hugging Face model:

```bash
python scripts/smoke_test.py --source-root <prepared-artifact-root>
```

## Experiment rules

- Keep KL and Dual-KD conditions matched in model weights, gate initialization,
  optimizer, schedule, data exposure, retention budget, and seed.
- Preserve the configured dataset splits and validate split sizes and example-ID
  disjointness.
- Derive reported values from machine-readable result artifacts; do not copy or
  edit numerical results manually in generated HTML.
- Keep teacher-uncertainty fidelity distinct from task accuracy, calibration,
  error detection, and selective-prediction utility.
- Do not fabricate missing results, confidence intervals, seeds, or experimental
  settings.
- Treat the single-seed results as seed-sensitive unless replicated evidence is
  available.

## Artifacts, credentials, and Git

- Do not commit datasets, model weights, checkpoints, predictions, NumPy arrays,
  W&B state, logs, or other large experiment outputs.
- `runs/` is local artifact storage. Only lightweight documentation and manifests
  explicitly exposed by `.gitignore` may be committed.
- `hpc/` contains local cluster scripts and must remain ignored. Do not force-add
  it or publish cluster account, partition, allocation, or job details.
- `paper/light2026_dualkd/` is an independent Git repository and is ignored by
  this parent repository.
- Keep Hugging Face, W&B, GitHub, and cluster credentials in the environment or
  an external secret manager. Never write credentials into source, configuration,
  logs, tests, or documentation.
- Preserve unrelated user changes in a dirty worktree and stage only files that
  belong to the requested change.
