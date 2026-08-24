# Iteration 17: UPGD Light paper enhancement

This run follows `iter/upgd_light_paper_enhancement_plan.md` on an NVIDIA A100
80GB using BF16 Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct.

Study grid:

- retention: 75%, 50%, 25% FFN channels per layer;
- learned objectives: lambda UQ = 0, 0.5, 1.0;
- three seeds per learned condition;
- 600 gate steps for both model scales;
- 768 gate-training examples, 384 lambda-selection examples, and 1,536 final examples;
- HellaSwag, ARC-Challenge, and MMLU auxiliary, with nine dataset-by-entropy cells in calibration.

The 54 training jobs are submitted through `hpc/slurm/iter17_train_array.sbatch`.
Quality scoring uses logical channel hooks on BF16 models, with separate physical
exports and equivalence checks in `physical_manifest.json`. The FLAP comparison is
the published WIFV fluctuation metric applied at a fixed per-layer FFN budget; it is
explicitly marked as a matched implementation rather than a reproduction of FLAP's
global attention/MLP allocation and bias-compensation path.

Expected handoff files:

- `run_manifest.json` — 54 learned masks plus heuristic controls;
- `analysis_manifest.json` — per-example paired metrics, bootstrap comparisons, severity/Pareto rows, and selected lambdas;
- `physical_manifest.json` — physical checkpoint dimensions, memory, and equivalence checks;
- `deployment/` — fixed prompt/batch latency and throughput measurements;
- `report/report.html` — canonical portable technical report.

Completed handoff:

- independent validation passed 215/215 checks; the repository test suite passed 50/50 tests;
- selected lambda values were 1.0/1.0/1.0 for the 1.5B model and 1.0/0.5/1.0 for the 3B model at 25%/50%/75% retention;
- all 18 physical exports passed FP32 logical/physical equivalence at 1e-4 normalized-choice probability tolerance, with BF16 checkpoints saved for deployment;
- both deployment manifests contain 40 fixed-prompt benchmark rows; report packaging passed with structural-only verification because Chromium is unavailable on this node.
