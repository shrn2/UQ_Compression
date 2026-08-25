# Dual-KD vs vanilla KL: full supported grid

Executed 2026-08-24/25 on Nova. Complete grid over the supported configuration
in `configs/experiment.yaml`, extended with a third model.

## Scope

- Models: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct (all BF16)
- Datasets: ARC-Challenge, MMLU-AUX, MMLU-Pro, HellaSwag (official splits)
- Methods: `kl_only`, `dual_kl`
- Retentions: 0.75, 0.50, 0.25
- 72 masks, one seed, two exact without-replacement epochs

`validate_results.py` passed: 12 calibrations, 72 training conditions, 144
prediction files, dataset partition counts and ID disjointness verified.

## Result: the entropy constraint binds; accuracy pays for it

**Uncertainty fidelity.** Entropy MAE to the dense teacher improves in 30 of 36
cells, scaling with compression pressure. On MMLU-AUX at 25% retention:
1.5B 0.6324 -> 0.2842 (-55%), 3B 0.8238 -> 0.1733 (-79%), 7B 0.6413 -> 0.1547 (-76%).

HellaSwag is the control: vanilla KL already reaches 0.02-0.09 entropy error
there, so the constraint has nothing to close and Dual-KD changes nothing
(delta within +/-0.006). The effect appears only where the fidelity gap is wide.

**Accuracy.** Dual-KD improves held-out accuracy in only 8 of 36 cells, and
loses most where the entropy gains are largest: 7B MMLU-AUX at 25% retention
falls 0.3816 -> 0.2689 (-11.3 points). ECE tracks accuracy rather than fidelity,
also improving in only 8 of 36.

**Selective prediction.** Of 72 paired bootstraps (5,000 replicates) on error
AUROC and AUPRC, 52 intervals cross zero; of the 20 that resolve, 13 favour
`kl_only` and 7 favour `dual_kl`. The clearest exception is the most aggressive
setting: 7B MMLU-AUX at 25% retention gains +0.1123 AUPRC [+0.0976, +0.1274].

The warranted conclusion is a trade-off, not a win: Dual-KD does what it is
designed to do, and costs capability nearly everywhere the constraint binds.

## Configuration changes

- Added `7b_bf16: Qwen/Qwen2.5-7B-Instruct` to the model grid (48 -> 72 cells).
  `config.py` validates methods and datasets exactly but leaves `models` open,
  so this is a supported extension.
- Raised HellaSwag `choice_microbatch_size` from 2 to 4. Measured 0.52 -> 0.37
  s/step at equal peak memory. Microbatching is gradient accumulation within a
  step, so this is numerically neutral and applied to both methods equally.

## Limitations

**Single seed.** `training.seed: 1`. The paired bootstrap resamples examples,
not training runs, so intervals capture example-level uncertainty only. Treat
every comparison as seed-sensitive.

**3B / MMLU-AUX is provisional.** After an out-of-memory reschedule, those three
cells trained `kl_only` on A100 (sm_80) and `dual_kl` on RTX PRO 6000 (sm_120),
so their difference carries a hardware term that cannot be separated from the
method effect. These rows include the -79% headline entropy figure. The 7B
result (-76%) comes from architecture-matched cells and does not depend on them.
The remaining 32 complete pairs are matched within sm_80.

**Claim boundary.** Masks are exact top-k logical channel gates applied to frozen
BF16 weights. These establish model quality only; no packed-kernel latency,
memory, or energy claim follows.

## Memory notes

Peak training memory is driven by prompt length, not choice count, and cannot be
reduced by `choice_microbatch_size` because the autograd graph retains every
microbatch until backward.

- 1.5B fits every card, including 40GB A100-PCIE.
- 3B needs 80GB+ for MMLU-AUX and MMLU-Pro.
- 7B needs 96GB for MMLU-AUX and MMLU-Pro; its prompt tail (p99 811, max 1367
  tokens) exceeds 80GB. ARC and HellaSwag fit in 80GB.
