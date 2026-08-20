# Iteration 11 Plan: Uncertainty-Preserving Gate Distillation (UPGD)

## Frozen question

Can joint, stochastic, exact-budget gate optimization produce a better static 50%-pruned model than one-shot activation importance while preserving the dense model's predictive uncertainty?

This plan is frozen after the Iteration 10 URSA gate failed and before any UPGD sparse-model outcome is scored.

## Why this method follows from the evidence

URSA changed only 0.62% of retained channels at 50% pruning and lost one accuracy point on held-out MMLU auxiliary data. Together with Iterations 2, 4, and 5, this shows that reweighting local importance—whether by uncertainty, cascade utility, or a constrained search—is unlikely to solve the main problem. Local scores are strongly coupled and their effects are non-additive.

UPGD changes the optimization level:

- all FFN gates are learned jointly through the complete frozen model;
- every forward pass uses an exact channel budget, so training matches physical export;
- stochastic top-k masks expose the optimizer to uncertainty over nearby compressed subnetworks;
- dense probability, entropy, and margin targets preserve both capability and predictive uncertainty;
- risk is aggregated robustly across low-, middle-, and high-entropy prompt strata;
- inference uses one deterministic physically pruned model with no gate parameters, router, or second pass.

## Model

For each transformer layer `l`, learn logits `a[l,g]` over FFN channels. At training step `t`, add annealed logistic noise and take the exact top `k=4480` of 8,960 channels. The forward gate is binary; gradients use a sigmoid straight-through relaxation around the current top-k boundary.

All Qwen weights are frozen. Gates multiply the gated-MLP hidden values immediately before `down_proj`, which is exactly equivalent to the channel subset exported by physical pruning.

For dense target distribution `p0` and gated distribution `q`, the per-example loss is

`KL(p0 || q) + (H(q)-H(p0))^2 + 0.5*(margin(q)-margin(p0))^2`.

The batch contains one example from each of three dense-entropy strata. The final objective is the mean loss plus `0.5` times a smooth worst-stratum loss with temperature `0.1`.

## Frozen configuration

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Calibration: the fixed 2,400 MMLU auxiliary training indices and dense probabilities cached before UPGD
- Initialization: matched mean-activation 50% mask from Iteration 10
- Sparsity: exactly 50% in every FFN layer
- Steps: 600
- Batch: one prompt from each entropy stratum (three total)
- Optimizer: Adam on gate logits only, learning rate `0.05`
- Gate temperature: exponential schedule `0.5 -> 0.1`
- Logistic noise: exponential schedule `0.5 -> 0.05`
- Gradient clipping: `5.0`
- Seed: `20260829`
- Evaluation: the fixed 600 MMLU auxiliary validation indices

## Baselines and gate

The primary comparator is the matched-data mean-activation 50% mask already scored in Iteration 10 (34.33% accuracy). URSA is a secondary diagnostic, not the baseline.

UPGD advances only if:

1. accuracy exceeds the mean-activation baseline by at least 2.0 percentage points;
2. the paired 95% bootstrap interval excludes zero;
3. every dense-uncertainty stratum is no worse than baseline;
4. entropy-rank Spearman correlation with dense is no worse than baseline;
5. dense-threshold conformal coverage is no worse than baseline by more than 2 points.

This stricter gate is appropriate because UPGD uses substantially more calibration compute than one-shot pruning.

## Resource and claim boundary

Only about 251k gate logits and their Adam state are trained. The 1.5B model is loaded once in BF16, its weights have no gradients, batches contain three prompts, and only answer-token logits are retained. This is designed for the available 8 GB GPU and 16 GB RAM.

Successful physical pruning establishes a parameter/FLOP proxy reduction. It does not establish kernel latency, throughput, memory, energy, or packed model size until measured separately.
