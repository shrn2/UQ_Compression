# Iteration 10 Plan: Uncertainty-Robust Structured Allocation (URSA)

## Frozen question

Can dense-model predictive uncertainty improve a **single static compressed model** when it is used as a robust calibration environment, rather than as an input router or a direct “protect uncertain examples” weight?

The method and gate below are frozen before Iteration 10 model outcomes are scored.

## Evidence motivating the pivot

Nine completed iterations rule out several nearby ideas in the tested Qwen2.5 setup:

- Teacher entropy was negatively associated with 4-bit perturbation in Iteration 1 (`rho=-0.510`), so higher predictive uncertainty is not a reliable synonym for compression vulnerability.
- Block-level capability and uncertainty sensitivity were nearly identical in Iteration 2 (`rho=0.959-0.993`), leaving little independent signal for uncertainty-aware blockwise mixed precision.
- Static cascade-aware bit allocation and constrained search failed held-out gates in Iterations 4-5; additive local sensitivity did not predict the best whole-model allocation.
- Structured 25%/50% FFN pruning created a real capacity hierarchy in Iterations 6-7, and the dense model corrected many sparse errors, but entropy, max probability, and learned uplift probes did not reliably select those corrections.
- Capacity disagreement was learnable in Iteration 8, but it transferred poorly to positive dense advantage in Iteration 9. Frozen CCR did not beat ordinary uncertainty at the matched-compute system gate.

The surviving opportunity is to use uncertainty **offline while constructing the sparse model**, with no router and no second inference pass.

## Literature boundary

URSA is deliberately distinct from:

- GPTQ and SparseGPT, which use approximate second-order reconstruction objectives;
- Wanda and OWL, which use activation-aware weight importance and non-uniform layer sparsity;
- BayesQ, which allocates quantization precision under a weight-posterior expected loss;
- UNComp, which uses matrix entropy for adaptive hidden-state and KV-cache compression;
- recent conformal benchmarking showing that pruning/quantization can decouple accuracy and uncertainty and can produce threshold-like uncertainty inflation.

URSA contributes an uncertainty-stratified robust objective for **static structured FFN-channel selection**. The matched baseline uses the same model, examples, activations, sparsity, and physical pruning implementation.

## Method

For calibration prompt `x`, let the dense multiple-choice distribution be `p0(.|x)` and define normalized predictive entropy

`u(x) = H[p0(.|x)] / log |Y_x|`.

Partition the calibration corpus into three stable equal-frequency strata: low, middle, and high uncertainty. For FFN channel `g` in transformer block `l` and stratum `b`, compute

`I[b,l,g] = E[|z[l,g](x)| | x in b] * ||W_down[l,:,g]||_2`,

where `z` is the prompt-final gated-MLP activation. Normalize each `(b,l)` vector to unit mass so that each uncertainty stratum defines a channel-importance distribution.

For retained set `S`, the omitted importance in stratum `b` is

`L_b(S) = sum_{g not in S} I_norm[b,l,g]`.

URSA greedily minimizes the smooth worst-stratum loss

`tau * log sum_b exp(L_b(S)/tau)`

with `tau=0.05`. Every 64 selected channels, it refreshes the adversarial stratum weights `softmax(L/tau)` and ranks remaining channels by their weighted marginal reduction. The full order yields nested 25% and 50% pruning masks.

The matched baseline ranks channels by the global mean of the same per-stratum raw importance, weighted by stratum frequency. Thus any difference is due to robust uncertainty stratification, not data, sparsity, or scoring implementation.

## Fixed model and data

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Calibration corpus: the 2,400 fixed training indices from the existing 3,000-example MMLU auxiliary corpus
- Development evaluation: the 600 fixed validation indices, unseen by mask construction
- External exploratory evaluation: the existing 1,000-example MMLU-Pro cohort
- Sparsity: physical per-block FFN-channel pruning at 25% and 50%
- Uncertainty strata: three equal-frequency bins from dense normalized answer entropy
- Temperature: `0.05`
- Adversarial-weight refresh: every 64 selected channels
- Batch size: 64; models loaded sequentially on the 8 GB GPU

The 600-example split is the primary development test. MMLU-Pro is not confirmatory: its aggregate outcomes influenced the research pivot in Iteration 9, although its labels and examples do not influence URSA masks.

## Metrics

At each sparsity level, compare URSA with the matched mean-activation baseline on:

- overall accuracy and paired bootstrap interval;
- accuracy within each dense-uncertainty stratum;
- KL divergence from dense predictions;
- prediction agreement with dense;
- entropy-rank Spearman correlation with dense;
- ECE, Brier score, error AUROC/AUPRC, and AURC;
- split-conformal LAC marginal coverage and mean prediction-set size, both with a model-specific threshold and with the dense threshold transferred unchanged.

Compute is a transformer matrix-parameter proxy. No latency, memory, energy, packed-size, or throughput claim is authorized by this experiment.

## Frozen development gate

URSA passes only if all conditions hold on the 600-example development evaluation:

1. URSA minus matched-baseline accuracy is at least `+1.0` percentage point at 50% pruning.
2. URSA minus matched-baseline accuracy is at least `-0.5` point at 25% pruning.
3. No dense-uncertainty stratum loses accuracy at 50% pruning.
4. Entropy-rank preservation at 50% pruning is no worse than the matched baseline.

Conformal metrics are required diagnostics but are not included in the pass gate because one deterministic conformal split is too noisy to support a tight optimization constraint.

## Interpretation rules

- A failed gate falsifies this particular entropy-stratified minimax channel allocator, not uncertainty-aware compression in general.
- A positive development gate is not sufficient for a paper claim; it requires a freshly chosen benchmark and at least one additional model family.
- If URSA masks overlap the baseline almost completely, the main conclusion is that uncertainty environments do not materially change activation-aware channel selection.
- If masks differ but outcomes do not improve, the uncertainty objective changes the model but does not improve the compression frontier.
- Hyperparameters, strata, and the gate will not be changed after outcomes are read.
