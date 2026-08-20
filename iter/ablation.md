# UPGD Ablation Study Plan

## Goal

The purpose of these ablations is to isolate **why UPGD works** and determine whether its gains come from:

1. simply learning the pruning mask with gradient-based optimization,
2. matching the dense teacher distribution with KL distillation,
3. explicitly preserving predictive uncertainty,
4. preserving logit-margin structure,
5. adding worst-stratum robustness pressure.

The current full UPGD objective is:

\[
\mathcal{L}_{\text{UPGD}}
=
KL(p_{\text{dense}}\Vert p_{\text{mask}})
+
\lambda_H(H_{\text{dense}}-H_{\text{mask}})^2
+
\lambda_M(M_{\text{dense}}-M_{\text{mask}})^2
+
\lambda_R L_{\text{smooth-worst-stratum}}
\]

with the current reported weights:

- \(\lambda_H = 1.0\)
- \(\lambda_M = 0.5\)
- \(\lambda_R = 0.5\)

The model weights remain frozen. Only the FFN gate logits are optimized. The final pruning budget remains an exact **50% FFN-channel retention per layer**.

The central scientific question is:

> Does uncertainty-aware distillation improve structured pruning beyond what is obtained from ordinary KL-based gate optimization?

---

# 1. Experimental Controls

All ablations must use the **same pruning and optimization protocol** unless a component is explicitly being ablated.

Keep fixed:

- base model checkpoint,
- model precision / quantization,
- calibration dataset and split,
- evaluation dataset and split,
- prompt formatting,
- 50% FFN-channel retention budget,
- gate parameterization,
- gate initialization,
- optimizer,
- learning rate,
- number of optimization steps,
- stochastic masking schedule,
- top-k implementation,
- temperature schedule,
- random seed,
- batch composition,
- evaluation code.

The only intended difference between objective ablations should be the **loss terms being enabled**.

This is essential because the main comparison is intended to distinguish:

\[
\text{better mask optimization}
\]

from

\[
\text{better uncertainty-aware mask optimization}.
\]

---

# 2. Core Ablations

## A0. Dense Model

### Purpose

Provide the upper-reference model for:

- task accuracy,
- predictive distribution,
- entropy,
- logit margin,
- calibration,
- selective prediction metrics.

### Training

None.

### Evaluation

Run the untouched dense model on every evaluation example and cache:

- logits,
- answer probabilities,
- predicted class,
- correctness,
- predictive entropy,
- top-1 / top-2 logit margin.

These cached outputs should be reused for all pruning methods.

---

## A1. Random 50% Structured Pruning

### Purpose

Provide a sanity floor.

This tests whether the heuristic and learned pruning methods are doing something meaningfully better than arbitrary structured deletion.

### Implementation

For every FFN layer:

1. enumerate all intermediate FFN channels,
2. sample exactly 50% of channel indices uniformly at random,
3. retain those channels,
4. mask or physically remove the others.

No training or teacher signal is used.

### Seeds

Run at least 3 random masks.

If compute permits, use 5.

### Report

Report mean and standard deviation across random masks.

---

## A2. One-Shot Activation-Importance Pruning

### Purpose

Retain the current simple structured-pruning baseline.

This tests:

> Does optimized gate learning outperform a cheap, one-shot channel-ranking heuristic?

### Implementation

Use the existing implementation from the current experiments.

The report currently identifies this as **one-shot activation-importance pruning**. The exact activation-importance statistic should be documented explicitly in the paper and code.

For each FFN layer:

1. run the calibration set through the dense model,
2. compute the implementation's activation-importance score for each FFN channel,
3. rank channels within the layer,
4. retain the top 50%,
5. freeze the resulting mask,
6. evaluate without further optimization.

### Important documentation requirement

Before submission, record the exact mathematical score used.

For example, if the implementation uses mean absolute activation:

\[
I_j =
\mathbb{E}_{x,t}\left[|h_j(x,t)|\right]
\]

state that explicitly.

Do not leave this baseline described only as "activation importance."

---

# 3. Learned-Gate Ablations

These experiments use the **same UPGD gate optimization machinery**.

The dense model remains frozen.

The only difference is the objective.

---

## A3. Learned Gates — KL Only

### Purpose

This is the most important ablation.

It isolates the benefit of **joint learned mask optimization** from the contribution of uncertainty-aware terms.

### Objective

\[
\mathcal{L}_{\text{KL}}
=
KL(p_{\text{dense}}\Vert p_{\text{mask}})
\]

### Implementation

Use the current UPGD pipeline unchanged:

1. initialize one gate logit per FFN channel,
2. add the same annealed logistic noise,
3. construct the same exact top-k mask,
4. perform the masked forward pass,
5. compute only KL divergence,
6. backpropagate only into gate logits,
7. keep model weights frozen,
8. use the same number of optimization steps,
9. export the deterministic final top-k mask.

### Critical control

Do **not** remove stochastic gate exploration or change training steps for this variant.

Otherwise KL-only and full UPGD will not be directly comparable.

### Interpretation

Compare:

\[
\text{One-shot} \rightarrow \text{KL-only learned gates}
\]

to determine the contribution of learned joint optimization.

Then compare:

\[
\text{KL-only learned gates} \rightarrow \text{Full UPGD}
\]

to determine the contribution of uncertainty-aware objectives.

---

## A4. KL + Entropy Matching

### Purpose

Test whether explicit predictive-entropy preservation provides value beyond KL.

### Objective

\[
\mathcal{L}_{\text{KL+H}}
=
KL(p_{\text{dense}}\Vert p_{\text{mask}})
+
\lambda_H
(H_{\text{dense}}-H_{\text{mask}})^2
\]

Use:

\[
\lambda_H = 1.0
\]

to match the current full objective.

### Question

Does directly matching teacher entropy improve:

- entropy correlation,
- ECE,
- AURC,
- uncertainty-stratified accuracy,

without hurting task accuracy?

---

## A5. KL + Margin Matching

### Purpose

Test whether preserving the teacher's decision-margin geometry is useful independently of entropy matching.

### Objective

\[
\mathcal{L}_{\text{KL+M}}
=
KL(p_{\text{dense}}\Vert p_{\text{mask}})
+
\lambda_M
(M_{\text{dense}}-M_{\text{mask}})^2
\]

Use:

\[
\lambda_M = 0.5
\]

### Question

Does logit-margin matching preserve ranking/confidence behavior that KL alone does not?

---

## A6. KL + Entropy + Margin

### Purpose

Measure the contribution of the two uncertainty/fidelity terms without the worst-stratum robustness term.

### Objective

\[
\mathcal{L}_{\text{KL+H+M}}
=
KL(p_{\text{dense}}\Vert p_{\text{mask}})
+
(H_{\text{dense}}-H_{\text{mask}})^2
+
0.5(M_{\text{dense}}-M_{\text{mask}})^2
\]

### Interpretation

Compare:

\[
\text{KL-only}
\]

vs.

\[
\text{KL + entropy + margin}
\]

to estimate the total contribution of explicit uncertainty/fidelity preservation.

Then compare:

\[
\text{KL + entropy + margin}
\]

vs.

\[
\text{Full UPGD}
\]

to isolate the effect of worst-stratum robustness.

---

## A7. Full UPGD

### Objective

\[
\mathcal{L}_{\text{UPGD}}
=
KL(p_{\text{dense}}\Vert p_{\text{mask}})
+
(H_{\text{dense}}-H_{\text{mask}})^2
+
0.5(M_{\text{dense}}-M_{\text{mask}})^2
+
0.5L_{\text{smooth-worst-stratum}}
\]

### Batch construction

Preserve the current uncertainty-stratified calibration procedure:

- low dense-entropy stratum,
- middle dense-entropy stratum,
- high dense-entropy stratum.

Each calibration batch should maintain the same stratum sampling policy used in the current implementation.

### Purpose

This is the proposed method.

---

# 4. Minimum Experiment Matrix

The minimum scientifically defensible matrix is:

| Method | Learned gates | KL | Entropy | Margin | Worst-stratum |
|---|---:|---:|---:|---:|---:|
| Dense | No | No | No | No | No |
| Random 50% | No | No | No | No | No |
| One-shot activation | No | No | No | No | No |
| KL-only gates | Yes | Yes | No | No | No |
| KL + H + M | Yes | Yes | Yes | Yes | No |
| Full UPGD | Yes | Yes | Yes | Yes | Yes |

If compute permits, additionally run:

| Method | Learned gates | KL | Entropy | Margin | Worst-stratum |
|---|---:|---:|---:|---:|---:|
| KL + entropy | Yes | Yes | Yes | No | No |
| KL + margin | Yes | Yes | No | Yes | No |

The **minimum submission-critical comparison** is:

\[
\boxed{
\text{One-shot}
\rightarrow
\text{KL-only gates}
\rightarrow
\text{Full UPGD}
}
\]

---

# 5. Strong External Baseline

## FLAP

A stronger structured-pruning baseline should be added if implementation time permits.

### Purpose

The current one-shot baseline is useful but weak as the only published-method comparison.

A strong retraining-free structured pruning baseline helps answer:

> Is UPGD competitive with established structured-pruning methods, not just with a simple heuristic?

### Matching constraints

Use as close as possible to:

- the same base model,
- the same calibration data budget,
- the same FFN compression target,
- the same downstream evaluation sets.

If FLAP's native structural allocation does not exactly match the UPGD pruning unit, document the mismatch rather than forcing an invalid comparison.

---

# 6. Model / Dataset Plan

Use the current two model scales where feasible:

## 1.5B BF16

Current protocol:

- 28 layers,
- 8,960 FFN channels per layer,
- 4,480 retained per layer,
- exact 50% retention,
- 600 optimization steps,
- physical pruning/export available.

This should be the **primary full ablation model**, because the physical pruning implementation already exists.

## 3B NF4

Current protocol:

- 36 layers,
- 11,008 FFN channels per layer,
- 5,504 retained per layer,
- exact 50% retention,
- current resource-adapted run uses 300 optimization steps,
- current evaluation uses logical hooks around quantized Linear4bit modules.

Because the 3B protocol differs from the 1.5B protocol, avoid presenting model-scale comparisons as perfectly controlled unless the training budgets are matched.

### Recommended priority

Run the full ablation suite first on the 1.5B model.

For 3B, prioritize:

1. one-shot,
2. KL-only,
3. full UPGD.

This gives the most important cross-scale evidence without multiplying compute unnecessarily.

---

# 7. Evaluation Metrics

Every pruning method should be evaluated with the same metrics.

## Task performance

- exact-choice accuracy.

## Dense-model fidelity

- \(KL(p_{\text{dense}}\Vert p_{\text{mask}})\),
- dense-model prediction agreement,
- entropy Spearman correlation.

## Calibration / uncertainty

- ECE,
- AURC,
- predictive entropy error,
- margin error.

Useful additional summaries:

\[
\text{Entropy MAE}
=
\frac{1}{N}
\sum_i
|H_{\text{dense},i}-H_{\text{mask},i}|
\]

\[
\text{Margin MAE}
=
\frac{1}{N}
\sum_i
|M_{\text{dense},i}-M_{\text{mask},i}|
\]

These directly measure the quantities optimized by UPGD.

---

# 8. Uncertainty-Stratified Evaluation

Retain the current dense-entropy stratification.

Partition the evaluation set according to the **dense model's entropy**, not the pruned model's entropy.

Use:

- low uncertainty,
- middle uncertainty,
- high uncertainty.

For each stratum report:

- accuracy,
- UPGD-minus-baseline accuracy,
- KL,
- entropy error,
- ECE or calibration error if sample size permits.

This is especially important for evaluating the worst-stratum objective.

A useful table is:

| Method | Low-U Acc. | Mid-U Acc. | High-U Acc. | Worst-stratum Acc. |
|---|---:|---:|---:|---:|

The worst-stratum term is only scientifically supported if it improves the weakest uncertainty region consistently.

---

# 9. Seed Strategy

The current results should not remain single-run comparisons for the final workshop submission.

Recommended:

- 3 seeds for KL-only,
- 3 seeds for full UPGD,
- 3 random-pruning masks.

If compute permits:

- 3 seeds for all learned-gate ablations.

For each metric report:

\[
\text{mean} \pm \text{standard deviation}
\]

and preferably paired bootstrap intervals for accuracy differences.

Use the same evaluation examples for paired comparisons.

---

# 10. Logging Requirements

Each experiment should save:

```text
experiment_config.json
training_log.jsonl
final_mask.pt
evaluation_predictions.jsonl
metrics.json
```

## `experiment_config.json`

Store:

- model name,
- precision,
- dataset,
- calibration split,
- evaluation split,
- seed,
- pruning ratio,
- loss weights,
- number of steps,
- optimizer,
- learning rate,
- temperature schedule,
- noise schedule,
- batch/stratum policy.

## `training_log.jsonl`

At every logging interval record:

- step,
- total loss,
- KL loss,
- entropy loss,
- margin loss,
- robust loss,
- mask turnover,
- gate statistics.

For disabled loss terms, log zero explicitly.

This makes objective ablations easy to verify.

## `evaluation_predictions.jsonl`

Per example record:

```json
{
  "id": "...",
  "label": "...",
  "dense_prediction": "...",
  "masked_prediction": "...",
  "dense_probs": [],
  "masked_probs": [],
  "dense_entropy": 0.0,
  "masked_entropy": 0.0,
  "dense_margin": 0.0,
  "masked_margin": 0.0,
  "correct": true,
  "uncertainty_stratum": "high"
}
```

---

# 11. Suggested Code Structure

Avoid duplicating the full UPGD training implementation for every ablation.

Use a loss configuration.

Example:

```python
@dataclass
class LossConfig:
    kl_weight: float = 1.0
    entropy_weight: float = 0.0
    margin_weight: float = 0.0
    robust_weight: float = 0.0
```

Then define:

```python
ABLATIONS = {
    "kl_only": LossConfig(
        kl_weight=1.0,
    ),
    "kl_entropy": LossConfig(
        kl_weight=1.0,
        entropy_weight=1.0,
    ),
    "kl_margin": LossConfig(
        kl_weight=1.0,
        margin_weight=0.5,
    ),
    "kl_entropy_margin": LossConfig(
        kl_weight=1.0,
        entropy_weight=1.0,
        margin_weight=0.5,
    ),
    "full_upgd": LossConfig(
        kl_weight=1.0,
        entropy_weight=1.0,
        margin_weight=0.5,
        robust_weight=0.5,
    ),
}
```

The training loop should remain identical across configurations.

Pseudo-code:

```python
dense_output = dense_model(batch)

mask = gate_sampler(
    gate_logits,
    retention_ratio=0.50,
    temperature=temperature,
    noise_scale=noise_scale,
)

masked_output = masked_model(batch, mask)

loss = 0

loss += cfg.kl_weight * kl_loss(
    dense_output.probs,
    masked_output.probs,
)

loss += cfg.entropy_weight * entropy_match_loss(
    dense_output.probs,
    masked_output.probs,
)

loss += cfg.margin_weight * margin_match_loss(
    dense_output.logits,
    masked_output.logits,
)

loss += cfg.robust_weight * smooth_worst_stratum_loss(
    per_example_loss,
    dense_entropy_strata,
)

loss.backward()

optimizer.step()
```

This minimizes implementation drift between conditions.

---

# 12. Quality-Control Checks

Before launching the full experiment grid, verify:

## Mask budget

Every layer must satisfy:

\[
\frac{\text{retained channels}}
{\text{total channels}}
=
0.50
\]

exactly.

## Frozen weights

Confirm:

```python
assert all(
    not p.requires_grad
    for p in model.parameters()
    if p not in gate_parameters
)
```

Only gate logits should receive gradients.

## KL-only condition

Verify numerically that:

```text
entropy contribution = 0
margin contribution = 0
robust contribution = 0
```

while gate updates still occur.

## Full UPGD

Verify all four objective components are nonzero when expected.

## Evaluation parity

All conditions must use identical:

- examples,
- labels,
- answer extraction,
- prompt formatting,
- metric implementation.

---

# 13. Main Statistical Comparisons

The primary paper comparisons should be predeclared.

## Comparison 1: Learned optimization

\[
\text{KL-only gates}
-
\text{one-shot activation pruning}
\]

Tests:

> Does jointly optimizing the structured mask improve over one-shot channel importance?

## Comparison 2: Uncertainty awareness

\[
\text{Full UPGD}
-
\text{KL-only gates}
\]

Tests:

> Does uncertainty-aware distillation provide value beyond ordinary KL-based gate learning?

This is the key contribution test.

## Comparison 3: Robustness term

\[
\text{Full UPGD}
-
\text{KL+entropy+margin}
\]

Tests:

> Does worst-stratum training specifically improve performance in difficult uncertainty regions?

---

# 14. Interpretation Scenarios

## Outcome A

```text
One-shot << KL-only ≈ Full UPGD
```

Interpretation:

The main improvement comes from joint learned gate optimization.

The uncertainty-aware terms provide little additional benefit.

The paper should not strongly claim uncertainty preservation as the main reason for accuracy gains.

---

## Outcome B

```text
One-shot < KL-only < Full UPGD
```

Interpretation:

Both learned gating and uncertainty-aware distillation contribute.

This is the strongest result for the current hypothesis.

---

## Outcome C

```text
Accuracy:
KL-only ≈ Full UPGD

Uncertainty metrics:
Full UPGD >> KL-only
```

Interpretation:

This is still a strong result.

The central claim becomes:

> Explicit uncertainty-preservation objectives improve the fidelity and calibration of structurally compressed models without sacrificing task accuracy.

For the LIGHT workshop, this may be more compelling than a raw-accuracy-only story.

---

## Outcome D

```text
Full UPGD < KL-only
```

Interpretation:

The current uncertainty objective is over-constraining pruning.

Investigate:

- loss weighting,
- calibration-set composition,
- entropy strata,
- shorter optimization budget,
- model/dataset sensitivity.

Do not hide this result.

It can motivate a narrower claim about when uncertainty-aware compression helps.

---

# 15. Recommended Execution Order

To minimize compute and maximize information gained:

## Phase 1 — Essential causal ablation

Run on 1.5B:

1. one-shot activation pruning,
2. KL-only gates,
3. full UPGD.

If KL-only and full UPGD are clearly different, proceed.

---

## Phase 2 — Component attribution

Run:

4. KL + entropy,
5. KL + margin,
6. KL + entropy + margin.

This determines which uncertainty component matters.

---

## Phase 3 — Baseline strengthening

Run:

7. random 50% pruning,
8. FLAP or another strong structured-pruning baseline.

---

## Phase 4 — Replication

Run 3 seeds for:

- KL-only,
- full UPGD,
- random pruning.

Repeat the core comparison on the 3B model.

---

## Phase 5 — Deployment measurements

For the physically pruned 1.5B model, measure:

- parameter count,
- model size,
- FLOPs,
- peak VRAM,
- latency,
- tokens/sec.

This connects the methodological result to actual lightweight-model deployment.

---

# 16. Submission-Critical Deliverables

Before the LIGHT workshop submission, aim to have:

- [ ] exact mathematical definition of the one-shot baseline,
- [ ] random-pruning sanity baseline,
- [ ] KL-only learned-gate ablation,
- [ ] full UPGD comparison,
- [ ] at least one uncertainty-component ablation,
- [ ] uncertainty-stratified results,
- [ ] at least 3 seeds for the core learned-gate comparison,
- [ ] one strong published structured-pruning baseline if feasible,
- [ ] physical compression measurements on 1.5B,
- [ ] matched evaluation tables across methods.

The most important scientific result is:

\[
\boxed{
\text{Full UPGD}
\quad \text{vs.} \quad
\text{KL-only learned gates}
}
\]

because this comparison directly tests whether the proposed uncertainty-aware objective contributes beyond standard distillation-based pruning.
