# Iteration 6 Plan: Uncertainty-Guided Progressive Densification

## Working Title

**U-Dense: Uncertainty-Guided Progressive Densification for Adaptive LLM Inference**

Alternative titles:

- **When Does a Sparse Language Model Need More Capacity?**
- **Uncertainty-Guided Dynamic Sparsity for Small Language Models**
- **Spend Capacity Where It Helps: Progressive Densification for Efficient LLM Inference**

---

# 1. Motivation

Iterations 1–5 focused primarily on **static quantization** and repeatedly found that one fixed compression policy is difficult to optimize for all examples:

- teacher uncertainty did not produce a better static bit allocation;
- probability-based trust-critical layers largely overlapped with capability-critical layers;
- quantization changed uncertainty and routing behavior substantially;
- routing drift was not always harmful;
- cascade-aware static mixed precision failed to generalize;
- capability-constrained static search also failed on held-out ARC.

These results suggest that the relevant heterogeneity may be:

\[
\boxed{
\text{across inputs rather than across layers}
}
\]

A single compressed model may be unnecessarily expensive for easy inputs and insufficiently capable for difficult inputs.

Iteration 6 therefore moves from:

\[
\text{one static compressed SLM}
\]

to:

\[
\boxed{
\text{input-adaptive model capacity}
}
\]

using **structured sparsity**.

The central hypothesis is:

> A highly sparse SLM can handle easy examples cheaply, while uncertainty can identify examples for which restoring model capacity is likely to improve the prediction.

---

# 2. Core Idea

Construct nested structured sparse variants of the same model:

\[
S_{50}
\subset
S_{25}
\subset
S_{0}
\]

where:

- \(S_{50}\): approximately 50% structured sparsity;
- \(S_{25}\): approximately 25% structured sparsity;
- \(S_0\): dense model.

The inference policy starts from the cheapest model:

\[
S_{50}(x).
\]

Estimate whether additional capacity is likely to help.

Then:

\[
S_{50}
\rightarrow
S_{25}
\rightarrow
S_0
\]

only when predicted benefit exceeds the cost of densification.

Optional final extension:

\[
S_0
\rightarrow
L_{7B}
\]

for examples where even the dense SLM is unlikely to be sufficient.

---

# 3. Main Research Questions

## RQ1 — Does uncertainty predict when densification helps?

For an input \(x\), define:

\[
Y_{50\rightarrow25}(x)
=
\mathbf{1}
[
S_{50}(x)\neq y^*,
S_{25}(x)=y^*
]
\]

and:

\[
Y_{50\rightarrow0}(x)
=
\mathbf{1}
[
S_{50}(x)\neq y^*,
S_0(x)=y^*
].
\]

Ask whether uncertainty from the sparse model predicts these events.

The primary quantity is:

\[
AUROC
(
U(x),
Y_{\text{densification-help}}
).
\]

This is more important than ordinary error detection.

---

## RQ2 — Is predictive uncertainty sufficient, or is a learned capacity-uplift estimator better?

Compare:

1. margin uncertainty;
2. predictive entropy;
3. max-probability uncertainty;
4. cross-sparsity disagreement;
5. learned capacity-uplift probe.

Main hypothesis:

\[
\text{uplift-aware estimator}
>
\text{generic error uncertainty}
\]

for predicting when additional model capacity is useful.

---

## RQ3 — Can uncertainty-guided dynamic sparsity improve the accuracy–compute frontier?

Compare:

\[
\text{always sparse}
\]

\[
\text{always dense}
\]

\[
\text{random densification}
\]

\[
\text{uncertainty-guided densification}.
\]

Main question:

> At equal expected compute, does U-Dense achieve higher accuracy?

Or equivalently:

> At equal accuracy, can U-Dense use less compute?

---

## RQ4 — Does progressive densification outperform directly jumping from sparse SLM to a larger model?

Compare:

\[
S_{50}\rightarrow S_{25}\rightarrow S_0
\]

against:

\[
S_{50}\rightarrow L_{7B}.
\]

This tests whether restoring capacity inside the same model is a useful intermediate action.

This is an optional second-stage experiment.

---

# 4. Model Setup

## Primary Model

```text
Qwen2.5-1.5B-Instruct
```

Use this as the base model because:

- it was already used in Iterations 3–5;
- evaluation infrastructure exists;
- it is small enough for the current environment;
- it provides room for multiple structured-sparsity variants.

---

## Optional Fallback Model

```text
Qwen2.5-7B-Instruct
```

Only add after the sparse→dense mechanism is shown to work.

The main Iteration-6 claim should not depend on the 7B fallback.

---

# 5. Sparsity Method

## 5.1 Use Structured MLP-Channel Pruning

Do not begin with arbitrary unstructured weight pruning.

Prune complete FFN intermediate channels.

For each Transformer block:

\[
h
\rightarrow
W_{up}
\rightarrow
\sigma
\rightarrow
W_{down}.
\]

Rank FFN intermediate channels and remove the least important channels consistently from:

- up projection;
- gate projection where applicable;
- down projection.

This changes actual matrix dimensions and gives a path toward real speedups.

---

## 5.2 Channel Importance

Use an established activation-aware magnitude score rather than introducing a novel pruning criterion.

For channel \(j\):

\[
I_j
=
\|W_j\|
\cdot
E_x[|a_j(x)|].
\]

Possible practical alternatives:

- Wanda-style activation-aware magnitude;
- simple weight magnitude;
- activation magnitude.

Primary:

\[
\boxed{\text{activation-aware magnitude}}
\]

because the novelty should be in **uncertainty-guided dynamic capacity**, not in the pruning criterion.

---

# 6. Nested Sparse Models

Construct nested masks:

\[
M_{50}
\subset
M_{25}
\subset
M_{0}.
\]

Recommended initial levels:

```text
Dense: 0% sparsity
Medium: 25% sparsity
Aggressive: 50% sparsity
```

Optional stress test:

```text
60% sparsity
```

The 50%-sparse model should use a strict subset of the channels used by the 25%-sparse model.

Nestedness makes densification interpretable:

\[
\text{restore capacity}
=
\text{add previously removed channels}.
\]

---

# 7. Uncertainty Estimators

## U1 — Margin Uncertainty

For multiple-choice probabilities:

\[
U_{margin}(x)
=
1-
\left(
p_{(1)}(x)-p_{(2)}(x)
\right).
\]

Advantages:

- one forward pass;
- no training;
- negligible cost.

Primary no-training baseline.

---

## U2 — Predictive Entropy

\[
U_H(x)
=
-\sum_k p_k(x)\log p_k(x).
\]

Another zero-training baseline.

---

## U3 — Maximum-Probability Uncertainty

\[
U_{max}(x)
=
1-\max_kp_k(x).
\]

Use mainly as a conventional reference.

---

## U4 — Cross-Sparsity Instability

If both \(S_{50}\) and \(S_{25}\) have been executed:

\[
U_{JS}(x)
=
JS(
p_{50}(x),
p_{25}(x)
).
\]

Also evaluate:

### Prediction disagreement

\[
D_{pred}(x)
=
\mathbf1[
\hat y_{50}(x)
\neq
\hat y_{25}(x)
].
\]

### Confidence shift

\[
\Delta m(x)
=
m_{25}(x)-m_{50}(x).
\]

These measure prediction stability under restoration of model capacity.

---

# 8. Primary Learned Estimator: Capacity-Uplift Probe

Generic uncertainty asks:

\[
P(S_{50}\text{ is wrong}\mid x).
\]

But densification should occur only if a denser model is likely to improve the result.

Define:

\[
Y_{50\rightarrow25}
=
\mathbf1[
S_{50}\neq y^*,
S_{25}=y^*
].
\]

Similarly:

\[
Y_{50\rightarrow0}
=
\mathbf1[
S_{50}\neq y^*,
S_0=y^*
].
\]

Train:

\[
r_\theta(\phi_{50}(x))
\rightarrow
P(Y_{50\rightarrow25}=1)
\]

or:

\[
P(Y_{50\rightarrow0}=1).
\]

---

## 8.1 Probe Features

Start with inexpensive scalar features:

\[
\phi(x)
=
[
H,
margin,
p_{max},
logit\ gap,
probability\ vector
].
\]

Then optionally add a pooled hidden-state representation:

\[
h_{final}(x).
\]

Recommended models:

```text
Logistic regression
2-layer MLP
```

Use logistic regression as the main lightweight baseline.

Do not begin with a large neural router.

---

# 9. Cost-Aware Densification Value

For transition:

\[
s\rightarrow s',
\]

define:

\[
DV_{s\rightarrow s'}(x)
=
E[
\Delta Q_{s\rightarrow s'}(x)
]
-
\lambda
\Delta C_{s\rightarrow s'}.
\]

Where:

- \(\Delta Q\): expected quality improvement;
- \(\Delta C\): additional compute cost.

Densify when:

\[
DV_{s\rightarrow s'}(x)>0.
\]

For the first experiment, approximate quality using correctness probability.

---

# 10. U-Dense Runtime Policy

## Two-Level Version

Start with:

\[
S_{50}.
\]

If:

\[
r_\theta(x)\le\tau,
\]

return sparse prediction.

Otherwise:

\[
S_{50}\rightarrow S_0.
\]

This is the simplest experiment.

---

## Three-Level Version

After the two-level experiment passes:

\[
S_{50}
\rightarrow
S_{25}
\rightarrow
S_0.
\]

Policy:

\[
a(x)
=
\begin{cases}
S_{50}, & DV_{50\rightarrow25}\le0\\
S_{25}, & DV_{50\rightarrow25}>0,\ DV_{25\rightarrow0}\le0\\
S_0, & DV_{25\rightarrow0}>0.
\end{cases}
\]

---

## Optional Four-Level Version

Only after the sparse/dense hierarchy is positive:

\[
S_{50}
\rightarrow
S_{25}
\rightarrow
S_0
\rightarrow
L_{7B}.
\]

This unifies:

- sparsity adaptation;
- model escalation.

---

# 11. Dataset

## 11.1 Primary Development Dataset — ARC-Challenge

Use the existing experimental infrastructure.

### Training / Calibration

```text
ARC-Challenge train
~750 examples
```

Use for:

- activation collection for pruning;
- uncertainty threshold calibration;
- uplift probe training;
- policy development.

Because this set is small, use cross-validation for learned probes.

Recommended:

```text
5-fold stratified cross-validation
```

during development.

---

## 11.2 Held-Out ARC Evaluation

```text
ARC-Challenge validation
299 examples
```

Use only after:

- masks;
- uncertainty estimator;
- probe architecture;
- thresholds;
- operating points

are frozen.

---

# 12. Main Scale-Up Benchmark — MMLU-Pro

Only after a positive ARC result.

Reasons:

- more examples;
- harder reasoning;
- multiple-choice probabilities;
- broad domains;
- better statistical power.

Suggested:

```text
Calibration: MMLU-Pro validation
Evaluation: 2,000–full test set
```

If compute is restrictive:

```text
1,000 deterministic category-stratified test examples
```

to maintain continuity with Iteration 3.

---

# 13. Optional Hard Benchmark — GPQA Diamond

Run only if the method survives MMLU-Pro.

Purpose:

- difficult reasoning;
- larger potential gain from restoring model capacity;
- test whether dynamic sparsity becomes more useful as task difficulty increases.

---

# 14. Phase 0: Static Sparse-Model Sanity Check

Before any routing experiment, evaluate:

\[
S_0,\quad S_{25},\quad S_{50}.
\]

Report:

- accuracy;
- prediction agreement;
- output KL;
- uncertainty/error AUROC;
- approximate FLOPs.

Required condition:

> Sparsity levels must produce a meaningful capability hierarchy.

Desired:

\[
Acc(S_0)
>
Acc(S_{25})
>
Acc(S_{50})
\]

without \(S_{50}\) becoming nearly random.

If 50% sparsity collapses the model, reduce aggressive sparsity to:

```text
40%
```

or:

```text
30%
```

before studying adaptive routing.

---

# 15. Phase 1: Does Uncertainty Predict Densification Benefit?

For every ARC calibration example, cache:

\[
p_0(x),
p_{25}(x),
p_{50}(x).
\]

Construct:

\[
Y_{50\rightarrow25},
\quad
Y_{50\rightarrow0},
\quad
Y_{25\rightarrow0}.
\]

Evaluate each uncertainty estimator against each target.

---

## Primary Metrics

### AUROC

\[
AUROC(U,Y_{densify}).
\]

### AUPRC

Important because useful densification events may be rare.

### Precision@k

At:

```text
top 10%
top 20%
top 30%
```

highest predicted benefit.

This directly measures whether a compute-limited policy selects useful examples.

---

# 16. Phase 1 Go/No-Go

Proceed to dynamic-policy evaluation if at least one cheap estimator or learned uplift probe shows useful densification discrimination.

Suggested internal heuristic:

\[
AUROC \ge 0.60
\]

and better than generic sparse-model error uncertainty.

More important than an absolute threshold:

\[
AUROC_{uplift-probe}
>
AUROC_{entropy/error}
\]

with stable cross-validation.

If no estimator can predict when densification helps:

> stop the direction before implementing dynamic sparse inference.

---

# 17. Phase 2: Two-Level Dynamic Sparsity

Policy:

\[
S_{50}
\rightarrow
S_0.
\]

At calibration-selected densification rates:

\[
r
\in
\{0.10,0.20,0.30,0.50\},
\]

compare which examples are densified.

---

# 18. Baselines

## B0 — Always Dense

\[
S_0
\]

Maximum SLM compute.

---

## B1 — Always Medium Sparse

\[
S_{25}
\]

---

## B2 — Always Aggressively Sparse

\[
S_{50}
\]

Minimum compute baseline.

---

## B3 — Random Densification

Starting from \(S_{50}\), densify a random:

```text
10%
20%
30%
50%
```

of examples.

Run at least:

```text
10 random seeds
```

---

## B4 — Error-Oracle Densification

Densify examples where \(S_{50}\) is wrong first.

This is an optimistic error-routing upper bound.

---

## B5 — Uplift-Oracle Densification

Rank examples by:

\[
Y_{50\rightarrow0}.
\]

This directly represents the best possible policy for identifying when densification fixes the sparse model.

This is the most important oracle.

---

## B6 — Margin Routing

Densify highest-margin-uncertainty examples.

---

## B7 — Entropy Routing

Densify highest-entropy examples.

---

## B8 — Max-Probability Routing

Conventional uncertainty baseline.

---

## B9 — Learned Error Probe

Train:

\[
P(S_{50}\text{ wrong}).
\]

This controls for whether uplift-specific training is necessary.

---

## B10 — Capacity-Uplift Probe

Proposed method:

\[
P(S_0\text{ fixes }S_{50}).
\]

---

# 19. Phase 3: Three-Level Progressive Densification

Only after the two-level policy is positive.

Compare:

\[
S_{50}\rightarrow S_{25}\rightarrow S_0
\]

against:

\[
S_{50}\rightarrow S_0.
\]

Main question:

> Does the intermediate sparse level provide a better compute–accuracy frontier?

---

# 20. Compute Metric

For the controlled experiment, estimate expected compute:

\[
C_{\pi}
=
E_x[
C(a(x))
].
\]

Normalize dense SLM cost:

\[
C(S_0)=1.
\]

Approximate:

\[
C(S_{25})<1,
\qquad
C(S_{50})<C(S_{25}).
\]

Use analytical FLOPs from retained channel counts initially.

Do **not** claim latency speedup from FLOP estimates.

---

# 21. Real Runtime Validation

Only after the method is positive in simulation.

Create actually smaller structured-pruned checkpoints.

Measure:

- model size;
- peak memory;
- prefill latency;
- decoding latency;
- tokens/sec;
- end-to-end expected latency under the dynamic policy.

This should use the same hardware for all models.

---

# 22. Main Evaluation Metrics

## Standalone Metrics

For each sparsity level:

- accuracy;
- prediction agreement with dense;
- output KL;
- ECE;
- error AUROC;
- AURC.

---

## Densification-Prediction Metrics

Primary:

- AUROC for densification benefit;
- AUPRC;
- Precision@10%;
- Precision@20%;
- Precision@30%.

---

## Dynamic-System Metrics

### Final accuracy

\[
Acc_{\pi}.
\]

### Dense execution rate

\[
DR_0
=
P[a(x)=S_0].
\]

### Medium execution rate

\[
DR_{25}
=
P[a(x)=S_{25}].
\]

### Expected compute

\[
E[C].
\]

### Densification correction rate

\[
DCR
=
P[
S_{sparse}\neq y^*,
S_{dense}=y^*
\mid
\text{densified}
].
\]

### Wasteful densification

\[
WD
=
P[
S_{sparse}=y^*
\mid
\text{densified}
].
\]

---

# 23. Main Paper Frontier

The central plot should be:

\[
\boxed{
\text{accuracy}
\quad\text{vs.}\quad
\text{expected compute}
}
\]

Curves/points for:

- always dense;
- always sparse;
- random densification;
- entropy;
- margin;
- error probe;
- uplift probe;
- oracle uplift.

A successful method should move the Pareto frontier outward.

---

# 24. Additional Main Figures

## Figure 1 — Accuracy vs Sparsity

Show static:

\[
S_0,\quad S_{25},\quad S_{50}.
\]

---

## Figure 2 — Can Uncertainty Predict Densification Benefit?

AUROC/AUPRC for:

- margin;
- entropy;
- max probability;
- error probe;
- uplift probe.

---

## Figure 3 — Accuracy vs Expected Compute

Main system result.

---

## Figure 4 — Densification Outcome Breakdown

At a fixed compute budget:

- beneficial densification;
- unnecessary densification;
- densification with no correction;
- missed corrections.

---

## Figure 5 — Cross-Sparsity Instability

Plot:

\[
JS(p_{50},p_{25})
\]

against probability that:

\[
S_{25}
\]

or:

\[
S_0
\]

corrects \(S_{50}\).

---

# 25. Statistical Analysis

Use paired evaluation because every policy acts on the same examples.

### AUROC/AUPRC

Bootstrap confidence intervals.

### Dynamic accuracy

Paired bootstrap CI.

### Accuracy comparison at fixed compute

Bootstrap:

\[
\Delta Acc
=
Acc_{U-Dense}
-
Acc_{baseline}.
\]

### Random routing

Report mean ± SD over at least 10 seeds.

---

# 26. Important Ablations

## A1 — Generic Error vs Capacity Uplift

Compare:

\[
P(S_{50}\text{ wrong})
\]

against:

\[
P(S_0\text{ fixes }S_{50}).
\]

This is the most important conceptual ablation.

---

## A2 — Scalar Features vs Hidden-State Probe

Compare:

```text
entropy/margin/logits only
```

against:

```text
+ pooled hidden state
```

---

## A3 — Two-Level vs Three-Level Policy

Compare:

\[
50\%\rightarrow0\%
\]

against:

\[
50\%\rightarrow25\%\rightarrow0\%.
\]

---

## A4 — Sparsity Levels

Compare:

```text
25% / 50%
```

with:

```text
20% / 40%
```

if the aggressive model is too degraded.

---

## A5 — Nested vs Independently Pruned Models

Primary method uses nested masks.

An optional ablation can compare independently optimized sparse checkpoints.

---

## A6 — Cross-Sparsity Disagreement

Test whether:

\[
JS(p_{50},p_{25})
\]

adds information beyond sparse-model entropy and margin.

---

# 27. Success Criteria

## Strong Positive Result

Proceed to MMLU-Pro if:

1. uncertainty or uplift probe predicts densification benefit substantially better than random;
2. the uplift-specific probe improves over generic error prediction;
3. uncertainty-guided densification improves held-out ARC accuracy at matched expected compute;
4. gains are not explained purely by running the dense model more often.

Ideal result:

\[
Acc(U\text{-Dense})
>
Acc(\text{entropy routing})
\]

at the same expected compute.

---

## Very Strong Result

A three-level policy:

\[
S_{50}\rightarrow S_{25}\rightarrow S_0
\]

strictly improves the compute–accuracy Pareto frontier over:

- every static sparsity level;
- random routing;
- conventional uncertainty routing.

---

## Negative Result

Stop if:

- sparse-model uncertainty does not predict densification benefit;
- uplift probe cannot beat generic error prediction;
- dynamic routing does not improve accuracy at matched compute;
- the gains disappear on held-out data.

If negative, the broader conclusion becomes:

> uncertainty may identify model error but does not reliably identify when restoring compressed capacity will correct that error.

That would still be scientifically informative given Iterations 1–5.

---

# 28. Minimum Viable Iteration 6

## Model

```text
Qwen2.5-1.5B-Instruct
```

## Dataset

```text
ARC train: ~750
ARC validation: 299
```

## Sparse Variants

```text
Dense
25% MLP-channel sparsity
50% MLP-channel sparsity
```

## Pruning

```text
Activation-aware magnitude
Nested structured FFN-channel masks
```

## UQ Methods

```text
Margin
Entropy
Max probability
Logistic error probe
Logistic uplift probe
```

## Dynamic Policy

First:

```text
50% sparse → dense
```

Then, only if positive:

```text
50% → 25% → dense
```

## Baselines

```text
Always dense
Always 25% sparse
Always 50% sparse
Random densification
Margin routing
Entropy routing
Error probe
Uplift probe
Error oracle
Uplift oracle
```

## Primary Gate

The first gate is:

\[
\boxed{
\text{Can we predict when densification helps?}
}
\]

Do not build a runtime system until this is answered positively.

---

# 29. Recommended Execution Order

## Stage 1 — Build Sparse Variants

- collect calibration activations;
- compute structured channel importance;
- construct nested 25% and 50% masks;
- evaluate static accuracy.

---

## Stage 2 — Cache Outputs

For every ARC calibration example:

- dense probabilities;
- 25%-sparse probabilities;
- 50%-sparse probabilities;
- hidden-state features if used.

---

## Stage 3 — Build Densification Labels

Create:

\[
Y_{50\rightarrow25}
\]

\[
Y_{50\rightarrow0}
\]

\[
Y_{25\rightarrow0}.
\]

Inspect event frequency before training any probe.

---

## Stage 4 — UQ Gate

Evaluate:

- margin;
- entropy;
- max probability;
- generic error probe;
- uplift probe.

Use cross-validation.

If none predict densification benefit:

> STOP.

---

## Stage 5 — Two-Level Dynamic Policy

Evaluate:

\[
S_{50}\rightarrow S_0
\]

at fixed compute/densification budgets.

Freeze thresholds.

---

## Stage 6 — Held-Out ARC

Use the untouched 299-example validation set.

Compare all baselines using paired statistics.

---

## Stage 7 — Three-Level Policy

Only if two-level policy passes.

---

## Stage 8 — MMLU-Pro

Only after positive held-out ARC.

---

## Stage 9 — Real Runtime

Only after cross-benchmark replication.

Measure actual latency and memory.

---

# 30. Intended Contribution

The main claim should not be:

> uncertainty tells us which weights to prune.

Instead:

> **Uncertainty tells us when a compressed model needs additional capacity.**

The proposed framework treats sparsity as a runtime resource-allocation variable:

\[
\boxed{
\text{uncertainty}
\rightarrow
\text{expected value of densification}
\rightarrow
\text{adaptive model capacity}
}
\]

The strongest conceptual contribution is:

> **Predicting whether a sparse model is wrong is not enough; an adaptive compression policy should predict whether spending additional capacity will actually improve the answer.**

This connects uncertainty estimation, dynamic sparsity, and value-based adaptive computation in a single framework.

---

# 31. Immediate Execution Checklist

- [ ] Build dense Qwen2.5-1.5B baseline.
- [ ] Collect ARC-train FFN activations.
- [ ] Compute activation-aware channel importance.
- [ ] Create nested 25% and 50% structured sparse variants.
- [ ] Evaluate static sparse-model accuracy.
- [ ] Adjust sparsity if 50% causes collapse.
- [ ] Cache dense/25%/50% probabilities on ARC train.
- [ ] Create densification-benefit labels.
- [ ] Measure label prevalence.
- [ ] Evaluate margin AUROC/AUPRC for densification benefit.
- [ ] Evaluate entropy AUROC/AUPRC.
- [ ] Evaluate max-probability baseline.
- [ ] Train cross-validated sparse-model error probe.
- [ ] Train cross-validated capacity-uplift probe.
- [ ] Compare error prediction vs uplift prediction.
- [ ] Evaluate Precision@10/20/30% compute budgets.
- [ ] Freeze the best estimator.
- [ ] Build 50%-sparse → dense routing policy.
- [ ] Compare with random and uncertainty baselines.
- [ ] Evaluate held-out ARC validation once.
- [ ] Bootstrap accuracy-at-compute differences.
- [ ] If positive, add 25% intermediate model.
- [ ] Test cross-sparsity disagreement features.
- [ ] If still positive, replicate on MMLU-Pro.
- [ ] Only then add Qwen2.5-7B fallback or real runtime measurements.
