# Iteration 7 Plan: Counterfactual Capacity Routing for Adaptive Sparse LLMs

## Working Title

**Counterfactual Capacity Routing: Predicting When a Sparse LLM Needs More Capacity**

Alternative titles:

- **Capacity Advantage Estimation for Adaptive Sparse Language Models**
- **Beyond Uncertainty: Predicting the Value of Densification in Sparse LLMs**
- **When Should a Sparse Language Model Become Dense?**
- **Adaptive Sparse Inference via Counterfactual Capacity Advantage**

---

# 1. Motivation

Iteration 6 established three important facts.

## 1.1 Structured sparsity creates a real capacity hierarchy

For Qwen2.5-1.5B-Instruct on ARC-Challenge:

- Dense accuracy: ~47.9%
- 25% FFN-channel sparsity: ~41.2%
- 50% FFN-channel sparsity: ~31.7%

The 50%-sparse model reduced the analytical matrix-compute proxy to ~0.559 of dense.

Therefore, structured FFN pruning creates a meaningful accuracy–compute hierarchy.

---

## 1.2 Useful densification opportunities clearly exist

A substantial number of examples are:

\[
S_{sparse}\text{ wrong},\quad S_{dense}\text{ correct}.
\]

In Iteration 6, approximately 24% of ARC calibration examples were corrected by dense inference after aggressive pruning.

Even more importantly, the held-out uplift oracle showed very large headroom.

At approximately 30% dense execution:

\[
Acc_{\text{uplift-oracle}}
\approx 58.5\%
\]

while:

\[
Acc_{\text{always dense}}
\approx 49.8\%.
\]

This means sparse and dense variants are **complementary**.

A router that knows which model is better per example can outperform either static model.

---

## 1.3 Generic uncertainty is not sufficient

Sparse-only signals such as:

- entropy;
- margin;
- max-probability uncertainty;

reached only moderate densification-benefit AUROC (~0.62–0.63).

The learned uplift probe also did not outperform those simple signals.

However, **cross-capacity prediction disagreement** between the aggressive and medium sparse variants reached a substantially stronger AUROC (~0.71) for predicting when dense correction occurs.

This suggests the relevant signal is not simply:

\[
\text{“Am I uncertain?”}
\]

but rather:

\[
\boxed{
\text{“Would more model capacity change or improve my prediction?”}
}
\]

Iteration 7 therefore shifts from **uncertainty-guided densification** to **counterfactual capacity advantage estimation**.

---

# 2. Central Hypothesis

For input \(x\), define the advantage of dense inference over sparse inference:

\[
A(x)
=
Q(S_0,x)-Q(S_s,x)
\]

where:

- \(S_s\): sparse SLM;
- \(S_0\): dense SLM;
- \(Q\): prediction quality.

The router should estimate:

\[
\boxed{
\hat A(x)
=
E[A(x)\mid \phi_s(x)]
}
\]

using information available from the sparse model.

Then densify when:

\[
\hat A(x)
>
\lambda \Delta C.
\]

The key hypothesis is:

> **A representation-level estimate of counterfactual capacity advantage can identify when restoring model capacity is useful better than generic predictive uncertainty.**

---

# 3. Main Research Questions

## RQ1 — Is signed capacity advantage more informative than a binary correction target?

Iteration 6 used:

\[
Y_+(x)
=
\mathbf1[
S_s(x)\neq y^*,
S_0(x)=y^*
].
\]

Iteration 7 instead defines:

\[
A_{01}(x)
=
\mathbf1[S_0(x)=y^*]
-
\mathbf1[S_s(x)=y^*].
\]

Therefore:

\[
A_{01}(x)\in\{-1,0,+1\}.
\]

Interpretation:

### \(+1\)

Sparse wrong, dense correct.

Densification is beneficial.

### \(0\)

Sparse and dense have the same correctness.

Densification has no correctness benefit.

### \(-1\)

Sparse correct, dense wrong.

Densification is harmful.

Main question:

> Does predicting signed advantage produce better routing than predicting only sparse-model error or dense correction?

---

## RQ2 — Can sparse hidden representations predict counterfactual dense advantage?

Given sparse-model hidden state:

\[
h_s^{(l)}(x),
\]

train:

\[
r_\theta(h_s^{(l)}(x))
\rightarrow
E[A(x)].
\]

Compare against output-level uncertainty.

Main hypothesis:

\[
\text{representation-level advantage probe}
>
\text{entropy/margin/max-probability}
\]

for selecting densification examples.

---

## RQ3 — Can a sparse representation predict cross-capacity instability without executing another model?

Iteration 6 found that:

\[
D_{s\rightarrow m}(x)
=
\mathbf1[
\hat y_s(x)\neq\hat y_m(x)
]
\]

was the strongest observed densification-benefit signal.

Train:

\[
g_\theta(h_s(x))
\rightarrow
P[
\hat y_s\neq\hat y_m
].
\]

This approximates the useful cross-sparsity signal **without actually executing the medium model**.

Question:

> Can predicted capacity instability serve as a cheap routing signal?

---

## RQ4 — Which starting sparsity level provides the best adaptive-compute opportunity?

Iteration 6 used:

\[
25\%,50\%.
\]

But 50% sparsity caused substantial degradation.

Iteration 7 should compare:

\[
s\in\{25\%,40\%,50\%\}
\]

or, if implementation is easier:

\[
s\in\{25\%,35\%,50\%\}.
\]

For each sparse variant, first estimate **oracle adaptive headroom**.

Question:

> Which sparse→dense pair gives the best potential accuracy–compute frontier before router learning?

---

# 4. Model Setup

## Base Model

```text
Qwen2.5-1.5B-Instruct
```

Reuse the Iteration-6 pruning infrastructure.

---

## Sparse Variants

Minimum:

```text
S25 — 25% FFN-channel sparsity
S50 — 50% FFN-channel sparsity
S0  — dense
```

Recommended addition:

```text
S40 — 40% FFN-channel sparsity
```

The exact intermediate level can be adjusted to the existing structured pruning implementation.

---

# 5. Structured Pruning Method

Reuse the Iteration-6 pruning method.

For FFN channel \(j\):

\[
I_j
=
E[|a_j(x)|]
\cdot
\|W^{down}_{:,j}\|_2.
\]

Retain the highest-scoring channels.

The same channel indices must be removed consistently from:

- gate projection;
- up projection;
- down projection.

Use **nested masks** where possible:

\[
M_{50}
\subset
M_{40}
\subset
M_{25}
\subset
M_0.
\]

Do not introduce a new pruning criterion in Iteration 7.

---

# 6. Phase 0 — Choose the Right Sparse Starting Point

Before training any new router, evaluate each sparse model:

\[
S_{25},S_{40},S_{50}.
\]

For each \(S_s\), compute:

### Static metrics

- accuracy;
- dense prediction agreement;
- output KL;
- ECE;
- error AUROC;
- AURC;
- analytical compute proxy.

---

## 6.1 Oracle Densification Frontier

For each sparsity level \(s\), define:

\[
A_{01}^{(s)}(x)
=
\mathbf1[S_0=y^*]
-
\mathbf1[S_s=y^*].
\]

At densification rates:

\[
r\in\{0.1,0.2,0.3,0.5\},
\]

construct the oracle policy by densifying examples with highest true advantage.

Measure:

\[
Acc^{oracle}_s(r).
\]

Compute:

\[
Gain^{oracle}_s(r)
=
Acc^{oracle}_s(r)
-
Acc(S_s).
\]

---

## 6.2 Oracle Efficiency

Define:

\[
OE_s(r)
=
\frac{
Acc^{oracle}_s(r)-Acc(S_s)
}{
C_{\pi_s(r)}-C(S_s)
}.
\]

This measures how much accuracy is recoverable per unit of extra compute.

Choose the primary sparse starting level based on:

- meaningful static efficiency;
- non-collapsed accuracy;
- large oracle headroom.

Do **not** automatically default to 50% sparsity.

---

# 7. Capacity Advantage Targets

## Target T1 — Binary Correction

Iteration-6 target:

\[
Y_+(x)
=
\mathbf1[
S_s\neq y^*,
S_0=y^*
].
\]

Keep only as a baseline.

---

## Target T2 — Signed Correctness Advantage

Primary discrete target:

\[
A_{01}(x)
=
\mathbf1[S_0=y^*]
-
\mathbf1[S_s=y^*].
\]

Values:

```text
+1 = dense beneficial
 0 = neutral
-1 = dense harmful
```

Train either:

### Regression

\[
r_\theta(x)
\rightarrow
E[A_{01}(x)]
\]

or:

### Three-class classification

\[
P(A=-1),P(A=0),P(A=+1).
\]

Expected advantage:

\[
\hat A
=
P(A=+1)-P(A=-1).
\]

---

## Target T3 — Loss-Based Advantage

Optional smoother target:

\[
A_{\ell}(x)
=
\ell(S_s(x),y^*)
-
\ell(S_0(x),y^*).
\]

For MCQ:

\[
\ell
=
-\log p(y^*).
\]

Positive value:

> dense assigns more probability to the correct answer.

Negative value:

> sparse assigns more probability to the correct answer.

This may provide a more data-efficient regression target than sparse binary outcomes.

---

# 8. Baseline Routing Signals

## B1 — Margin

\[
U_{margin}
=
1-(p_1-p_2).
\]

---

## B2 — Entropy

\[
U_H
=
-\sum_i p_i\log p_i.
\]

---

## B3 — Max-Probability Uncertainty

\[
U_{max}
=
1-\max_i p_i.
\]

---

## B4 — Sparse Error Probe

Train:

\[
P(S_s\neq y^*).
\]

This tests whether ordinary error prediction is sufficient.

---

## B5 — Iteration-6 Binary Uplift Probe

Train:

\[
P(S_s\neq y^*,S_0=y^*).
\]

---

# 9. Proposed Router 1 — Hidden-State Capacity Advantage Probe

## 9.1 Representation

Extract sparse residual-stream representations at selected layers.

Recommended:

\[
l\in
\left\{
\frac14L,
\frac12L,
\frac34L,
L
\right\}.
\]

For MCQ prompts, use:

```text
final prompt-token representation
```

or the representation corresponding to the answer-scoring position.

---

## 9.2 Probe Models

Start with:

```text
ridge regression
logistic regression
multinomial logistic regression
```

Do not use a large neural router initially.

Optional second stage:

```text
2-layer MLP
```

only if linear probing shows meaningful signal.

---

## 9.3 Probe Variants

### H1 — final-layer only

\[
h^{(L)}_s
\]

### H2 — concatenated selected layers

\[
[
h^{(L/4)},
h^{(L/2)},
h^{(3L/4)},
h^{(L)}
]
\]

### H3 — scalar uncertainty + hidden state

\[
[
h_s,
H,
margin,
p_{max}
].
\]

---

# 10. Proposed Router 2 — Capacity-Instability Probe

Iteration 6 found that actual cross-sparsity prediction disagreement was the strongest calibration signal.

Define:

\[
Y_{instability}(x)
=
\mathbf1[
\hat y_s(x)\neq\hat y_m(x)
].
\]

Train:

\[
g_\theta(h_s(x))
\rightarrow
P(Y_{instability}=1).
\]

This probe requires only sparse inference at deployment.

---

## 10.1 Alternative Soft Instability Target

Instead of binary disagreement:

\[
D_{JS}(x)
=
JS(
p_s(x),p_m(x)
).
\]

Train:

\[
g_\theta(h_s(x))
\rightarrow
D_{JS}(x).
\]

This may provide a smoother target.

---

# 11. Combined Counterfactual Router

If both probes show signal, combine:

\[
\phi_{route}(x)
=
[
\hat A(x),
\hat D_{instability}(x),
H_s,
margin_s
].
\]

Then fit a final lightweight ranking model:

\[
r(x)
\rightarrow
\text{expected benefit of dense execution}.
\]

Do this only after evaluating each signal independently.

---

# 12. Router Decision

For sparse level \(s\):

\[
DV_s(x)
=
\hat A_s(x)
-
\lambda
\left(
C(S_0)-C(S_s)
\right).
\]

Densify iff:

\[
DV_s(x)>0.
\]

For fixed-rate evaluation, ignore \(\lambda\) initially and rank examples by:

\[
\hat A_s(x).
\]

---

# 13. Dataset Protocol

## 13.1 ARC-Challenge — Development Only

Iteration 6 already consumed the ARC validation split.

Therefore, Iteration 7 should **not** treat ARC validation as a fresh confirmatory test.

Use:

```text
ARC train (~750)
```

for development through cross-validation.

ARC validation may be used only for:

- descriptive comparison with Iteration 6;
- sanity checks;
- exploratory analysis.

Do not use it to claim new held-out generalization.

---

## 13.2 Cross-Validation on ARC Train

Recommended:

```text
5-fold stratified cross-validation
```

All learned-probe predictions must be out-of-fold.

Evaluate:

- ranking quality;
- signed advantage prediction;
- policy accuracy at fixed dense rates.

Use this phase only as a **development gate**.

---

# 14. Main Fresh Evaluation Benchmark — MMLU-Pro

MMLU-Pro should be the next true held-out evaluation if the ARC-development gate passes.

## Calibration / Probe Training

Options:

### Preferred

Use MMLU-Pro validation plus additional training-like examples if available.

### Minimum

Use MMLU-Pro validation for threshold calibration and any light probe adaptation.

If training a representation probe on ARC and testing zero-shot on MMLU-Pro, treat that as a stronger OOD generalization experiment.

---

## Held-Out Evaluation

Prefer:

```text
2,000–full MMLU-Pro test examples
```

If compute is restrictive:

```text
1,000 deterministic category-stratified examples
```

Reuse the same sampling procedure as prior work if possible.

---

# 15. OOD Evaluation Modes

Evaluate two modes.

## Mode A — In-Domain Probe

Train/calibrate on MMLU-Pro validation.

Evaluate on MMLU-Pro test.

---

## Mode B — Cross-Dataset Probe

Train router on ARC.

Evaluate directly on MMLU-Pro without retraining.

This asks:

> Does the representation encode a task-general notion of capacity advantage?

This is scientifically interesting but should be secondary.

---

# 16. Core Evaluation Metric: Net Advantage Capture

At densification budget \(r\), let:

\[
D_r(x)
=
\mathbf1[
x\text{ selected for dense execution}
].
\]

Define:

\[
NAC(r)
=
E[
A_{01}(x)D_r(x)
].
\]

This directly measures net correctness gained by densification.

Higher is better.

---

# 17. Oracle Recovery

Define:

\[
OR(r)
=
\frac{
Acc_{\pi}(r)-Acc(S_s)
}{
Acc_{\pi_{oracle}}(r)-Acc(S_s)
}.
\]

Interpretation:

```text
0%   = random/no useful oracle recovery
100% = oracle capacity selection
```

This should be a central metric.

---

# 18. Ranking Metrics

For advantage estimation, evaluate:

### Spearman correlation

\[
\rho(
\hat A,
A_{\ell}
).
\]

### AUROC

For:

\[
A_{01}=+1
\]

versus remaining examples.

### AUPRC

Important because positive advantage events may be imbalanced.

### Pairwise ranking accuracy

For example pairs:

\[
A(x_i)>A(x_j),
\]

does:

\[
\hat A(x_i)>\hat A(x_j)?
\]

---

# 19. Policy Evaluation

At dense-execution rates:

\[
r
\in
\{0.1,0.2,0.3,0.5\},
\]

evaluate:

- final accuracy;
- dense execution rate;
- expected compute;
- net advantage capture;
- oracle recovery;
- useful densification;
- harmful densification;
- wasted densification;
- missed positive advantage.

---

# 20. Matched-Compute Evaluation

This is essential.

For every routing score:

1. rank examples;
2. densify exactly the top \(r\);
3. compare final accuracy.

This isolates routing quality from threshold calibration.

Primary comparison:

\[
Acc_{\text{advantage-probe}}(r)
-
Acc_{\text{entropy}}(r).
\]

---

# 21. Frozen-Threshold Evaluation

As a secondary deployment-transfer evaluation:

1. choose threshold on calibration data;
2. freeze threshold;
3. report actual held-out dense rate;
4. report accuracy and compute.

Do not mix this with matched-compute conclusions.

---

# 22. Baselines

## Static

### B0 — Always Sparse

\[
S_s.
\]

### B1 — Always Dense

\[
S_0.
\]

---

## Random

### B2 — Random Densification

At matched rates:

```text
10 seeds minimum
```

---

## Generic Uncertainty

### B3 — Margin

### B4 — Entropy

### B5 — Max Probability

---

## Learned Baselines

### B6 — Sparse Error Probe

\[
P(S_s\text{ wrong}).
\]

### B7 — Binary Correction Probe

\[
P(S_s\text{ wrong},S_0\text{ correct}).
\]

---

## Proposed

### B8 — Hidden-State Signed-Advantage Probe

### B9 — Hidden-State Loss-Advantage Probe

### B10 — Predicted Capacity-Instability Probe

### B11 — Combined Counterfactual Router

---

## Oracle

### B12 — Error Oracle

Prioritize sparse errors.

### B13 — Positive-Uplift Oracle

Prioritize:

\[
S_s\text{ wrong},S_0\text{ correct}.
\]

### B14 — Signed-Advantage Oracle

Rank:

\[
+1 > 0 > -1.
\]

This is the true routing ceiling.

---

# 23. Primary Comparisons

## C1 — Does signed advantage help?

Compare:

\[
\text{binary correction probe}
\]

vs.

\[
\text{signed advantage probe}.
\]

---

## C2 — Do hidden states help?

Compare:

\[
\text{entropy/margin/max-prob}
\]

vs.

\[
\text{hidden-state advantage probe}.
\]

---

## C3 — Can predicted instability recover the Iteration-6 signal?

Compare:

\[
\text{actual }S_s\leftrightarrow S_m\text{ disagreement}
\]

as an expensive diagnostic upper bound

against:

\[
\text{predicted disagreement from }h_s.
\]

---

## C4 — Which starting sparsity is best?

Compare:

\[
S_{25}\rightarrow S_0,
\]

\[
S_{40}\rightarrow S_0,
\]

\[
S_{50}\rightarrow S_0.
\]

Use oracle frontier first, then router performance.

---

# 24. Main Figures

## Figure 1 — Oracle Headroom by Sparsity

x-axis:

\[
\text{expected compute}
\]

y-axis:

\[
\text{oracle accuracy}
\]

for:

- S25→dense;
- S40→dense;
- S50→dense.

This decides the best adaptive-capacity regime.

---

## Figure 2 — Capacity-Advantage Prediction

Bar plot/table:

- entropy;
- margin;
- max prob;
- error probe;
- binary uplift probe;
- signed advantage probe;
- instability probe.

Metrics:

- AUROC;
- AUPRC;
- rank correlation.

---

## Figure 3 — Accuracy vs Expected Compute

Main system figure.

Curves for:

- random;
- entropy;
- max probability;
- error probe;
- uplift probe;
- advantage probe;
- instability probe;
- combined router;
- oracle.

---

## Figure 4 — Oracle Recovery

x-axis:

\[
\text{dense execution rate}
\]

y-axis:

\[
OR(r).
\]

This shows how much exploitable complementarity each router captures.

---

## Figure 5 — Routing Outcome Decomposition

At \(r=0.30\):

- beneficial densifications;
- harmful densifications;
- neutral densifications;
- missed beneficial cases.

---

# 25. Statistical Analysis

Use paired bootstrap because all policies are evaluated on identical examples.

Recommended:

```text
2,000 bootstrap replicates
```

Report 95% CIs for:

- accuracy difference;
- net advantage capture difference;
- oracle recovery difference.

Use McNemar tests for final correctness comparisons where useful.

---

# 26. Representation-Probe Diagnostics

If hidden-state probing works, inspect layer dependence.

For each layer \(l\):

\[
AUROC_l,
\quad
AUPRC_l,
\quad
OR_l(r).
\]

Question:

> At what depth does the sparse model know whether more capacity would help?

This could become a scientifically interesting secondary result.

---

# 27. Optional Recovery Ablation

Iteration 6 used direct post-pruning models without recovery.

If compute permits, add a light recovery procedure:

```text
small LoRA adaptation
```

or another low-memory recovery method.

Do **not** make this the primary method.

Compare:

\[
S_s^{zero-shot}
\]

vs.

\[
S_s^{recovered}.
\]

Ask:

1. Does sparse standalone accuracy improve?
2. Does oracle complementarity remain?
3. Does capacity advantage become easier to predict?

A recovery method that eliminates complementarity may actually reduce routing headroom.

---

# 28. Success Criteria

## Gate A — Oracle Headroom

Continue only if at least one sparsity level has substantial adaptive headroom.

Suggested:

\[
Acc_{oracle}(30\%)
-
Acc_{sparse}
\ge
10\text{ pp}.
\]

Iteration 6 already suggests this is plausible.

---

## Gate B — Representation Signal

Proceed to fresh MMLU-Pro evaluation if at least one representation-based method:

1. exceeds output-level uncertainty in cross-validation;
2. shows stable advantage ranking;
3. improves net advantage capture.

A practical heuristic:

\[
AUROC \ge 0.68
\]

or:

\[
OR(30\%)\ge 0.40.
\]

These are internal gates, not universal thresholds.

---

## Strong Positive Result

On fresh held-out evaluation:

\[
Acc_{\text{advantage-router}}(r)
>
Acc_{\text{entropy}}(r)
\]

at equal expected compute, with a paired CI excluding zero at one or more meaningful operating points.

Also desirable:

\[
OR(30\%)
\]

substantially exceeds generic uncertainty.

---

## Very Strong Result

A sparse→dense adaptive policy achieves:

\[
Acc_{\pi}
>
Acc_{\text{always dense}}
\]

while:

\[
E[C_\pi]
<
C_{\text{dense}}.
\]

Iteration 6's oracle indicates this is theoretically possible.

---

## Negative Result

Stop this branch if:

- hidden states do not outperform ordinary uncertainty;
- predicted instability cannot recover actual cross-capacity disagreement;
- router gains vanish on MMLU-Pro;
- oracle headroom is large but no deployable signal captures it.

Then the final scientific conclusion is:

> Sparse and dense LLMs exhibit strong example-level complementarity, but that complementarity is not predictably encoded in cheap sparse-model signals.

That is still a useful negative result.

---

# 29. Minimum Viable Iteration 7

## Models

```text
Qwen2.5-1.5B-Instruct
```

Sparse variants:

```text
S25
S50
Dense
```

Optional:

```text
S40
```

---

## Development Data

```text
ARC train
5-fold cross-validation
```

Do not treat ARC validation as fresh test data.

---

## Fresh Evaluation

```text
MMLU-Pro
```

only after the development gate passes.

---

## Targets

```text
binary correction
signed correctness advantage
loss-based advantage
capacity disagreement
```

---

## Features

```text
entropy
margin
max probability
final sparse hidden state
selected-layer sparse hidden states
```

---

## Probe Models

```text
ridge/logistic regression
multinomial logistic regression
```

Optional:

```text
small 2-layer MLP
```

---

## Main Baselines

```text
Random
Entropy
Margin
Max probability
Error probe
Binary uplift probe
Signed-advantage probe
Capacity-instability probe
Oracle
```

---

## Main Metrics

```text
AUROC
AUPRC
Net Advantage Capture
Oracle Recovery
Accuracy at matched compute
Expected compute
```

---

# 30. Recommended Execution Order

## Stage 1 — Oracle Sparsity Selection

- evaluate S25/S40/S50;
- compute signed advantage against dense;
- build oracle compute–accuracy frontiers;
- choose the best sparse starting level.

---

## Stage 2 — Build Counterfactual Targets

For chosen sparse level:

- binary correction;
- signed correctness advantage;
- loss advantage;
- actual sparse↔medium disagreement.

---

## Stage 3 — Extract Representations

Cache:

- output probabilities;
- hidden state at L/4;
- L/2;
- 3L/4;
- L.

Use only sparse forward passes.

---

## Stage 4 — Cross-Validated Probe Study

Train and evaluate:

- error probe;
- binary uplift probe;
- signed advantage probe;
- loss advantage probe;
- instability probe.

Compare against entropy/margin/max probability.

---

## Stage 5 — Compute-Aware Routing Simulation

At:

```text
10%
20%
30%
50%
```

dense rates:

- select top-ranked examples;
- compute accuracy;
- compute net advantage capture;
- compute oracle recovery.

---

## Stage 6 — Development Gate

Proceed only if a representation-level router clearly beats generic uncertainty.

---

## Stage 7 — Fresh MMLU-Pro Evaluation

Freeze:

- sparsity level;
- probe features;
- probe type;
- routing score;
- operating points.

Then evaluate on MMLU-Pro.

---

## Stage 8 — Runtime Work

Only after successful MMLU-Pro replication.

Measure:

- real latency;
- throughput;
- memory;
- end-to-end compute cost.

---

# 31. Intended Contribution

Iteration 7 should not claim:

> uncertainty controls sparsity.

That formulation is now too weak.

The stronger contribution is:

> **Adaptive sparse inference should estimate the counterfactual value of restoring model capacity, not merely model uncertainty.**

The proposed principle is:

\[
\boxed{
\text{sparse representation}
\rightarrow
\text{estimated dense-vs-sparse advantage}
\rightarrow
\text{capacity allocation}
}
\]

The central scientific insight is:

> **A model can be uncertain without benefiting from more capacity, and it can be confident while a denser realization would change or improve its prediction.**

Therefore, the relevant routing target is:

\[
\boxed{
\text{expected capacity advantage}
}
\]

rather than generic uncertainty.

---

# 32. Immediate Checklist

- [ ] Reuse physical structured-pruning implementation from Iteration 6.
- [ ] Add S40 if inexpensive.
- [ ] Evaluate static S25/S40/S50/dense hierarchy.
- [ ] Compute oracle sparse→dense frontiers.
- [ ] Choose primary sparsity level using oracle efficiency.
- [ ] Construct signed correctness advantage labels.
- [ ] Construct loss-based advantage targets.
- [ ] Construct sparse↔medium disagreement targets.
- [ ] Cache sparse output probabilities.
- [ ] Cache sparse hidden states at selected layers.
- [ ] Run 5-fold CV on ARC train.
- [ ] Evaluate entropy/margin/max probability.
- [ ] Train generic error probe.
- [ ] Train binary uplift probe.
- [ ] Train signed advantage probe.
- [ ] Train loss-advantage probe.
- [ ] Train capacity-instability probe.
- [ ] Compare AUROC/AUPRC.
- [ ] Compute Net Advantage Capture.
- [ ] Compute Oracle Recovery.
- [ ] Compare all methods at matched dense rates.
- [ ] Identify best sparse representation layer.
- [ ] Freeze method only if representation probe clearly wins.
- [ ] Do not reuse ARC validation as a fresh confirmatory test.
- [ ] Use MMLU-Pro as the next fresh held-out benchmark.
- [ ] Only after positive replication, measure real runtime.
