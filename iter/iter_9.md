# Iteration 9 Plan: Capacity-Contrast Routing for Adaptive Sparse LLM Inference

## Working Title

**Capacity-Contrast Routing: Label-Free Cross-Capacity Supervision for Adaptive Sparse LLM Inference**

Alternative titles:

- **Learning When to Restore Capacity in Sparse Language Models**
- **Cross-Capacity Supervision for Adaptive Sparse LLM Routing**
- **Routing Sparse LLMs with Distilled Capacity Disagreement**
- **From Compression Disagreement to Adaptive Capacity Allocation**

---

# 1. Motivation

Iteration 8 produced a more useful result than the original nonlinear-CCD hypothesis suggested.

The original goal was:

\[
\text{nonlinear CCD router}
>
\text{linear capacity-instability router}.
\]

That specific hypothesis failed.

On the held-out paired-capacity split:

\[
AUROC_{\text{linear}}
\approx 0.678
\]

and:

\[
AUROC_{\text{CCD-Hard}}
\approx 0.677.
\]

Therefore, nonlinear routing architecture was not the source of improvement.

However, the much more important comparison is with Iteration 7.

The earlier small-data hidden-state instability predictor achieved approximately:

\[
AUROC \approx 0.545.
\]

After expanding paired-capacity supervision to approximately 2,400 training examples, even a lightweight linear router achieved:

\[
AUROC \approx 0.678.
\]

This suggests:

\[
\boxed{
\text{paired-capacity supervision is learnable and data-efficient enough to be useful}
}
\]

while:

\[
\boxed{
\text{router complexity is unnecessary}
}
\]

at least in the current setting.

Iteration 9 therefore stops architecture search and asks the system-level question that Iteration 8 did not reach:

> **Does a cheap router trained on label-free S50↔S25 disagreement actually improve sparse→dense capacity allocation over conventional uncertainty at matched compute?**

This is the critical positive-method gate.

---

# 2. Proposed Method

We refer to the simplified method as:

\[
\boxed{
\textbf{CCR: Capacity-Contrast Routing}
}
\]

## Training

For unlabeled prompt \(x\):

\[
x \rightarrow S_{50}(x)
\]

and:

\[
x \rightarrow S_{25}(x).
\]

Define the capacity-contrast target:

\[
D_{50,25}(x)
=
\mathbf1[
\hat y_{50}(x)
\neq
\hat y_{25}(x)
].
\]

Train a lightweight router:

\[
r_\theta(\phi_{50}(x))
\rightarrow
P(D_{50,25}=1).
\]

The target requires no ground-truth answer.

## Deployment

Run only:

\[
S_{50}(x).
\]

Compute:

\[
r_\theta(\phi_{50}(x)).
\]

If the CCR score is low:

\[
\rightarrow
\text{return }S_{50}.
\]

If the CCR score is high:

\[
\rightarrow
S_0
\]

and return the dense-model answer.

Thus:

\[
\boxed{
\text{paired capacity during training}
\rightarrow
\text{single sparse pass before routing at deployment}
}
\]

---

# 3. Central Hypothesis

The main Iteration-9 hypothesis is not:

> capacity disagreement can be predicted.

Iteration 8 already provides evidence for that.

The central hypothesis is:

\[
\boxed{
\text{CCR ranking improves actual densification decisions}
}
\]

relative to ordinary uncertainty.

Specifically, at matched dense-execution rate \(r\):

\[
Acc_{\text{CCR}}(r)
>
Acc_{\text{entropy}}(r)
\]

and:

\[
Acc_{\text{CCR}}(r)
>
Acc_{\text{max-prob}}(r).
\]

The paper-worthy result is the **accuracy–compute improvement**, not disagreement AUROC alone.

---

# 4. Research Questions

## RQ1 — Does capacity-contrast routing predict actual dense correction?

Define:

\[
Y_{+}(x)
=
\mathbf1[
S_{50}(x)\neq y^*,
S_0(x)=y^*
].
\]

Evaluate whether:

\[
r_\theta(x)
\]

predicts \(Y_+\) better than:

- entropy;
- margin;
- max-probability uncertainty;
- generic sparse-model error prediction.

## RQ2 — Does CCR improve the matched-compute frontier?

At exact dense rates:

\[
r
\in
\{0.10,0.20,0.30,0.50\},
\]

compare final cascade accuracy.

The most important operating points are:

\[
r=0.20
\]

and:

\[
r=0.30.
\]

## RQ3 — How much of the oracle adaptive-capacity opportunity does CCR recover?

For signed sparse→dense advantage:

\[
A(x)
=
\mathbf1[S_0=y^*]
-
\mathbf1[S_{50}=y^*],
\]

define:

\[
OR(r)
=
\frac{
Acc_{\text{CCR}}(r)-Acc(S_{50})
}{
Acc_{\text{oracle}}(r)-Acc(S_{50})
}.
\]

Compare CCR Oracle Recovery with conventional uncertainty.

## RQ4 — Is the gain really due to capacity-contrast supervision?

Compare CCR against a router trained on:

\[
P(S_{50}\text{ wrong}).
\]

If error prediction performs equally well, cross-capacity supervision may not be necessary.

If CCR is superior:

\[
\boxed{
\text{cross-capacity supervision adds decision-relevant information}
}
\]

beyond generic error detection.

---

# 5. Primary Model

Use:

```text
Qwen2.5-1.5B-Instruct
```

Do not change the base model in the primary Iteration-9 experiment.

Reuse the same structured pruning hierarchy from Iterations 6–8.

---

# 6. Capacity Levels

Use:

```text
S50 = 50% FFN-channel sparsity
S25 = 25% FFN-channel sparsity
S0  = dense model
```

Roles:

```text
S50 = deployment sparse model
S25 = privileged training-capacity model
S0  = densification target / dense execution
```

S25 is **not required during normal deployment**.

---

# 7. CCR Router

Freeze the router class to the simplest successful Iteration-8 design.

## Input Features

Primary:

\[
\phi_{50}(x)
=
[
h_{14},
H_{50},
margin_{50},
p_{\max,50},
logit\ gap
].
\]

where:

- \(h_{14}\): final-prompt-token residual state after block 14;
- \(H_{50}\): predictive entropy;
- \(margin_{50}\): top-1 vs top-2 answer probability margin;
- \(p_{\max,50}\): maximum answer probability.

This matches the strong linear comparator from Iteration 8.

## Model

Use:

```text
regularized logistic regression
```

or equivalent linear probabilistic classifier.

Do not introduce MLP tuning in Iteration 9.

---

# 8. Training Target

Primary:

\[
D_{50,25}
=
\mathbf1[
\hat y_{50}
\neq
\hat y_{25}
].
\]

Train:

\[
r_\theta
=
P(D_{50,25}=1\mid\phi_{50}).
\]

This is label-free with respect to task correctness.

---

# 9. Existing Paired Training Corpus

Reuse the Iteration-8 paired corpus:

```text
3,000 MMLU auxiliary-train prompts
```

with:

```text
2,400 training
600 held-out paired-capacity validation
```

MMLU-Pro remains excluded from this paired training set.

Do not use the 600-example held-out paired split for router fitting.

---

# 10. Pre-Registered Router Freeze

Before evaluating task correctness:

Freeze:

- feature set;
- normalization;
- logistic regularization;
- model seed;
- training examples;
- router weights.

No architecture or feature selection may be changed after inspecting the new outcome evaluation.

The purpose is to test the existing CCR router rather than search for a favorable configuration.

---

# 11. Critical Data Protocol

The next result must be based on **fresh labeled examples** relative to the router-development loop.

Do not use the consumed ARC validation split as a new confirmatory test.

---

# 12. Primary Fresh Benchmark — MMLU-Pro

Use MMLU-Pro as the main fresh labeled system benchmark.

Reasons:

- harder than standard MMLU;
- broad multi-domain reasoning;
- multiple-choice probability extraction remains straightforward;
- substantial number of examples for better statistical power;
- adaptive capacity should be more meaningful on difficult items.

---

# 13. Leakage / Duplicate Check

Before evaluation, compare the paired-capacity training corpus against the selected MMLU-Pro evaluation questions.

Normalize question strings:

```text
lowercase
strip whitespace
normalize punctuation
collapse repeated spaces
```

Remove:

- exact duplicate questions;
- near-exact duplicates if trivially identifiable.

Record:

```text
number checked
number removed
final evaluation size
```

Do not inspect correctness outcomes while determining duplicates.

---

# 14. Evaluation Size

Preferred:

```text
2,000 MMLU-Pro examples
```

If runtime is too high:

```text
1,000 deterministic category-stratified examples
```

Use a fixed seed and save IDs before model inference.

Do not select examples based on model behavior.

---

# 15. Cache Strategy for 8 GB GPU

Do not keep multiple models resident simultaneously.

## Pass A — S50

For every evaluation example cache:

- answer probabilities;
- predicted answer;
- h14;
- entropy;
- margin;
- max probability;
- logit gap;
- CCR score.

Unload S50.

## Pass B — Dense S0

Cache:

- answer probabilities;
- predicted answer;
- correctness.

Unload dense.

## Optional Pass C — S25

Not needed for deployment evaluation.

Run only if required for:

- analysis of actual capacity disagreement;
- diagnostic correlation with CCR;
- replication of the privileged signal.

This should be secondary.

---

# 16. Core Outcome Definitions

## Sparse correctness

\[
C_s(x)
=
\mathbf1[
S_{50}(x)=y^*
].
\]

## Dense correctness

\[
C_d(x)
=
\mathbf1[
S_0(x)=y^*
].
\]

## Positive Dense Advantage

\[
Y_+(x)
=
\mathbf1[
C_s=0,
C_d=1
].
\]

## Harmful Dense Advantage

\[
Y_-(x)
=
\mathbf1[
C_s=1,
C_d=0
].
\]

## Signed Capacity Advantage

\[
A(x)
=
C_d(x)-C_s(x).
\]

Therefore:

\[
A\in\{-1,0,+1\}.
\]

---

# 17. Baselines

## B0 — Always S50

Static sparse baseline.

## B1 — Always Dense

Static dense baseline.

## B2 — Random Densification

At exact rates:

\[
10\%,20\%,30\%,50\%.
\]

Use at least:

```text
20 random seeds
```

for a stable random baseline.

## B3 — Margin

Rank by:

\[
1-(p_1-p_2).
\]

## B4 — Entropy

Rank by:

\[
-\sum_i p_i\log p_i.
\]

## B5 — Max-Probability Uncertainty

Rank by:

\[
1-\max_i p_i.
\]

## B6 — Generic Error Probe

If inexpensive, train a separate lightweight probe to estimate:

\[
P(S_{50}\text{ wrong}).
\]

Use labeled calibration data only.

This tests whether generic error prediction is sufficient.

## B7 — CCR

Primary proposed method.

Train on:

\[
D_{50,25}.
\]

## B8 — Actual S50↔S25 Disagreement

Diagnostic only.

Requires S25 execution.

Do not present as deployment-cost-equivalent.

## B9 — Positive-Uplift Oracle

Rank:

\[
Y_+=1
\]

first.

## B10 — Signed-Advantage Oracle

Rank:

\[
+1 > 0 > -1.
\]

True capacity-routing ceiling.

---

# 18. Primary Metric: Accuracy at Matched Compute

At exact dense rate \(r\):

\[
D_r(x)
=
\mathbf1[
x\text{ is among top }r\text{ router scores}
].
\]

Final prediction:

\[
\hat y_\pi(x)
=
\begin{cases}
\hat y_0(x), & D_r(x)=1\\
\hat y_{50}(x), & D_r(x)=0.
\end{cases}
\]

Compute:

\[
Acc_\pi(r).
\]

Primary comparisons:

\[
\Delta Acc_{\text{CCR-Entropy}}(r)
=
Acc_{\text{CCR}}(r)
-
Acc_{\text{Entropy}}(r)
\]

and:

\[
\Delta Acc_{\text{CCR-MaxProb}}(r)
=
Acc_{\text{CCR}}(r)
-
Acc_{\text{MaxProb}}(r).
\]

---

# 19. Compute Proxy

Reuse the Iteration-6/7 sequential compute convention:

\[
C_\pi(r)
=
C(S_{50})
+
rC(S_0).
\]

For the current structured pruning hierarchy:

\[
C(S_{50})
\approx0.559.
\]

Therefore:

### 10% dense

\[
C\approx0.659.
\]

### 20% dense

\[
C\approx0.759.
\]

### 30% dense

\[
C\approx0.859.
\]

### 50% dense

\[
C\approx1.059.
\]

This is an analytical matrix-compute proxy.

Do not claim actual latency or energy savings yet.

---

# 20. Secondary Metric: Positive-Advantage Prediction

Evaluate each score against:

\[
Y_+.
\]

Metrics:

- AUROC;
- AUPRC;
- Precision@10%;
- Precision@20%;
- Precision@30%.

This tells whether the capacity-contrast score is actually aligned with densification benefit.

---

# 21. Net Advantage Capture

At exact dense rate \(r\):

\[
NAC(r)
=
\frac{1}{N}
\sum_x
A(x)D_r(x).
\]

This counts:

- beneficial densification positively;
- harmful densification negatively;
- neutral densification as zero.

Compare:

\[
NAC_{\text{CCR}}
\]

against:

\[
NAC_{\text{Entropy}}
\]

and:

\[
NAC_{\text{MaxProb}}.
\]

---

# 22. Oracle Recovery

Define:

\[
OR_\pi(r)
=
\frac{
Acc_\pi(r)-Acc(S_{50})
}{
Acc_{\text{oracle}}(r)-Acc(S_{50})
}.
\]

Primary question:

> Does CCR recover materially more adaptive headroom than uncertainty?

---

# 23. Routing Outcome Decomposition

At:

\[
r=0.30
\]

report the percentage of selected examples that are:

### Beneficial

\[
S50\text{ wrong},S0\text{ correct}.
\]

### Harmful

\[
S50\text{ correct},S0\text{ wrong}.
\]

### Neutral both correct

### Neutral both wrong

This makes the mechanism interpretable.

---

# 24. Main Success Criterion

The most important gate is system performance.

## GO

Proceed toward LIGHT method framing if CCR:

1. beats entropy or max-probability at matched compute;
2. improves at a meaningful operating point, preferably 20% or 30%;
3. shows positive Net Advantage Capture improvement;
4. shows improved Oracle Recovery.

Target practical improvement:

\[
\Delta Acc
\ge
1.5\text{--}2.0\text{ pp}
\]

at the same dense-execution rate.

Stronger if the paired bootstrap CI excludes zero.

---

# 25. Strong Positive Result

A strong result would look like:

At:

\[
r=30\%,
\]

\[
Acc_{\text{CCR}}
>
Acc_{\text{Entropy}}
\]

and:

\[
OR_{\text{CCR}}
\ge
0.45\text{--}0.50
\]

while entropy/max-prob remain closer to the previous approximately one-third oracle recovery.

This would support the claim that:

> **cross-capacity supervision improves adaptive capacity allocation beyond generic uncertainty.**

---

# 26. Very Strong Result

Ideal:

\[
Acc_{\text{CCR}}(r)
\ge
Acc(S_0)
\]

while:

\[
C_\pi(r)<1.
\]

That would mean adaptive sparse inference matches or exceeds always-dense accuracy using less estimated compute.

Do not require this for success.

---

# 27. NO-GO Criterion

Stop the branch if:

\[
Acc_{\text{CCR}}(r)
\le
Acc_{\text{Entropy/MaxProb}}(r)
\]

at all meaningful matched-compute operating points.

Also stop if CCR predicts disagreement but:

\[
AUROC(CCR,Y_+)
\]

is not better than conventional uncertainty.

Interpretation:

> cross-capacity disagreement is learnable but insufficiently aligned with actual densification value.

Do not respond by immediately adding a larger router.

---

# 28. Statistical Analysis

Use paired bootstrap:

```text
2,000–5,000 replicates
```

for:

\[
\Delta Acc_{\text{CCR-Entropy}}
\]

\[
\Delta Acc_{\text{CCR-MaxProb}}
\]

\[
\Delta NAC
\]

\[
\Delta OR.
\]

Use the same sampled examples for each paired comparison.

---

# 29. McNemar Analysis

For selected operating points, compare final system correctness between:

```text
CCR
Entropy
Max probability
```

using exact McNemar tests.

This is secondary to effect sizes and paired CIs.

---

# 30. Calibration / Threshold Evaluation

Matched-compute ranking is primary.

As a secondary deployment test:

Choose a threshold on a small calibration split to target:

```text
20% dense
30% dense
```

Freeze it.

Report actual evaluation dense rate and accuracy.

This measures threshold transfer.

Do not mix threshold-transfer conclusions with ranking conclusions.

---

# 31. Optional Training-Data Ablation

Only if the primary CCR result is positive.

Train the same linear router using:

```text
750
1,500
2,400
```

paired examples.

Plot:

\[
\text{training size}
\rightarrow
\text{disagreement AUROC}
\]

and:

\[
\text{training size}
\rightarrow
\text{system accuracy}.
\]

This can support the claim that paired-capacity supervision scales predictably.

Do not run before the main system gate.

---

# 32. Optional Feature Ablation

Only after positive system evidence.

Compare:

### Scalars only

\[
[
H,
margin,
p_{\max},
logitgap
].
\]

### h14 only

\[
h_{14}.
\]

### h14 + scalars

Primary CCR.

This identifies whether the benefit comes from representation or simple uncertainty features.

---

# 33. Optional Cross-Capacity Target Ablation

If CCR works:

Compare training labels:

\[
D_{50,25}
\]

versus:

\[
D_{50,0}.
\]

Question:

> Is an inexpensive intermediate capacity sufficient as privileged supervision?

S50↔dense training requires more dense passes and is therefore secondary.

---

# 34. Optional Second Sparse Starting Point

If time permits and CCR is positive on S50:

Test:

\[
S40\rightarrow S0.
\]

Keep the same conceptual procedure.

Do not do this before S50 success.

---

# 35. Main Figures

## Figure 1 — CCR Method

Training:

\[
S50(x)
\parallel
S25(x)
\rightarrow
D_{50,25}
\]

\[
\phi_{50}(x)
\rightarrow
\text{linear CCR}
\]

Deployment:

\[
S50
\rightarrow
CCR
\rightarrow
\begin{cases}
S50\\
Dense
\end{cases}
\]

## Figure 2 — Densification-Benefit AUROC

Compare:

- entropy;
- margin;
- max probability;
- error probe;
- CCR;
- actual disagreement.

## Figure 3 — Accuracy vs Expected Compute

This is the central result.

Plot:

- always S50;
- always dense;
- random;
- entropy;
- max probability;
- CCR;
- disagreement diagnostic;
- oracle.

## Figure 4 — Oracle Recovery

x-axis:

\[
\text{dense rate}
\]

y-axis:

\[
OR(r).
\]

## Figure 5 — Routing Outcome Composition

At 30% dense execution:

- beneficial;
- harmful;
- neutral-correct;
- neutral-wrong.

Compare CCR vs entropy.

---

# 36. Contribution if Positive

The main claim becomes:

> **We introduce Capacity-Contrast Routing (CCR), a lightweight adaptive-compression router trained from label-free disagreement between nested sparse realizations of the same language model. During inference, CCR uses only a single sparse-model pass to identify inputs that benefit from restored model capacity.**

The conceptual contribution is:

\[
\boxed{
\text{cross-capacity behavior as privileged routing supervision}
}
\]

rather than generic uncertainty.

---

# 37. Why Linear Routing Is a Feature, Not a Weakness

If the system result is positive, emphasize that the router is deliberately simple.

Benefits:

- negligible training cost;
- negligible inference overhead;
- compatible with 8 GB hardware;
- easy reproducibility;
- no end-to-end model fine-tuning;
- no additional S25 pass at deployment;
- isolates the value of capacity-contrast supervision.

The contribution is the supervision signal, not router complexity.

---

# 38. Minimum Viable Iteration 9

## Model

```text
Qwen2.5-1.5B-Instruct
```

## Capacity Levels

```text
S50
S25
Dense
```

## Router

```text
linear logistic CCR
```

## Training Data

```text
2,400 paired MMLU auxiliary-train prompts
```

## Evaluation

```text
1,000–2,000 fresh MMLU-Pro examples
```

after de-duplication.

## Baselines

```text
S50
Dense
Random
Margin
Entropy
Max probability
Error probe
CCR
Actual disagreement diagnostic
Oracle
```

## Primary Operating Points

```text
20% dense
30% dense
```

## Primary Metric

```text
accuracy at exact matched dense rate
```

## Secondary Metrics

```text
positive-advantage AUROC/AUPRC
Net Advantage Capture
Oracle Recovery
routing outcome decomposition
```

---

# 39. Recommended Execution Order

## Stage 1 — Freeze CCR

- reuse 2,400-example paired training corpus;
- train/freeze linear CCR;
- do not tune against outcome labels.

## Stage 2 — Prepare MMLU-Pro Evaluation IDs

- deterministic stratified sample;
- deduplicate against CCR training prompts;
- freeze IDs.

## Stage 3 — Cache S50

- probabilities;
- prediction;
- h14;
- uncertainty features;
- CCR score.

## Stage 4 — Cache Dense

- probabilities;
- predictions;
- correctness.

## Stage 5 — Compute Capacity Advantage

Construct:

\[
Y_+,
Y_-,
A.
\]

## Stage 6 — Evaluate Ranking

Compute:

- positive-advantage AUROC;
- AUPRC;
- Precision@k.

Compare CCR with uncertainty baselines.

## Stage 7 — Matched-Compute System Test

At:

```text
10%
20%
30%
50%
```

dense execution:

- final accuracy;
- compute;
- NAC;
- Oracle Recovery.

## Stage 8 — Primary Statistical Test

At:

```text
20%
30%
```

compare:

```text
CCR vs entropy
CCR vs max probability
```

using paired bootstrap.

## Stage 9 — Decide

### If positive:

- run feature/data-size ablations;
- optionally test S40;
- measure actual runtime;
- prepare LIGHT submission.

### If negative:

- stop the branch;
- do not introduce a more complex router automatically.

---

# 40. Immediate Checklist

- [ ] Reuse Qwen2.5-1.5B S50/S25/dense models.
- [ ] Reuse 2,400 paired-capacity training examples.
- [ ] Train the matched linear CCR router.
- [ ] Freeze router weights before labeled system evaluation.
- [ ] Sample fresh MMLU-Pro evaluation IDs.
- [ ] De-duplicate against MMLU paired training questions.
- [ ] Cache S50 outputs and h14.
- [ ] Compute CCR scores.
- [ ] Cache dense outputs.
- [ ] Construct positive/harmful/signed capacity advantage.
- [ ] Evaluate CCR positive-advantage AUROC.
- [ ] Evaluate entropy/margin/max-probability AUROC.
- [ ] Run random routing baseline.
- [ ] Evaluate exact 10/20/30/50% dense rates.
- [ ] Compute final accuracy.
- [ ] Compute expected sequential compute.
- [ ] Compute Net Advantage Capture.
- [ ] Compute Oracle Recovery.
- [ ] Compare CCR vs entropy at 20% and 30%.
- [ ] Compare CCR vs max probability at 20% and 30%.
- [ ] Bootstrap paired accuracy differences.
- [ ] Decompose beneficial/harmful/neutral densifications.
- [ ] Proceed to ablations only if CCR improves system performance.
- [ ] Proceed to runtime only after a positive system result.
