# Iteration 8 Plan: Capacity-Contrast Distillation for Adaptive Sparse LLMs

## Working Title

**Capacity-Contrast Distillation: Learning When a Sparse Language Model Needs More Capacity**

Alternative titles:

- **Distilling Cross-Capacity Disagreement for Adaptive Sparse LLM Inference**
- **Learning the Value of Densification from Paired Sparse Models**
- **Capacity-Contrast Routing for Adaptive Sparse Language Models**
- **From Sparse Uncertainty to Counterfactual Capacity Signals**

---

# 1. Motivation

Iterations 6 and 7 established a useful but unresolved phenomenon.

For Qwen2.5-1.5B-Instruct on ARC-Challenge:

- Dense model accuracy is substantially higher than aggressively pruned variants.
- Sparse and dense variants make **complementary errors**.
- Oracle sparse→dense routing provides large accuracy gains at reduced expected compute.
- Ordinary sparse-model uncertainty is only moderately predictive of when densification helps.
- A linear hidden-state counterfactual probe does not recover much of the available oracle gain.
- However, **actual cross-capacity prediction disagreement** is substantially more informative.

The strongest signal observed so far is:

\[
D_{50,25}(x)
=
\mathbf1[
\hat y_{50}(x)
\neq
\hat y_{25}(x)
].
\]

This signal is useful but expensive because it requires executing both sparse-capacity levels.

Iteration 8 asks:

\[
\boxed{
\text{Can this expensive cross-capacity signal be distilled into a cheap single-pass router?}
}
\]

The central methodological idea is to treat paired compressed-model behavior as **privileged training supervision**.

---

# 2. Main Contribution

Proposed method:

\[
\boxed{
\textbf{CCD: Capacity-Contrast Distillation}
}
\]

During training:

\[
x
\rightarrow
S_{50}(x),
S_{25}(x)
\]

to construct a capacity-contrast target.

At deployment:

\[
x
\rightarrow
S_{50}(x)
\rightarrow
r_\theta(h_{50}(x))
\]

where the learned router predicts whether additional capacity is likely to materially change or improve the prediction.

Therefore:

\[
\text{expensive paired-capacity supervision during training}
\]

is distilled into:

\[
\text{cheap single-capacity routing at inference}.
\]

---

# 3. Core Research Questions

## RQ1 — Can paired-capacity disagreement be distilled?

Given:

\[
D_{50,25}(x)
=
\mathbf1[
\hat y_{50}
\neq
\hat y_{25}
],
\]

train:

\[
r_\theta(\phi_{50}(x))
\rightarrow
P(D_{50,25}=1).
\]

Question:

> Can a nonlinear router predict cross-capacity instability substantially better than the linear probe from Iteration 7?

Primary target:

\[
AUROC_{CCD}
>
AUROC_{\text{linear-instability-probe}}.
\]

---

## RQ2 — Does capacity-contrast supervision improve actual densification decisions?

Disagreement is only a proxy.

The deployment objective remains:

\[
A(x)
=
\mathbf1[S_0=y^*]
-
\mathbf1[S_{50}=y^*].
\]

Question:

> Does a router trained on capacity disagreement identify positive dense advantage better than conventional uncertainty?

Compare against:

- entropy;
- margin;
- max probability;
- error probe;
- binary uplift probe.

---

## RQ3 — Does soft cross-capacity contrast outperform binary disagreement?

Compare:

### Hard target

\[
D_{hard}(x)
=
\mathbf1[
\hat y_{50}
\neq
\hat y_{25}
].
\]

### Soft probability divergence

\[
D_{JS}(x)
=
JS(
p_{50}(x),
p_{25}(x)
).
\]

### Logit/probability shift

\[
D_{KL}(x)
=
KL(
p_{50}(x)
\Vert
p_{25}(x)
).
\]

Hypothesis:

> A soft contrast target provides denser supervision and may generalize better than binary disagreement.

---

## RQ4 — Can limited labeled advantage supervision improve CCD?

Cross-capacity disagreement does not require labels.

Correctness advantage does.

Use a small labeled subset to fine-tune the router toward actual deployment value:

\[
A_{01}
=
\mathbf1[S_0=y^*]
-
\mathbf1[S_{50}=y^*].
\]

Question:

> Can abundant unlabeled disagreement supervision + limited labeled advantage supervision outperform either alone?

---

# 4. Primary Model

Use:

```text
Qwen2.5-1.5B-Instruct
```

Keep the same model used in Iterations 6–7.

Reasons:

- established pruning infrastructure;
- known S25/S40/S50 hierarchy;
- known capacity complementarity;
- large enough to retain useful redundancy after pruning;
- feasible within the 8 GB GPU constraint;
- avoids introducing model-family confounds.

---

# 5. Sparse Variants

Primary pair:

```text
Teacher-capacity model: S25
Deployment sparse model: S50
```

Also retain:

```text
Dense: S0
```

for evaluation of actual capacity advantage.

Definitions:

```text
S0  = dense
S25 = 25% structured FFN-channel sparsity
S50 = 50% structured FFN-channel sparsity
```

Reuse the nested masks from Iterations 6–7.

Do not redesign the pruning algorithm in this iteration.

---

# 6. Training-Time Capacity Pair

During CCD supervision generation:

\[
x
\rightarrow
S_{50}
\]

and:

\[
x
\rightarrow
S_{25}.
\]

Cache:

- S50 answer probabilities;
- S25 answer probabilities;
- S50 predicted label;
- S25 predicted label;
- selected S50 hidden states;
- entropy;
- margin;
- max probability;
- cross-capacity contrast targets.

S0/dense execution is not required for unlabeled CCD supervision.

---

# 7. Capacity-Contrast Targets

## T1 — Hard Prediction Disagreement

\[
T_{hard}(x)
=
\mathbf1[
\hat y_{50}
\neq
\hat y_{25}
].
\]

Advantages:

- label-free;
- directly motivated by Iteration 7;
- previously observed to be informative for dense correction.

---

## T2 — Jensen–Shannon Contrast

\[
T_{JS}(x)
=
JS(
p_{50}(x),
p_{25}(x)
).
\]

This gives a continuous target.

Train with:

\[
\mathcal L_{JS}
=
(
r_\theta(x)
-
T_{JS}(x)
)^2.
\]

---

## T3 — KL Contrast

\[
T_{KL}(x)
=
KL(
p_{50}(x)
\Vert
p_{25}(x)
).
\]

Secondary soft target.

---

## T4 — Confidence-Shift Contrast

\[
T_{\Delta m}(x)
=
|
m_{25}(x)-m_{50}(x)
|.
\]

where:

\[
m_s
=
p^{(1)}_s-p^{(2)}_s.
\]

Use only as an ablation.

---

# 8. Optional Labeled Capacity-Advantage Target

For labeled examples:

\[
A_{01}(x)
=
\mathbf1[S_0=y^*]
-
\mathbf1[S_{50}=y^*].
\]

Values:

```text
+1 = dense beneficial
 0 = neutral
-1 = dense harmful
```

Also define:

\[
Y_+
=
\mathbf1[A_{01}=+1].
\]

This is the true densification-benefit event.

---

# 9. Router Features

The deployable router may only use features from the S50 forward pass.

## F1 — Scalar Output Features

\[
[
H_{50},
margin_{50},
p_{max,50},
logit\ gap
].
\]

---

## F2 — Multi-Layer Hidden States

Use prompt-final-token residual states from selected layers:

\[
h_7,
h_{14},
h_{21},
h_{28}.
\]

Do not concatenate all four full states directly unless memory permits.

Preferred approach:

### Per-layer projection

\[
z_l
=
W_lh_l
\]

with:

```text
projection size: 64 or 128
```

then concatenate:

\[
z
=
[z_7,z_{14},z_{21},z_{28},\text{scalar features}].
\]

---

# 10. Router Architecture

Iteration 7 used linear probes.

Iteration 8 should test a small nonlinear router.

Recommended primary architecture:

```text
per-layer projection: hidden_dim → 64
concatenation
MLP: 256–300 → 128 → 32 → 1
```

Use:

- ReLU or GELU;
- dropout 0.1–0.3;
- weight decay;
- early stopping.

The router should remain tiny relative to the SLM.

---

# 11. Training Regimes

## R1 — Linear Baseline

Reproduce Iteration-7-style linear probe.

Purpose:

> verify the nonlinear improvement is real.

---

## R2 — CCD-Hard

Train MLP on:

\[
T_{hard}.
\]

Loss:

\[
\mathcal L_{CCD-hard}
=
BCE(
r_\theta(x),
T_{hard}(x)
).
\]

---

## R3 — CCD-Soft

Train MLP on:

\[
T_{JS}.
\]

Loss:

\[
\mathcal L_{CCD-soft}
=
MSE(
r_\theta(x),
T_{JS}(x)
).
\]

At inference, rank examples by predicted contrast.

---

## R4 — Multi-Task CCD

Predict both:

\[
T_{hard}
\]

and:

\[
T_{JS}.
\]

Loss:

\[
\mathcal L
=
\mathcal L_{hard}
+
\lambda_{JS}
\mathcal L_{JS}.
\]

---

## R5 — CCD + Advantage Fine-Tuning

First train on paired-capacity contrast.

Then fine-tune router using labeled:

\[
A_{01}
\]

or:

\[
Y_+.
\]

Objective:

\[
\mathcal L
=
\mathcal L_{CCD}
+
\lambda_A
\mathcal L_{advantage}.
\]

This is the strongest proposed variant.

---

# 12. Optional Representation Adapter

Only if the nonlinear router is promising but insufficient.

Insert a small trainable adapter on frozen S50 hidden states:

\[
h_l
\rightarrow
h_l+\Delta h_l.
\]

Use:

```text
LoRA or bottleneck adapter
```

Keep the Qwen backbone frozen otherwise.

Train the adapter jointly with CCD supervision.

Purpose:

> make cross-capacity information easier to expose without full model fine-tuning.

Do not start Iteration 8 with this.

---

# 13. GPU-Constrained Execution Strategy

Target GPU budget:

```text
8 GB
```

Do not keep S50, S25, and dense resident simultaneously.

## Pass A — S50 Feature Cache

Load:

```text
S50
```

For every training prompt, cache:

- selected hidden states;
- probabilities;
- logits if practical;
- predicted label;
- margin;
- entropy;
- max probability.

Then unload S50.

## Pass B — S25 Contrast Cache

Load:

```text
S25
```

Cache:

- probabilities;
- predicted label.

Construct:

\[
T_{hard},
T_{JS},
T_{KL}.
\]

Unload S25.

## Pass C — Dense Advantage Cache

Only for labeled calibration/evaluation examples.

Load:

```text
S0
```

Cache:

- probabilities;
- correctness;
- signed advantage.

Unload S0.

## Pass D — Router Training

Train router entirely from cached tensors/features.

GPU requirement should be negligible.

CPU training is also acceptable for the initial MLP.

---

# 14. Development Dataset

## 14.1 ARC-Challenge

Reuse:

```text
ARC train ~750
```

for direct continuity with Iterations 6–7.

However, 750 examples is too small to be the sole CCD training source.

Use ARC for:

- labeled advantage supervision;
- five-fold development;
- direct comparison with Iterations 6–7.

---

# 15. Additional Unlabeled / Weakly-Labeled Training Data

CCD disagreement supervision does not require correctness labels.

Therefore add several thousand multiple-choice prompts.

Target:

```text
3,000–10,000 prompts
```

depending on runtime.

Possible sources:

- MMLU training/dev prompts;
- MMLU-Pro validation/development prompts;
- ARC Easy train;
- OpenBookQA train;
- CommonsenseQA train.

Important:

> Do not use examples from the final MMLU-Pro held-out test partition for CCD training.

The goal is to increase capacity-contrast supervision diversity.

---

# 16. Training Dataset Construction

For each prompt \(x\):

```text
S50 features
S25 probabilities
S50/S25 disagreement
JS divergence
KL divergence
```

No ground-truth label required for the CCD phase.

For labeled ARC examples, additionally record:

```text
S50 correctness
S25 correctness
dense correctness
signed advantage
```

---

# 17. Validation Protocol

## Development

Use:

```text
5-fold cross-validation on ARC train
```

for labeled densification-benefit metrics.

For large unlabeled CCD data:

```text
train / validation split
```

by dataset/source.

Avoid prompt leakage across splits.

---

# 18. Main Development Gate

The first gate is:

\[
\boxed{
\text{Can CCD predict actual cross-capacity disagreement?}
}
\]

Compare:

```text
linear instability probe
CCD-Hard
CCD-Soft
CCD-MultiTask
```

Primary metric:

\[
AUROC(
\hat D,
D_{50,25}
).
\]

Internal success heuristic:

\[
AUROC \ge 0.65.
\]

Strong:

\[
AUROC \ge 0.70.
\]

---

# 19. Second Gate — Does CCD Predict Actual Capacity Benefit?

Evaluate CCD scores against:

\[
Y_+
=
\mathbf1[
S_{50}\neq y^*,
S_0=y^*
].
\]

Metrics:

- AUROC;
- AUPRC;
- Precision@10%;
- Precision@20%;
- Precision@30%.

Required comparison:

\[
CCD
>
\text{entropy}
\]

and:

\[
CCD
>
\text{max probability}.
\]

---

# 20. Main Policy Evaluation

At exact densification rates:

\[
r
\in
\{0.10,0.20,0.30,0.50\},
\]

route the top \(r\) fraction by each score.

This is a **matched-compute** comparison.

For selected examples, use dense S0 output.

Otherwise use S50 output.

---

# 21. Core Metrics

## 21.1 Final Accuracy

\[
Acc_\pi(r).
\]

## 21.2 Expected Compute

Using the existing analytical proxy:

\[
C_\pi(r)
=
C(S_{50})
+
rC(S_0).
\]

This is a sequential-compute proxy only.

Do not call it measured speedup.

## 21.3 Net Advantage Capture

\[
NAC(r)
=
E[
A_{01}(x)
D_r(x)
].
\]

## 21.4 Oracle Recovery

\[
OR(r)
=
\frac{
Acc_\pi(r)-Acc(S_{50})
}{
Acc_{\text{oracle}}(r)-Acc(S_{50})
}.
\]

## 21.5 Beneficial Densification Rate

\[
BDR
=
P[
A=+1
\mid
D=1
].
\]

## 21.6 Harmful Densification Rate

\[
HDR
=
P[
A=-1
\mid
D=1
].
\]

---

# 22. Main Baselines

## B0 — Always S50

Static sparse baseline.

## B1 — Always Dense

Static capacity baseline.

## B2 — Random Densification

At matched dense rates.

Use:

```text
>=10 random seeds
```

## B3 — Margin

\[
1-(p_1-p_2).
\]

## B4 — Entropy

\[
-\sum p_i\log p_i.
\]

## B5 — Max-Probability Uncertainty

\[
1-\max p_i.
\]

## B6 — Error Probe

Predict:

\[
P(S_{50}\text{ wrong}).
\]

## B7 — Binary Uplift Probe

Predict:

\[
P(
S_{50}\text{ wrong},
S_0\text{ correct}
).
\]

## B8 — Linear Instability Probe

Iteration-7 baseline:

\[
h_{50}
\rightarrow
D_{50,25}.
\]

## B9 — CCD-Hard

Primary proposed method.

## B10 — CCD-Soft

Soft contrast method.

## B11 — CCD-MultiTask

Hard + soft capacity contrast.

## B12 — CCD + Advantage Fine-Tuning

Strongest proposed method.

## B13 — Actual Cross-Capacity Disagreement

\[
D_{50,25}.
\]

Use only as an **expensive diagnostic upper bound**, not a deployment-equivalent baseline.

## B14 — Signed Oracle

True advantage ranking.

Ceiling only.

---

# 23. Key Ablations

## A1 — Hard vs Soft Contrast

Compare:

\[
D_{hard}
\]

against:

\[
JS(p_{50},p_{25}).
\]

## A2 — Output Features Only vs Hidden States

Compare:

```text
entropy/margin/logits only
```

against:

```text
+ hidden-state features
```

## A3 — Single Layer vs Multi-Layer Features

Compare:

```text
h14 only
```

against:

```text
h7+h14+h21+h28
```

## A4 — Linear vs Nonlinear Router

Critical ablation.

Compare:

```text
logistic/ridge
```

against:

```text
2-layer MLP
```

## A5 — Training Data Size

Train CCD with approximately:

```text
750
2,000
5,000
10,000
```

paired prompts.

Question:

> Is Iteration 7 primarily data-limited?

## A6 — Contrast-Only vs Advantage Fine-Tuning

Compare:

\[
\mathcal L_{CCD}
\]

against:

\[
\mathcal L_{CCD}
+
\lambda_A\mathcal L_{advantage}.
\]

## A7 — S50↔S25 vs S50↔Dense Distillation

If feasible, compare teacher contrast:

\[
D_{50,25}
\]

against:

\[
D_{50,0}.
\]

S50↔dense may be more directly aligned but requires dense teacher execution during training.

Use only if compute permits.

---

# 24. Main Figures

## Figure 1 — Capacity-Contrast Distillation Diagram

Training:

\[
S50(x)
\quad\Vert\quad
S25(x)
\rightarrow
\text{contrast target}
\]

\[
h_{50}(x)
\rightarrow
r_\theta
\rightarrow
\hat D.
\]

Deployment:

\[
S50(x)
\rightarrow
r_\theta
\rightarrow
\begin{cases}
S50,&\text{keep sparse}\\
S0,&\text{densify}
\end{cases}
\]

## Figure 2 — Contrast Prediction Quality

AUROC/AUPRC for:

- linear instability;
- CCD-Hard;
- CCD-Soft;
- CCD-MultiTask.

## Figure 3 — Densification-Benefit Prediction

Compare:

- entropy;
- max probability;
- uplift probe;
- CCD variants;
- actual disagreement diagnostic.

## Figure 4 — Accuracy vs Expected Compute

Main system figure.

Curves for:

- random;
- entropy;
- max probability;
- uplift probe;
- CCD;
- actual disagreement;
- oracle.

## Figure 5 — Oracle Recovery

Plot:

\[
OR(r)
\]

for:

\[
r\in\{0.1,0.2,0.3,0.5\}.
\]

---

# 25. Statistical Analysis

Use paired comparisons.

Recommended:

```text
2,000 bootstrap replicates
```

Report 95% confidence intervals for:

- CCD vs entropy accuracy;
- CCD vs max-probability accuracy;
- net advantage capture;
- oracle recovery.

Use McNemar for final correctness where useful.

---

# 26. Success Criteria

## Gate 1 — Distillation Works

Proceed if:

\[
AUROC(
CCD,
D_{50,25}
)
\ge0.65
\]

and clearly exceeds the Iteration-7 linear instability predictor.

Strong target:

\[
\ge0.70.
\]

## Gate 2 — Real Capacity Advantage Improves

CCD must outperform conventional uncertainty for:

\[
Y_+.
\]

Desired:

\[
AUROC_{CCD}
>
0.65
\]

and:

\[
AUROC_{CCD}
>
AUROC_{\max-prob}.
\]

## Gate 3 — Matched-Compute Policy Improvement

At one or more operating points:

\[
Acc_{CCD}(r)
>
Acc_{entropy}(r)
\]

and:

\[
Acc_{CCD}(r)
>
Acc_{max-prob}(r).
\]

Strong result:

\[
\Delta Acc \ge 2\text{ pp}
\]

with stable bootstrap support.

## Ideal Result

At:

\[
r=30\%,
\]

CCD recovers:

\[
\ge50\%
\]

of oracle adaptive gain.

This would be a major improvement over the approximately one-third recovery observed previously.

---

# 27. Fresh Benchmark Gate

Only after positive ARC development results.

Do not proceed to MMLU-Pro if CCD does not clearly outperform uncertainty.

---

# 28. Main Fresh Evaluation — MMLU-Pro

Use:

```text
Qwen2.5-1.5B-Instruct
S50
S25
Dense
```

Keep:

- pruning masks;
- CCD architecture;
- training objective;
- routing score

frozen.

## Evaluation Size

Preferred:

```text
2,000–full MMLU-Pro test
```

Minimum:

```text
1,000 deterministic category-stratified examples
```

---

# 29. MMLU-Pro Evaluation Modes

## Mode A — Router Transfer

Train CCD using non-test paired data.

Evaluate directly on MMLU-Pro.

## Mode B — Light Calibration

Allow only:

- threshold calibration;
- optional small advantage calibration set.

Do not retrain architecture/hyperparameters after inspecting test results.

---

# 30. Optional Cross-Model Replication

Only if the method succeeds on MMLU-Pro.

Candidate:

```text
Qwen3-1.7B
```

Purpose:

> test whether CCD is specific to Qwen2.5 or generalizes across model families/generations.

Do not prioritize this before the primary result is secure.

---

# 31. Optional Router Adapter Experiment

If frozen representations plateau but CCD clearly helps:

Train:

```text
small LoRA or bottleneck adapter
```

jointly with the router.

Only a small subset of model parameters should be trainable.

Goal:

\[
h_{50}
\]

becomes explicitly capacity-contrast-aware.

This becomes a stronger method extension:

\[
\boxed{
\text{Capacity-Contrast Representation Learning}
}
\]

but should not be required for the minimum viable paper.

---

# 32. Real Runtime Evaluation

Only after positive MMLU-Pro replication.

Measure:

- S50 latency;
- dense latency;
- router overhead;
- expected end-to-end latency;
- peak memory;
- throughput.

Do not claim speedup from the analytical compute proxy.

---

# 33. Minimum Viable Iteration 8

## Model

```text
Qwen2.5-1.5B-Instruct
```

## Sparse Pair

```text
S50
S25
```

## Dense Evaluation Model

```text
S0
```

## Development Data

```text
ARC train
+ 3k–5k additional paired MCQ prompts
```

## Router

```text
multi-layer hidden-state projection
2-layer MLP
```

## Primary Targets

```text
hard S50/S25 disagreement
soft JS divergence
```

## Baselines

```text
Entropy
Margin
Max probability
Error probe
Binary uplift probe
Linear instability probe
CCD-Hard
CCD-Soft
Actual disagreement diagnostic
Signed oracle
```

## Primary Metrics

```text
disagreement AUROC
positive-advantage AUROC
AUPRC
accuracy at matched dense rate
Net Advantage Capture
Oracle Recovery
```

---

# 34. Recommended Execution Order

## Stage 1 — Reuse Existing Sparse Models

- confirm S50/S25/dense outputs;
- reuse pruning masks;
- avoid rebuilding pruning infrastructure.

## Stage 2 — Build Paired Capacity Dataset

For 3k–10k prompts:

- run S50;
- cache hidden states and probabilities;
- unload;
- run S25;
- cache probabilities;
- construct hard/soft contrast labels.

## Stage 3 — CCD Distillation Gate

Train:

- linear instability baseline;
- CCD-Hard;
- CCD-Soft;
- CCD-MultiTask.

Evaluate held-out contrast prediction.

If no nonlinear model beats the linear baseline clearly:

> STOP.

## Stage 4 — ARC Advantage Gate

Use ARC labels and dense outputs.

Evaluate whether CCD predicts:

\[
S50\text{ wrong},S0\text{ correct}.
\]

Compare with entropy/max probability.

If CCD does not beat them:

> STOP.

## Stage 5 — Matched-Compute Routing

At:

```text
10%
20%
30%
50%
```

dense budgets:

- compute final accuracy;
- compute NAC;
- compute oracle recovery.

## Stage 6 — Optional Advantage Fine-Tuning

Only if contrast-only CCD is promising.

## Stage 7 — Freeze Method

Freeze:

- router architecture;
- hidden layers;
- training target;
- threshold/ranking method;
- operating points.

## Stage 8 — MMLU-Pro

Run fresh evaluation.

## Stage 9 — Runtime / Cross-Model

Only after successful replication.

---

# 35. Intended Positive Contribution

The main claim should be constructive:

> **We introduce Capacity-Contrast Distillation, a lightweight routing method that learns from paired sparse-capacity executions during training and predicts when additional model capacity is valuable using only a single sparse-model pass at inference.**

The methodological principle is:

\[
\boxed{
\text{counterfactual compression behavior as privileged supervision}
}
\]

rather than:

\[
\text{generic uncertainty as the routing target}.
\]

The proposed method converts:

\[
\text{expensive cross-capacity disagreement}
\]

into:

\[
\text{cheap deployable capacity routing}.
\]

---

# 36. Immediate Checklist

- [ ] Keep Qwen2.5-1.5B-Instruct as the primary model.
- [ ] Reuse S50/S25/dense pruning artifacts.
- [ ] Identify 3k–10k additional MCQ prompts.
- [ ] Ensure final MMLU-Pro test data are excluded.
- [ ] Cache S50 hidden states.
- [ ] Cache S50 probabilities.
- [ ] Cache S25 probabilities.
- [ ] Build hard disagreement targets.
- [ ] Build JS contrast targets.
- [ ] Build KL contrast targets.
- [ ] Reproduce linear instability baseline.
- [ ] Implement per-layer projections.
- [ ] Implement 2-layer CCD MLP.
- [ ] Train CCD-Hard.
- [ ] Train CCD-Soft.
- [ ] Train CCD-MultiTask.
- [ ] Evaluate disagreement AUROC.
- [ ] Stop if CCD does not clearly beat linear probing.
- [ ] Evaluate ARC dense-correction AUROC.
- [ ] Compare with entropy/max probability.
- [ ] Evaluate matched-compute routing.
- [ ] Compute Net Advantage Capture.
- [ ] Compute Oracle Recovery.
- [ ] Optionally fine-tune with signed advantage.
- [ ] Freeze the winning method.
- [ ] Run MMLU-Pro only after positive ARC development results.
- [ ] Measure real runtime only after cross-benchmark success.
