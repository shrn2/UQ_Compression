# Iteration 2 Plan: Trust-Critical Layers Under Quantization

## Working Title

**Where Does Self-Knowledge Break? Localizing Trust-Critical Layers in Quantized Small Language Models**

Alternative:

**Capability-Critical vs. Trust-Critical Layers in Quantized Small Language Models**

---

# 1. Motivation

Iteration 1 produced a clear NO-GO for teacher-aware bit allocation:

- teacher-aware policies did not reliably outperform student-only sensitivity;
- the teacher-correct / student-wrong oracle did not improve allocation at mixed budgets;
- therefore the failure is not explained only by a poor uncertainty proxy.

However, Iteration 1 also showed strong degradation in uncertainty structure under aggressive compression. FP-to-quantized uncertainty-rank preservation dropped substantially as average precision decreased.

This motivates a different question:

> **Are the layers that preserve predictions under quantization the same layers that preserve the model's uncertainty/self-knowledge?**

Recent compression work shows that predictive accuracy and uncertainty can decouple under quantization. Existing mixed-precision methods primarily identify layers important for preserving logits, perplexity, accuracy, or numerical reconstruction. This iteration instead asks whether **trustworthiness has a different sensitivity map**.

---

# 2. Core Hypothesis

Quantization sensitivity is not one-dimensional.

A layer may be:

1. important for preserving predictions but not uncertainty;
2. important for preserving uncertainty but not predictions;
3. important for both;
4. unimportant for both.

Define:

### Capability sensitivity

\[
C_l
\]

as the amount of predictive behavior disrupted when layer \(l\) is quantized.

### Trust / uncertainty sensitivity

\[
R_l
\]

as the amount of uncertainty structure disrupted when layer \(l\) is quantized.

Main hypothesis:

\[
\operatorname{rank}(C_l)
\neq
\operatorname{rank}(R_l).
\]

If true, conventional accuracy-oriented mixed-precision quantization can preserve task performance while damaging the model's ability to identify difficult or risky examples.

---

# 3. Research Questions

## RQ1 — Are capability-critical and trust-critical layers different?

For every transformer block \(l\), measure both:

\[
C_l^{(b)}
\]

and

\[
R_l^{(b)}
\]

when the block is quantized to bit-width \(b\).

Test:

\[
\rho_{CR}^{(b)}
=
\operatorname{Spearman}
(
C_l^{(b)},
R_l^{(b)}
).
\]

### Hypothesis

The rankings will not perfectly align, especially under aggressive quantization.

A low/moderate correlation would show distinct effects on:

- what the model predicts;
- how well it understands the reliability of those predictions.

---

## RQ2 — Which architectural regions are trust-critical?

Determine whether uncertainty sensitivity concentrates in particular regions.

Analyze:

- early vs. middle vs. late transformer blocks;
- attention vs. MLP modules if module-level probing is feasible;
- optionally Q/K/V/O and gate/up/down projections.

### Question

Are uncertainty-sensitive components systematically located differently from capability-sensitive components?

---

## RQ3 — Does protecting trust-critical layers preserve selective reliability?

If RQ1 finds meaningful ranking differences, construct simple diagnostic mixed-precision policies:

### Capability-aware policy

Allocate higher precision using \(C_l\).

### Trust-aware policy

Allocate higher precision using \(R_l\).

At the same average-bit budget, compare whether the trust-aware policy better preserves:

- error-detection AUROC;
- AURC;
- uncertainty ranking;
- calibration;

without catastrophically sacrificing accuracy.

The goal is not yet to claim a new state-of-the-art quantizer. RQ3 is a mechanism validation.

---

# 4. Models

Keep the existing setup initially:

```text
Student: Qwen2.5-0.5B-Instruct
Reference teacher: Qwen2.5-1.5B-Instruct
```

The teacher is **not used for allocation** in Iteration 2.

It may be retained only for optional analysis of whether trust-critical layers better preserve agreement with a stronger model.

Primary analysis should depend only on the student.

---

# 5. Dataset

## 5.1 Primary Dataset — ARC-Challenge

Reuse Iteration 1.

### Probing / calibration pool

```text
750 ARC-Challenge training examples
```

Use the full deterministic pool for layer-sensitivity estimation where runtime permits.

If subsampling is necessary:

```text
5 × 500-example stratified subsets
```

### Evaluation

```text
Complete ARC-Challenge validation split
n = 299
```

ARC remains the primary diagnostic because:

- infrastructure already exists;
- student/teacher probabilities are available;
- one-block 2/4/8-bit probes can be reused;
- uncertainty metrics are clean for multiple-choice outputs.

---

## 5.2 Secondary Dataset — MMLU-Pro

Only run after ARC establishes that capability and trust rankings meaningfully differ.

Purpose:

- test generalization beyond ARC;
- evaluate broad domains;
- test stability of trust-critical layer locations across tasks.

Suggested first pass:

```text
1,000–2,000 held-out MMLU-Pro examples
```

Use train/development examples for calibration if needed.

Do not use evaluation examples for selecting layer policies.

---

# 6. Quantization Conditions

Use controlled fake quantization first.

For every block:

\[
b \in \{2,4,8\}
\]

with the same Iteration 1 implementation:

```text
symmetric weight quantization
group size = 128
```

Keep all other model components at FP precision while probing one block.

This gives:

\[
S_{Q_{l,b}}
\]

for every layer \(l\) and precision \(b\).

---

# 7. Capability-Sensitivity Metrics

Use multiple definitions instead of assuming one metric.

## C1 — Output KL Divergence

Primary metric:

\[
C_{l,KL}^{(b)}
=
\mathbb{E}_x
\left[
D_{KL}
(
p_{FP}(x)
\Vert
p_{Q_{l,b}}(x)
)
\right].
\]

This follows the logic of KL-based forward-only sensitivity methods.

---

## C2 — Prediction Flip Rate

\[
C_{l,flip}^{(b)}
=
P
[
\hat y_{Q_{l,b}}(x)
\neq
\hat y_{FP}(x)
].
\]

This measures decision-level instability.

---

## C3 — Accuracy Drop

\[
C_{l,acc}^{(b)}
=
Acc(S_{FP})
-
Acc(S_{Q_{l,b}}).
\]

This is noisier than KL but directly interpretable.

---

# 8. Trust / Uncertainty-Sensitivity Metrics

Let:

\[
u_{FP}(x)
\]

be the FP student's uncertainty score and:

\[
u_{Q_{l,b}}(x)
\]

the uncertainty after quantizing only layer \(l\).

Use predictive entropy first because it is already available.

Also evaluate top-1/top-2 margin as a robustness check.

---

## R1 — Uncertainty Rank Disruption

Primary trust-sensitivity metric:

\[
R_{l,\rho}^{(b)}
=
1
-
\rho
\left(
u_{FP},
u_{Q_{l,b}}
\right).
\]

Interpretation:

> How much does quantizing layer \(l\) change the ordering of which examples look easy vs. difficult?

---

## R2 — Error-AUROC Drop

\[
R_{l,AUROC}^{(b)}
=
AUROC(u_{FP}, error)
-
AUROC(u_{Q_{l,b}}, error).
\]

---

## R3 — AURC Increase

\[
R_{l,AURC}^{(b)}
=
AURC_{Q_{l,b}}
-
AURC_{FP}.
\]

Higher values mean worse selective prediction.

---

## R4 — Calibration Degradation

Secondary:

\[
R_{l,ECE}^{(b)}
=
ECE_{Q_{l,b}}
-
ECE_{FP}.
\]

Calibration is secondary because the core hypothesis concerns uncertainty **ranking**, not only absolute probability calibration.

---

# 9. Primary Layer-Level Analysis

For each bit-width \(b\), construct:

\[
\{C_l^{(b)}\}_{l=1}^{L}
\]

and:

\[
\{R_l^{(b)}\}_{l=1}^{L}.
\]

Then compute:

### Ranking correlation

\[
\rho_{CR}^{(b)}
=
Spearman(C^{(b)},R^{(b)}).
\]

### Top-k overlap

For:

\[
k \in \{3,5,8\}
\]

measure:

\[
Overlap_k
=
\frac{
|Top_k(C)\cap Top_k(R)|
}{k}.
\]

### Rank displacement

\[
D_l
=
|
rank_C(l)
-
rank_R(l)
|.
\]

Layers with large \(D_l\) are especially informative.

---

# 10. Layer Taxonomy

Partition layers into four conceptual groups.

## Group A — Capability-Critical + Trust-Critical

High \(C_l\), high \(R_l\).

These should receive high precision in almost any policy.

## Group B — Capability-Critical Only

High \(C_l\), low \(R_l\).

These mainly preserve task predictions.

## Group C — Trust-Critical Only

Low/moderate \(C_l\), high \(R_l\).

These are the most scientifically interesting.

They imply:

> Quantizing the layer leaves many predictions unchanged but damages which examples the model recognizes as uncertain.

## Group D — Neither

Low \(C_l\), low \(R_l\).

Natural candidates for aggressive quantization.

---

# 11. Main Visualizations

## Figure 1 — Capability vs. Trust Sensitivity Map

For each layer:

\[
x=C_l
\]

\[
y=R_l.
\]

One point per transformer block.

Produce separate plots for:

- 4-bit probe;
- 2-bit probe.

This should be the central figure.

---

## Figure 2 — Sensitivity Across Depth

x-axis:

```text
Transformer block index
```

y-axis:

```text
normalized sensitivity
```

Plot:

\[
\tilde C_l
\]

and:

\[
\tilde R_l.
\]

Goal:

> Show whether capability and trust sensitivity peak in different model regions.

If module-level probing is added, repeat for:

- attention;
- MLP.

---

# 12. Controlled Mixed-Precision Validation

If RQ1 finds meaningful differences, construct simple policies.

Use:

\[
b_l \in \{2,4,8\}
\]

at budgets:

\[
3.0,\quad 3.5,\quad 4.0.
\]

## P1 — Capability-Aware

Rank using:

\[
C_l.
\]

## P2 — Trust-Aware

Rank using:

\[
R_l.
\]

## P3 — Joint

\[
J_l(\lambda)
=
(1-\lambda)\tilde C_l
+
\lambda\tilde R_l.
\]

Use only:

\[
\lambda\in\{0.25,0.5,0.75\}.
\]

Avoid extensive hyperparameter tuning.

---

# 13. Baselines

## B0 — FP Student

```text
Qwen2.5-0.5B-Instruct FP/BF16
```

Reference for all task and uncertainty metrics.

---

## B1 — Uniform Quantization

Uniform 4-bit where supported.

If a true uniform 3-bit backend is unavailable, state this limitation clearly.

---

## B2 — Random Mixed Precision

Randomly assign bits while exactly matching:

- average-bit budget;
- parameter count per allocation.

Run at least:

```text
5 random seeds
```

---

## B3 — KL Capability Sensitivity

\[
C_l
=
E[
KL(p_{FP}\Vert p_{Q_l})
].
\]

This is the main performance-oriented forward-sensitivity baseline.

---

## B4 — Prediction-Flip Sensitivity

\[
P(\hat y_{Q_l}\neq\hat y_{FP}).
\]

Controls for the possibility that output KL is not the best task-sensitivity metric.

---

## B5 — Accuracy-Drop Sensitivity

\[
Acc_{FP}-Acc_{Q_l}.
\]

Provides a label-aware task-oriented oracle baseline.

---

## B6 — Trust-Aware Sensitivity

Proposed diagnostic:

\[
R_l
=
1-\rho(u_{FP},u_{Q_l}).
\]

---

## B7 — Joint Capability + Trust

\[
J_l
=
(1-\lambda)\tilde C_l+\lambda\tilde R_l.
\]

Stretch baseline after the core phenomenon is validated.

---

# 14. Evaluation Metrics

Evaluation must report both capability and reliability.

## Task Performance

- accuracy;
- prediction agreement with FP;
- KL divergence from FP.

## Reliability / UQ

- error-detection AUROC;
- error-detection AUPRC;
- AURC;
- ECE;
- Brier score.

## Uncertainty Preservation

### Spearman rank preservation

\[
\rho(U_Q,U_{FP}).
\]

### Mean uncertainty distortion

\[
E[
|U_Q(x)-U_{FP}(x)|
].
\]

### Pairwise ordering flip rate

Measure:

\[
P[
sign(u_{FP}(i)-u_{FP}(j))
\neq
sign(u_Q(i)-u_Q(j))
].
\]

This directly quantifies uncertainty-ranking corruption.

---

# 15. Statistical Analysis

For layer rankings:

- bootstrap examples;
- recompute \(C_l\) and \(R_l\);
- obtain confidence intervals on:
  - layer scores;
  - ranking correlations;
  - top-k overlap.

For whole-model policy evaluation:

- paired bootstrap confidence intervals;
- paired McNemar tests for accuracy;
- bootstrap differences in AUROC/AURC.

Do not call a layer uniquely trust-critical unless its ranking is stable under resampling.

---

# 16. Key Ablations

## A1 — Entropy vs. Margin

Compute trust-critical layers using:

### Predictive entropy

\[
u_H(x)=H[p(y|x)].
\]

### Margin

\[
u_M(x)
=
1-(p_{(1)}-p_{(2)}).
\]

Question:

> Are trust-critical layers specific to one uncertainty estimator?

---

## A2 — 4-bit vs. 2-bit Sensitivity

Compare:

\[
R_l^{(4)}
\]

against:

\[
R_l^{(2)}.
\]

Question:

> Do trust-critical layers emerge gradually or only under aggressive quantization?

---

## A3 — Correct vs. Incorrect Examples

Compute uncertainty-rank disruption separately for:

- FP-correct examples;
- FP-incorrect examples.

Question:

> Do trust-critical layers mainly affect confidence on mistakes or globally reorder uncertainty?

---

## A4 — Optional Teacher Agreement

Without using the teacher for allocation, measure whether quantizing trust-critical layers disproportionately changes:

\[
D(p_T,p_S).
\]

Exploratory only.

---

# 17. Success Criteria

## Strong Positive Result

Proceed toward a trust-aware compression paper if:

1. capability and trust rankings differ substantially, e.g.:

\[
\rho(C,R)<0.7
\]

at one or more meaningful precisions;

2. top-5 overlap is low/moderate;
3. ranking differences are stable under bootstrap resampling;
4. identifiable trust-only layers exist;
5. trust-aware allocation preserves significantly better:
   - FP-UQ rank correlation;
   - AUROC/AURC;

   at similar task accuracy and bit budget.

The \(\rho<0.7\) threshold is an experimental heuristic, not a literature-defined boundary.

---

## Moderate Positive Result

If rankings differ but trust-aware allocation does not clearly improve whole-model reliability:

> Frame the work primarily as an empirical characterization of **trust-critical layers**, rather than a new compression algorithm.

Possible contribution:

> Accuracy-oriented layer sensitivity is insufficient to characterize reliability damage from quantization.

---

## Negative Result

If:

\[
\rho(C,R)\approx1
\]

across precision levels and uncertainty estimators, trust-critical layers are not meaningfully distinct.

Do not pursue UQ-aware mixed precision.

Pivot to:

> quantization-induced whole-model uncertainty / delegation drift.

---

# 18. Minimum Viable Iteration 2

## Dataset

```text
ARC-Challenge only
750-example probing pool
299-example held-out evaluation
```

## Model

```text
Qwen2.5-0.5B-Instruct
```

## Layer probes

```text
24 transformer blocks
2-bit
4-bit
```

8-bit can be retained if already cached.

## Capability metrics

- output KL;
- prediction flip rate.

## Trust metrics

- entropy-rank disruption;
- error-AUROC drop;
- AURC increase.

## Main outputs

1. capability-vs-trust sensitivity scatter;
2. layer-depth sensitivity curves;
3. Spearman ranking correlation;
4. top-k overlap;
5. simple capability-aware vs. trust-aware mixed-precision comparison.

This is enough for the next go/no-go decision.

---

# 19. Stretch Experiments

Only after ARC is positive:

1. MMLU-Pro replication;
2. Qwen2.5-1.5B as the compressed student;
3. attention-vs-MLP module-level sensitivity;
4. real GPTQ/AWQ quantization;
5. actual memory and throughput;
6. joint capability + trust-aware allocation.

---

# 20. Literature Context

## Compression and UQ

**Tong et al., 2026 — "Does Compression Preserve Uncertainty?"**  
arXiv:2606.01850

Shows that compression can decouple predictive accuracy and uncertainty, with smaller models particularly vulnerable.

Iteration 2 moves from:

> Does uncertainty degrade?

to:

> **Where inside the model does that degradation originate?**

---

## KL-Based Quantization Sensitivity

**Kong et al., 2026 — "A KL Lens on Quantization"**  
arXiv:2604.13440

Uses forward-pass KL divergence to identify components sensitive to quantization and reports that KL-based rankings align with performance degradation better than MSE/SQNR.

Iteration 2 uses this as the main capability-sensitivity comparison but asks whether the same rankings preserve uncertainty.

---

## Mixed-Precision Quantization

**MixQuant, 2026**  
arXiv:2607.23047

Models context-dependent layer sensitivity and allocates mixed precision for performance.

**MXSens, 2026**  
arXiv:2607.17733

Shows that quantization sensitivity is highly nonuniform across layers/columns and allocates precision using performance sensitivity.

These motivate sensitivity-aware compression but primarily optimize predictive performance.

Iteration 2 asks whether **trustworthiness requires a different sensitivity map**.

---

# 21. Intended Contribution

The initial claim should not be:

> We introduce a better mixed-precision quantizer.

The cleaner contribution is:

> **We distinguish capability-critical from trust-critical components in quantized small language models. By probing layers independently, we test whether the components that preserve predictions are also those that preserve uncertainty ranking, calibration, and selective reliability.**

If rankings diverge and trust-aware protection works, the stronger conclusion becomes:

> **Accuracy-oriented quantization sensitivity is insufficient for trustworthy compression; preserving self-knowledge requires explicitly protecting trust-critical layers.**

---

# 22. Immediate Execution Checklist

- [ ] Reuse/load Iteration 1 per-example one-block outputs.
- [ ] Compute per-layer output-KL capability sensitivity.
- [ ] Compute per-layer prediction-flip sensitivity.
- [ ] Compute per-layer entropy-rank disruption.
- [ ] Compute per-layer error-AUROC degradation.
- [ ] Compute per-layer AURC degradation.
- [ ] Repeat at 2-bit and 4-bit.
- [ ] Compare capability vs. trust rankings.
- [ ] Compute Spearman correlation and top-k overlap.
- [ ] Bootstrap layer rankings.
- [ ] Identify trust-only / capability-only / both / neither layers.
- [ ] Plot sensitivity across network depth.
- [ ] If rankings differ, build simple trust-aware bit allocation.
- [ ] Compare trust-aware vs. KL capability-aware allocation at matched budgets.
- [ ] Make go/no-go decision before adding MMLU-Pro or real quantization.
