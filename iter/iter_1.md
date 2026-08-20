# Iteration 1 Plan: Teacher-Uncertainty-Aware Compression

## Objective

The pilot does **not** yet support the original confident-disagreement weighting:

\[
w(x) = (1-\tilde H_T(x)) \cdot D_{\mathrm{KL}}(p_T \Vert p_S)
\]

The immediate goal of Iteration 1 is therefore **not** to scale to many benchmarks. It is to determine whether teacher uncertainty contains a reproducible compression-relevant signal beyond ordinary student-only quantization sensitivity.

The key question is:

> **Does teacher uncertainty interact with quantization sensitivity in a way that changes which student layers should be protected?**

If yes, proceed to full benchmark evaluation.  
If no, stop investing in teacher-aware bit allocation and pivot back to the quantization → uncertainty → delegation-drift direction.

---

## Pilot Findings to Carry Forward

The ARC pilot used:

- Teacher: `Qwen2.5-1.5B-Instruct`
- Student: `Qwen2.5-0.5B-Instruct`
- Calibration: 32 ARC-Challenge training examples
- Evaluation: 64 held-out ARC-Challenge validation examples
- Candidate widths: `{3, 4, 8}` bits
- Fake quantization
- 24 transformer blocks

Important observations:

1. **Teacher-aware combined score did not beat student-only sensitivity**
   - 4.0 bits:
     - Combined: 32.8%
     - Student-only: 35.9%
   - 3.5 bits:
     - Combined: 28.1%
     - Student-only: 29.7%

2. **Teacher-uncertainty-only weighting was the strongest teacher-based ablation**
   - 4.0 bits: 40.6%
   - 3.5 bits: 34.4%

3. **Combined and student-only importance rankings were nearly identical**
   - Spearman correlation: \(\rho = 0.947\)
   - Allocations differed in only 5 of 24 blocks.

4. **Combined and disagreement-only policies produced identical allocations**
   - Teacher confidence was not changing the selected policy enough.

5. **The 3.0-bit point was not informative**
   - 3 bits was the minimum allowed precision.
   - All policies collapsed to uniform 3-bit allocation.

6. **Calibration was too small**
   - Teacher beat student by only 3.1 points on calibration data.
   - Teacher beat student by 21.9 points on held-out data.
   - This suggests the 32-example calibration sample poorly represented the true teacher–student capability gap.

7. **Fake quantization is insufficient for deployment claims**
   - No real packed size, latency, memory, or throughput measurement yet.

---

# Research Questions for Iteration 1

## RQ1 — Does teacher uncertainty predict quantization vulnerability?

For each calibration example \(x\) and layer \(l\), measure:

\[
H_T(x)
\]

and

\[
\Delta_l^{(b)}(x)
=
D_{\mathrm{KL}}
\left(
p_S^{FP}(x)
\Vert
p_S^{Q_{l,b}}(x)
\right)
\]

for each candidate bit-width \(b\).

Test whether:

\[
H_T(x)
\]

is associated with larger or smaller quantization perturbation.

### Main analyses

For each layer:

\[
\operatorname{Corr}
\left(
H_T(x),
\Delta_l^{(b)}(x)
\right)
\]

and globally:

\[
\mathbb{E}[\Delta_l^{(b)} \mid H_T \text{ high}]
-
\mathbb{E}[\Delta_l^{(b)} \mid H_T \text{ low}]
\]

The purpose is to determine whether teacher uncertainty provides information not already contained in student-only sensitivity.

---

## RQ2 — Which examples are actually worth protecting?

Compare three competing hypotheses.

### H1 — Protect teacher-confident examples

\[
w_{\text{conf}}(x)
=
1-\tilde H_T(x)
\]

Interpretation:

> Preserve behavior where the teacher is confident.

This corresponds most closely to the original intuition.

---

### H2 — Protect teacher-uncertain examples

\[
w_{\text{unc}}(x)
=
\tilde H_T(x)
\]

Interpretation:

> Teacher uncertainty may mark broadly fragile or difficult capabilities that are disproportionately damaged by compression.

This direction was unexpectedly strongest in the pilot and must now be tested explicitly.

---

### H3 — Protect teacher-correct / student-wrong examples

Use labels only for a diagnostic oracle:

\[
R(x)
=
\mathbf{1}
[
T(x)=y^*
\land
S(x)\neq y^*
]
\]

Then define:

\[
I_l^{\text{oracle}}
=
\mathbb{E}
[
\Delta_l(x)
\mid
R(x)=1
]
\]

This is not necessarily the final deployable method.

Its purpose is to answer:

> If we had a perfect indicator of “teacher knows something the student does not,” would teacher-aware compression help at all?

This is the most important conceptual sanity check.

---

## RQ3 — Does teacher information produce a genuinely different layer ranking?

The previous combined method largely reproduced student-only sensitivity.

For each method, compare layer rankings using:

- Spearman correlation
- top-\(k\) layer overlap
- final bit-allocation overlap
- number of changed layer assignments

The teacher-based policy is only scientifically interesting if it changes the compression allocation in a stable and interpretable way.

---

# Method Revision

## 1. Separate generic sensitivity from teacher-specific interaction

Instead of:

\[
I_l
=
\mathbb{E}
[
w(x)\Delta_l(x)
]
\]

use two components.

### Generic student sensitivity

\[
B_l
=
\mathbb{E}_x[\Delta_l(x)]
\]

### Teacher-specific interaction

One simple formulation:

\[
M_l
=
\mathbb{E}
[
\Delta_l(x)
\mid
H_T(x) \text{ high}
]
-
\mathbb{E}
[
\Delta_l(x)
\mid
H_T(x) \text{ low}
]
\]

Alternative:

\[
M_l
=
\operatorname{Cov}
\left(
z(H_T(x)),
z(\Delta_l(x))
\right)
\]

Then:

\[
I_l
=
B_l
+
\lambda M_l
\]

This ensures that the teacher term represents **additional information**, rather than simply rescaling the same student sensitivity ranking.

---

## 2. Empirically probe every candidate precision

The pilot measured perturbation at 4 bits and extrapolated other precisions.

Iteration 1 should instead compute:

\[
\Delta_{l,2}(x),
\quad
\Delta_{l,3}(x),
\quad
\Delta_{l,4}(x),
\quad
\Delta_{l,8}(x)
\]

directly.

Recommended candidate set:

\[
b_l \in \{2,3,4,8\}
\]

or, if runtime is limited:

\[
b_l \in \{2,4,8\}
\]

This avoids assuming a smooth theoretical error curve across bit-widths.

---

# Dataset

## Calibration

Increase from 32 examples to:

\[
500-1000
\]

examples.

Initial Iteration 1 recommendation:

- ARC-Challenge train/development data
- stratified where possible by:
  - teacher correctness
  - student correctness
  - teacher entropy quantile

Use at least:

\[
5
\]

independent calibration seeds/subsets.

### Important requirement

Before building any policy, verify that the calibration sample reproduces a meaningful teacher–student capability gap.

For each calibration seed report:

- teacher accuracy
- student accuracy
- teacher–student gap
- number of teacher-correct / student-wrong examples

If the calibration subset contains too few teacher-correct/student-wrong cases, the teacher-aware signal is not being properly tested.

---

## Evaluation

For Iteration 1, stay focused on ARC.

Use:

- full ARC-Challenge validation set
- approximately 299 examples

Do **not** expand immediately to MMLU-Pro/GSM8K until the signal passes the ARC diagnostic.

---

# Models

Keep the current model pair initially:

- Teacher: `Qwen2.5-1.5B-Instruct`
- Student: `Qwen2.5-0.5B-Instruct`

Reason:

> Iteration 1 should isolate the effect of better calibration and a better teacher-aware signal, rather than simultaneously changing models.

A larger teacher can be tested later if the mechanism works.

---

# Policies to Compare

At minimum:

1. Full-precision student
2. Uniform quantization
3. Random mixed precision
4. Student-only sensitivity
5. Teacher-confidence interaction
6. Teacher-uncertainty interaction
7. Teacher–student disagreement
8. Oracle teacher-correct/student-wrong interaction

For random mixed precision:

- use at least 5 seeds
- enforce exactly matched bit budgets

---

# Bit Budgets

Recommended:

\[
4.0,\quad
3.5,\quad
3.0
\]

bits/weight.

However, unlike the pilot, 3.0 bits must remain a **mixed-precision operating point**.

Therefore include a precision below 3 bits, e.g.:

\[
\{2,3,4,8\}
\]

Otherwise remove the 3.0-bit budget entirely.

---

# Evaluation Metrics

## Primary compression outcome

### Accuracy

\[
Acc(S_Q)
\]

Report paired differences relative to:

- FP student
- student-only sensitivity
- uniform quantization

Use bootstrap confidence intervals.

---

## Uncertainty preservation

### ECE

\[
ECE
\]

### Brier score

\[
BS
\]

### Error-detection AUROC

\[
AUROC
\left(
U(x),
\mathbf{1}[\hat y \neq y]
\right)
\]

### AURC

Risk–coverage performance.

### FP–quantized uncertainty rank preservation

\[
\rho
\left(
U_{S_Q},
U_{S_{FP}}
\right)
\]

---

## Layer-ranking diagnostics

For every teacher-aware method versus student-only:

\[
\rho(I_l^{teacher}, I_l^{student})
\]

Also report:

- top-5 layer overlap
- top-10 layer overlap
- number of different bit assignments
- per-layer precision allocation

---

# Critical Diagnostic Plots

## Plot 1 — Teacher uncertainty vs quantization perturbation

For each example:

\[
x = H_T(x)
\]

\[
y = \Delta_l(x)
\]

Aggregate across layers and also inspect selected layers.

Goal:

> Determine whether high- or low-teacher-uncertainty examples are systematically more compression-sensitive.

---

## Plot 2 — Layer interaction score

For each layer:

\[
M_l
=
E[\Delta_l \mid H_T \text{ high}]
-
E[\Delta_l \mid H_T \text{ low}]
\]

Plot \(M_l\) across all transformer blocks.

Goal:

> Identify whether teacher uncertainty reveals a structured subset of fragile layers.

---

## Plot 3 — Student-only vs teacher-aware importance

Scatter:

\[
x = B_l
\]

\[
y = I_l^{teacher}
\]

Report Spearman correlation.

If correlation remains near 1.0, the teacher signal is adding little.

---

## Plot 4 — Accuracy vs average bits

Compare:

- uniform
- student-only
- teacher-confidence
- teacher-uncertainty
- oracle

Goal:

> Determine whether teacher-derived allocation improves the performance–compression frontier.

---

# Statistical Protocol

For each calibration seed:

1. construct the policy using calibration data only;
2. evaluate on the same fixed held-out ARC validation set;
3. save per-example predictions.

Report:

- mean ± standard deviation across calibration seeds
- paired bootstrap confidence intervals
- McNemar test for paired accuracy comparisons where appropriate

Do not interpret single-seed differences of a few percentage points as meaningful.

---

# Go / No-Go Criteria

## GO: proceed to full paper-scale experiments if

Teacher-aware compression satisfies most of the following:

1. Teacher-aware layer ranking is meaningfully different from student-only:
   \[
   \rho < 0.9
   \]
   or substantial top-\(k\) allocation differences emerge.

2. The ranking difference is stable across calibration seeds.

3. Teacher-aware policy beats student-only sensitivity at:
   - at least two non-trivial bit budgets, or
   - one budget with a clear reliability advantage at matched accuracy.

4. Improvement is not explained by random mixed-precision variance.

5. Oracle teacher-correct/student-wrong weighting shows a clear advantage.

If these hold, proceed to:

- MMLU-Pro
- GSM8K
- GPQA/GPQA-Diamond
- GPTQ
- AWQ
- one current mixed-precision baseline
- real packed quantization/deployment evaluation

---

## CONDITIONAL GO

If:

- oracle teacher-aware weighting works,
- but entropy-based teacher weighting does not,

then the research hypothesis may still be valid.

Next step:

> develop a better proxy for “teacher knows something useful that the student lacks.”

Possible signals:

- teacher margin
- calibrated teacher confidence
- ensemble/disagreement uncertainty
- semantic entropy
- teacher correctness predictor
- teacher–student logit-margin gap

---

## NO-GO

Stop pursuing teacher-aware bit allocation if:

1. oracle teacher-aware weighting does not beat student-only sensitivity;
2. teacher-aware ranking remains almost identical to student-only ranking;
3. gains disappear across calibration seeds;
4. improvements are no larger than random mixed-precision variance.

If this occurs, pivot to the alternative direction:

> **Quantization-induced uncertainty distortion and delegation drift in SLM → LLM cascades.**

---

# Literature-Grounded Interpretation

The next iteration should explicitly treat the direction of teacher uncertainty as an open empirical question.

Relevant neighboring results suggest competing interpretations:

- **GRACE**: high teacher entropy is associated with unreliable teacher supervision and should be down-weighted.
- **EdgeRazor**: teacher entropy is useful for changing the form of knowledge transfer rather than simply discarding uncertain examples.
- **Compression/UQ studies**: quantization can preserve task accuracy while degrading uncertainty and calibration.
- **Recent mixed-precision work**: allocation should be evaluated under true per-bit sensitivity and matched global storage budgets rather than relying on a single-precision approximation.

The proposed contribution should therefore shift from:

> “Teacher confidence tells us what to protect.”

to:

> **“Teacher uncertainty may reveal which capabilities are disproportionately fragile under compression; we test whether this interaction can guide mixed-precision allocation beyond student-only sensitivity.”**

---

# Immediate Execution Checklist

- [ ] Increase ARC calibration set to 500–1000 examples.
- [ ] Create at least 5 calibration seeds.
- [ ] Verify teacher–student capability gap per seed.
- [ ] Cache teacher probabilities, entropy, correctness, and student probabilities.
- [ ] Measure per-layer perturbation separately at 2/3/4/8 bits.
- [ ] Compute student-only baseline sensitivity.
- [ ] Compute teacher-confidence interaction.
- [ ] Compute teacher-uncertainty interaction.
- [ ] Compute disagreement-only signal.
- [ ] Compute oracle teacher-correct/student-wrong signal.
- [ ] Compare layer-ranking correlations and allocation overlap.
- [ ] Evaluate all policies on full ARC validation.
- [ ] Report bootstrap CIs and calibration-seed variance.
- [ ] Make go/no-go decision before adding more datasets.
