# Iteration 3 Plan: Quantization-Induced Delegation Drift

## Working Title

**When Compression Changes Who Gets Help: Quantization-Induced Delegation Drift in Small-to-Large Language Model Cascades**

Alternative titles:

- **Accuracy Preserved, Delegation Shifted: Quantization-Induced Routing Drift in SLM-to-LLM Cascades**
- **Can a Quantized Small Model Still Know When to Ask for Help?**
- **When Compression Breaks Delegation: Trustworthy Routing from Quantized Small Language Models**

---

# 1. Motivation

Iterations 1 and 2 ruled out two related hypotheses for the current setup:

1. **Teacher-aware mixed-precision allocation** did not reliably outperform student-only sensitivity, even when using a teacher-correct/student-wrong oracle.
2. **Trust-critical layers** did not form a distinct layer map from capability-critical layers. Capability and uncertainty-rank sensitivity were almost collinear across blocks.

However, the earlier whole-model experiments showed that aggressive compression can substantially alter the ordering of model uncertainty even when task accuracy does not degrade in the same way.

This suggests that the most important effect of quantization may not be:

> which layer should receive more bits?

but instead:

> **whether the quantized SLM still knows which inputs should be escalated to a stronger model.**

This is a system-level reliability question.

A small-to-large cascade is:

\[
x
\rightarrow
S(x)
\rightarrow
u(x)
\rightarrow
\begin{cases}
S(x), & u(x) \le \tau \\
L(x), & u(x) > \tau
\end{cases}
\]

where:

- \(S\): small language model,
- \(L\): stronger fallback model,
- \(u(x)\): SLM uncertainty,
- \(\tau\): escalation threshold.

Quantization changes:

\[
S_{FP}
\rightarrow
S_Q.
\]

Even if:

\[
Acc(S_Q)
\approx
Acc(S_{FP}),
\]

the uncertainty ordering may change:

\[
u_Q(x)
\not\approx
u_{FP}(x),
\]

causing:

\[
D_Q(x)
\neq
D_{FP}(x).
\]

The central chain is therefore:

\[
\boxed{
\text{quantization}
\rightarrow
\text{uncertainty drift}
\rightarrow
\text{delegation drift}
\rightarrow
\text{system-level regret}
}
\]

---

# 2. Main Research Questions

## RQ0 — Is the full-precision SLM uncertainty signal viable for routing?

Before studying compression-induced routing drift, verify that the FP SLM has a meaningful uncertainty/error relationship.

Evaluate several inexpensive uncertainty scores:

### Predictive entropy

\[
u_H(x)
=
-\sum_k p_k(x)\log p_k(x)
\]

### Maximum probability

\[
u_{\max}(x)
=
1-\max_k p_k(x)
\]

### Top-1 / top-2 margin

\[
u_M(x)
=
1-
\left(
p_{(1)}(x)-p_{(2)}(x)
\right)
\]

### Optional calibrated error probability

\[
\hat r(x)
=
P(
S(x)\text{ incorrect}
\mid
u(x)
)
\]

estimated with isotonic or logistic calibration.

### Gate

The routing study should only proceed with an uncertainty score that meaningfully discriminates errors.

Suggested experimental gate:

\[
AUROC_{\text{error}} > 0.60
\]

on the FP SLM.

This threshold is a practical project heuristic, not a literature-defined criterion.

If no uncertainty score meets this gate:

- change benchmark;
- use a larger SLM;
- or use a stronger uncertainty estimator.

Do not interpret failure of a weak router as evidence that quantization does not affect delegation.

---

## RQ1 — Does quantization change escalation decisions?

Define the FP routing policy:

\[
D_{FP}(x;\tau)
=
\mathbf{1}
[
u_{FP}(x)>\tau
].
\]

For a quantized model:

\[
D_Q(x;\tau)
=
\mathbf{1}
[
u_Q(x)>\tau
].
\]

Measure:

\[
DFR(Q,\tau)
=
P[
D_Q(x;\tau)
\neq
D_{FP}(x;\tau)
].
\]

This is the **Delegation Flip Rate**.

### Hypothesis

Quantization will induce non-trivial routing flips even when the SLM's aggregate task accuracy is approximately preserved.

---

## RQ2 — Are routing changes caused by score drift or ranking drift?

Separate two distinct failure modes.

### Score drift

The uncertainty scale changes:

\[
u_Q(x)
=
a\,u_{FP}(x)+b+\epsilon.
\]

Then a fixed threshold may produce a different delegation rate.

This can potentially be repaired with recalibration.

### Ranking drift

The ordering of examples changes:

\[
rank(u_Q)
\neq
rank(u_{FP}).
\]

This is more serious because threshold adjustment alone cannot fully recover the FP routing policy.

Measure:

\[
\rho_Q
=
Spearman(
u_Q,
u_{FP}
).
\]

Also compute pairwise ordering inversion rate:

\[
IOR_Q
=
P[
\operatorname{sign}(u_{FP}(i)-u_{FP}(j))
\neq
\operatorname{sign}(u_Q(i)-u_Q(j))
].
\]

### Hypothesis

Aggressive quantization will produce both score and ranking drift.

---

## RQ3 — Which delegation flips actually matter?

Not all routing changes have the same consequence.

Separate flips into categories.

### Harmful false acceptance

FP router delegates but quantized router accepts locally:

\[
D_{FP}=1,\qquad D_Q=0.
\]

and:

\[
S_Q(x)\neq y^*,
\qquad
L(x)=y^*.
\]

This is the most important reliability failure.

Define:

\[
HFA
=
P[
D_{FP}=1,
D_Q=0,
S_Q\neq y^*,
L=y^*
].
\]

---

### Wasteful false escalation

FP router keeps the example local but the quantized router escalates:

\[
D_{FP}=0,\qquad D_Q=1,
\]

while:

\[
S_Q(x)=y^*.
\]

This increases compute/cost without improving correctness.

Define:

\[
WFE
=
P[
D_{FP}=0,
D_Q=1,
S_Q=y^*
].
\]

---

### Beneficial escalation

Quantization causes escalation and the fallback fixes an SLM error.

---

### Benign flip

Routing changes but the final system output remains correct and system utility changes negligibly.

---

## RQ4 — Can post-quantization recalibration recover the routing frontier?

Compare three deployment conditions.

### A. Frozen FP threshold

Use:

\[
\tau_Q=\tau_{FP}.
\]

This measures direct policy transfer after quantization.

### B. Quantization-specific threshold

Choose:

\[
\tau_Q
\]

using calibration data.

This corrects score drift while preserving the same uncertainty ordering.

### C. Probability-of-error calibration

Learn:

\[
g_Q(u_Q)
\approx
P(
S_Q \text{ incorrect}
\mid u_Q
).
\]

Use isotonic regression as the primary lightweight method.

### Main question

If recalibration restores the same delegation rate but not the same cascade performance, then compression has caused **ranking-level routing damage**, not merely calibration shift.

---

# 3. Experimental Setup

## Models

Initial diagnostic:

```text
SLM: Qwen2.5-0.5B-Instruct
Fallback: Qwen2.5-1.5B-Instruct
```

Reasons:

- continuity with Iterations 1 and 2;
- existing infrastructure and cached outputs;
- same tokenizer/model family;
- low computational cost.

### Conditional scale-up

If the FP SLM fails the routing-viability gate, use a stronger pair such as:

```text
SLM: Qwen2.5-1.5B-Instruct
Fallback: Qwen2.5-7B-Instruct
```

or another same-family pair with a larger capability gap.

Do not expand model scale unless necessary.

---

# 4. Quantization Conditions

Use **uniform whole-model quantization first**.

Do not use mixed precision in the main causal experiment.

This isolates the variable:

\[
\text{precision}.
\]

Initial precision conditions:

\[
FP16/BF16,\quad
8\text{-bit},\quad
4\text{-bit},\quad
2\text{-bit}.
\]

If a reliable 3-bit implementation is easily available, add:

\[
3\text{-bit}.
\]

Use the same quantization implementation across precision levels where possible.

### Phase A

Controlled fake quantization is acceptable to establish the phenomenon.

### Phase B

Only if routing drift is positive, validate using real quantizers such as:

- GPTQ;
- AWQ;
- llama.cpp/GGUF quantization where appropriate.

Do not make deployment-speed claims from fake quantization.

---

# 5. Dataset

## 5.1 Primary Diagnostic Dataset — ARC-Challenge

Reuse the current setup.

### Calibration

```text
ARC-Challenge training pool
n ≈ 750
```

Use calibration data for:

- uncertainty-score selection;
- FP threshold determination;
- quantized threshold recalibration;
- isotonic/logistic calibration.

### Held-out evaluation

```text
ARC-Challenge validation
n = 299
```

This set remained reserved after Iteration 2 and should be used only after all routing policies are frozen.

---

## 5.2 Main Scale-Up Dataset — MMLU-Pro

If ARC shows positive routing drift, use MMLU-Pro as the main broader benchmark.

Why:

- larger evaluation set;
- harder reasoning;
- broad subject coverage;
- multiple-choice probability distributions;
- clean entropy/margin uncertainty;
- no sequence-length confound.

Suggested evaluation size:

```text
1,000–3,000 examples initially
```

or full test evaluation if compute permits.

Use development/training data only for calibration.

---

## 5.3 Optional Third Dataset

Prefer another multiple-choice benchmark before free-form generation.

Candidates:

- GPQA / GPQA Diamond;
- another difficult multiple-choice reasoning benchmark.

GSM8K is optional only after the main phenomenon is established because sequence-level uncertainty adds additional design choices.

---

# 6. Router Construction

## Step 1 — Compute FP uncertainty

For every calibration example:

\[
S_{FP}(x)
\rightarrow
p_{FP}(y|x).
\]

Compute:

- entropy;
- max-prob uncertainty;
- top-1/top-2 margin.

Select the best routing signal using calibration data only.

---

## Step 2 — Define operating points

Instead of one arbitrary threshold, define target FP delegation rates:

\[
r \in
\{0.10,0.20,0.30,0.50\}.
\]

For each target rate \(r\), choose:

\[
\tau_{FP}^{(r)}
\]

such that approximately \(r\) of calibration examples are escalated.

This creates a reliability–compute frontier.

---

## Step 3 — Freeze the FP router

Once:

\[
u^*(x)
\]

and thresholds:

\[
\tau_{FP}^{(r)}
\]

are chosen, freeze them before held-out evaluation.

---

# 7. Quantization-Induced Drift Analysis

For every precision \(Q\):

\[
S_Q(x)
\rightarrow
p_Q(y|x)
\rightarrow
u_Q(x).
\]

Measure three levels of drift.

---

## 7.1 Score Drift

### Mean shift

\[
\Delta \mu_U
=
E[u_Q]-E[u_{FP}].
\]

### Scale shift

Compare:

\[
Var(u_Q)
\]

with:

\[
Var(u_{FP}).
\]

### Distribution divergence

Optional:

- Wasserstein distance;
- KS statistic.

---

## 7.2 Ranking Drift

### Spearman correlation

\[
\rho_Q
=
Spearman(
u_Q,
u_{FP}
).
\]

### Pairwise inversion rate

\[
IOR_Q.
\]

### Top-k uncertainty overlap

For the top:

\[
10\%,20\%,30\%
\]

most uncertain examples, compute overlap between FP and quantized rankings.

---

## 7.3 Decision Drift

At every FP threshold:

\[
DFR(Q,r)
=
P[
D_Q(x;\tau_{FP}^{(r)})
\neq
D_{FP}(x;\tau_{FP}^{(r)})
].
\]

Also report:

- FP delegation rate;
- quantized delegation rate;
- absolute delegation-rate shift.

---

# 8. Two Core Routing Experiments

## Experiment A — Frozen Threshold

Use:

\[
\tau_{FP}^{(r)}
\]

unchanged for every quantized model.

Question:

> What happens if an SLM is quantized after its router has already been calibrated?

This measures deployment policy transfer.

---

## Experiment B — Matched Delegation Rate

For each quantized model, choose:

\[
\tau_Q^{(r)}
\]

so that:

\[
P[D_Q=1]
\approx r.
\]

Now every precision escalates the same fraction of examples.

This removes pure score/calibration shift.

### Critical interpretation

If:

\[
Acc_{\text{cascade},Q}(r)
<
Acc_{\text{cascade},FP}(r)
\]

at the same delegation rate, the cause must involve which examples are being selected rather than simply how many are escalated.

This is the key ranking-drift experiment.

---

# 9. End-to-End Cascade Evaluation

For every input:

\[
\hat y_{\text{cascade}}
=
\begin{cases}
S_Q(x), & D_Q(x)=0 \\
L(x), & D_Q(x)=1.
\end{cases}
\]

Measure:

### Cascade accuracy

\[
Acc_{\text{cascade}}.
\]

### Delegation rate

\[
DR
=
P[D_Q=1].
\]

### Fallback correction rate

\[
FCR
=
P[
S_Q\neq y^*,
L=y^*
\mid
D_Q=1
].
\]

### Harmful false acceptance

\[
HFA.
\]

### Wasteful false escalation

\[
WFE.
\]

---

# 10. Routing Regret

Define a simple system utility:

\[
U(\pi)
=
Acc(\pi)
-
\lambda_c C(\pi)
-
\lambda_l L(\pi).
\]

For the initial study, cost can be approximated with fallback-call rate:

\[
C(\pi)
=
P[D=1].
\]

Then define compression-induced routing regret:

\[
R_Q
=
U(\pi_{FP})
-
U(\pi_Q).
\]

Evaluate over several values of:

\[
\lambda_c.
\]

Do not select one arbitrary cost coefficient as the sole result.

Also report the raw accuracy-vs-delegation frontier so conclusions do not depend on utility weights.

---

# 11. Recalibration Methods

## R0 — No Recalibration

Use frozen FP score and thresholds.

---

## R1 — Threshold Recalibration

Choose a new threshold for each quantized precision.

This is the simplest score-shift correction.

---

## R2 — Temperature Scaling

Calibrate probabilities:

\[
p_Q
\rightarrow
p_Q^{(T)}.
\]

Recompute entropy/margin.

Useful conventional calibration baseline.

---

## R3 — Isotonic Error Calibration

Fit:

\[
g_Q(u_Q)
\rightarrow
P(error).
\]

Then route using calibrated error probability.

This is the main lightweight recovery method.

---

# 12. Baselines

## B0 — Always Small Model

\[
D(x)=0
\quad \forall x.
\]

Provides:

- minimum cost;
- baseline SLM accuracy.

---

## B1 — Always Large Model

\[
D(x)=1
\quad \forall x.
\]

Provides:

- upper-cost reference;
- fallback-model accuracy.

---

## B2 — Random Escalation

At each target delegation rate:

\[
r
\]

randomly select the same fraction of examples.

Run at least:

```text
10 random seeds
```

Purpose:

> determine whether uncertainty routing selects genuinely useful fallback calls.

---

## B3 — Oracle Error Router

Escalate SLM errors first.

At fixed rate \(r\):

- rank incorrect SLM examples ahead of correct examples.

This provides an upper bound on routing quality given the fallback model.

---

## B4 — FP Uncertainty Router

Main reference:

\[
u_{FP}(x),
\quad
\tau_{FP}.
\]

---

## B5 — Quantized Router + Frozen FP Threshold

Tests direct transfer after compression.

---

## B6 — Quantized Router + Retuned Threshold

Tests whether score recalibration is sufficient.

---

## B7 — Quantized Router + Isotonic Calibration

Tests whether mapping uncertainty to empirical error probability recovers cascade performance.

---

# 13. Primary Evaluation Metrics

## Small-Model Metrics

For each precision:

- accuracy;
- ECE;
- Brier score;
- error-detection AUROC;
- error-detection AUPRC;
- AURC.

These characterize the SLM before cascading.

---

## Drift Metrics

### Uncertainty Spearman

\[
\rho(u_Q,u_{FP})
\]

### Pairwise ordering inversion

\[
IOR_Q
\]

### Delegation Flip Rate

\[
DFR(Q,r)
\]

### Delegation-rate shift

\[
|DR_Q-DR_{FP}|.
\]

---

## System Metrics

- cascade accuracy;
- delegation rate;
- harmful false acceptance rate;
- wasteful false escalation rate;
- fallback correction rate;
- routing regret.

---

# 14. Main Figures

## Figure 1 — Accuracy vs Precision vs UQ Preservation

x-axis:

```text
bit-width
```

y-axis:

- SLM accuracy;
- FP-to-quantized uncertainty rank correlation.

Purpose:

> show whether accuracy and uncertainty behavior degrade differently.

---

## Figure 2 — Delegation Flip Rate vs Precision

For target delegation rates:

```text
10%
20%
30%
50%
```

plot:

\[
DFR(Q,r).
\]

This is the direct routing-drift figure.

---

## Figure 3 — Cascade Accuracy vs Delegation Rate

Curves for:

- FP router;
- Q8;
- Q4;
- Q2;
- random escalation;
- oracle router.

This should be one of the main paper figures.

---

## Figure 4 — Fixed Threshold vs Matched Rate

For each precision, compare:

```text
Frozen FP threshold
Retuned threshold
Isotonic calibration
```

Purpose:

> separate correctable score drift from persistent ranking drift.

---

## Figure 5 — Routing Flip Taxonomy

Stacked categories:

- harmful false accepts;
- wasteful false escalations;
- beneficial escalations;
- benign flips.

This explains why routing drift matters.

---

# 15. Threshold-Proximity Analysis

Hypothesis:

> routing flips should concentrate near the FP uncertainty threshold.

For every example define distance:

\[
d(x)
=
|u_{FP}(x)-\tau_{FP}|.
\]

Compare flip probability as a function of:

\[
d(x).
\]

If many harmful flips occur far from the threshold, this indicates severe uncertainty-rank distortion.

If almost all flips occur extremely close to the threshold, simple calibration may be sufficient.

---

# 16. Accuracy-Preserved Subset Analysis

To establish that delegation drift is not merely a consequence of changed predictions, analyze examples where:

\[
\hat y_Q(x)
=
\hat y_{FP}(x).
\]

Within this subset measure:

\[
P[
D_Q(x)\neq D_{FP}(x)
].
\]

This is an important control.

A strong result would show:

> **Quantization changes the escalation decision even when the model gives the same answer.**

This cleanly separates prediction drift from routing drift.

---

# 17. Statistical Analysis

Use paired evaluation because all model variants see the same examples.

### Accuracy

- paired bootstrap confidence intervals;
- McNemar tests where appropriate.

### DFR / HFA / WFE

- bootstrap confidence intervals over examples.

### Cascade frontier

Bootstrap:

\[
Acc_{\text{cascade}}(r)
\]

for each target rate.

### Comparison at matched rate

Report:

\[
\Delta Acc(r)
=
Acc_{FP}(r)-Acc_Q(r)
\]

with 95% bootstrap CI.

---

# 18. Success Criteria

## Strong Positive Result

Proceed to paper-scale validation if:

1. quantization meaningfully changes uncertainty ranking:

\[
\rho(u_Q,u_{FP})
\]

drops with compression;

2. Delegation Flip Rate is non-trivial;

3. harmful false accepts occur at meaningful frequency;

4. quantized routers underperform FP at the **same delegation rate**;

5. recalibration improves score drift but does not fully recover the FP routing frontier.

This would support:

> **Compression changes which examples the SLM believes require help, not merely its calibration scale.**

---

## Moderate Positive Result

If frozen thresholds fail but matched-rate performance is approximately recovered:

> quantization mainly causes **score/calibration drift**, not ranking drift.

The paper can then focus on:

> compression-aware router recalibration.

This is still useful, but weaker than the ranking-drift story.

---

## Negative Result

If:

- uncertainty ranking remains stable;
- DFR is small;
- matched-rate cascade performance is unchanged;

then quantization does not materially damage delegation for this setup.

Do not scale the direction.

Consider alternative research directions such as:

- precision instability as UQ;
- multi-layer interaction effects;
- larger or more capable SLMs;
- generative reasoning tasks.

---

# 19. Minimum Viable Iteration 3

## Models

```text
SLM: Qwen2.5-0.5B-Instruct
Fallback: Qwen2.5-1.5B-Instruct
```

## Dataset

```text
Calibration: ARC-Challenge train pool (~750)
Evaluation: ARC-Challenge validation (299)
```

## Precision

```text
FP
8-bit
4-bit
2-bit
```

## Router signals

```text
entropy
margin
max-probability
```

Select one using calibration data.

## Operating points

```text
10%
20%
30%
50% delegation
```

## Baselines

1. always SLM;
2. always LLM;
3. random escalation;
4. oracle error routing;
5. FP uncertainty router;
6. quantized + frozen threshold;
7. quantized + retuned threshold;
8. quantized + isotonic calibration.

## Main outputs

1. FP router viability;
2. uncertainty-rank preservation;
3. Delegation Flip Rate;
4. harmful/wasteful flip taxonomy;
5. cascade accuracy vs delegation rate;
6. frozen vs matched-rate comparison;
7. recalibration recovery.

---

# 20. Scale-Up After Positive ARC Result

Only after the ARC diagnostic is positive:

## Add MMLU-Pro

Use a larger held-out evaluation set.

## Add stronger model pair if needed

Example:

```text
SLM: Qwen2.5-1.5B
Fallback: Qwen2.5-7B
```

## Add real quantization

At minimum:

- GPTQ 4-bit;
- AWQ 4-bit;
- one 8-bit condition.

Then measure:

- actual model size;
- peak memory;
- latency;
- throughput;
- fallback compute.

This turns the controlled phenomenon into a deployment result.

---

# 21. Literature Context

## Compression and uncertainty

**Tong et al., 2026 — "Does Compression Preserve Uncertainty?"**  
arXiv:2606.01850

Shows that compression can decouple task accuracy from uncertainty quality.

Iteration 3 asks the downstream systems question:

> Does that uncertainty distortion change which queries are delegated?

---

## Quantized model calibration

**Zhong et al., ACL 2025 — "Quantized Can Still Be Calibrated"**

Shows that quantized models can suffer degraded calibration and that post-calibration can recover part of it.

Iteration 3 distinguishes:

\[
\text{calibration/score drift}
\]

from:

\[
\text{ranking-driven delegation drift}.
\]

---

## Uncertainty-Aware SLM-to-LLM Routing

**Chuang et al., 2025 — "Confident or Seek Stronger"**  
arXiv:2502.04428

Establishes uncertainty-based routing between small and stronger language models and shows that uncertainty–correctness alignment is central to cascade performance.

Iteration 3 tests whether this routing assumption survives compression.

---

## Cost-Optimal Calibrated Routing

**UCCI, 2026 — "Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing"**  
arXiv:2605.18796

Uses calibrated uncertainty/error probability for cost-aware routing.

Iteration 3 includes isotonic calibration as a recovery baseline but focuses on whether compression causes ordering changes that calibration cannot repair.

---

# 22. Intended Contribution

Avoid the weak claim:

> Quantization harms calibration.

Instead target:

> **We study whether deployment-time compression preserves the delegation behavior of uncertainty-driven SLM-to-LLM cascades. We separate score, ranking, and decision drift, quantify harmful and wasteful routing flips, and test whether post-quantization recalibration restores the original reliability–compute frontier.**

A stronger result would support:

> **Preserving small-model task accuracy is insufficient for preserving cascade behavior: quantization can change which inputs the model asks a stronger model to solve.**

---

# 23. Immediate Execution Checklist

- [ ] Build FP/8/4/2 whole-model student variants.
- [ ] Cache FP and quantized choice probabilities on ARC calibration.
- [ ] Cache fallback-model outputs on calibration and validation.
- [ ] Evaluate FP entropy, margin, and max-probability error AUROC.
- [ ] Select the routing signal using calibration only.
- [ ] Verify FP router viability.
- [ ] Define 10/20/30/50% FP delegation thresholds.
- [ ] Freeze router configuration.
- [ ] Run all model variants on reserved ARC validation.
- [ ] Compute uncertainty score drift.
- [ ] Compute FP-to-Q uncertainty rank correlation.
- [ ] Compute pairwise ordering inversion.
- [ ] Compute Delegation Flip Rate.
- [ ] Decompose harmful false accepts and wasteful escalations.
- [ ] Evaluate cascade accuracy at frozen FP thresholds.
- [ ] Retune thresholds to matched delegation rate.
- [ ] Fit isotonic calibration on calibration data only.
- [ ] Evaluate calibration recovery.
- [ ] Analyze routing flips conditional on identical FP/Q predictions.
- [ ] Analyze flip probability vs distance from FP threshold.
- [ ] Make ARC go/no-go decision before scaling to MMLU-Pro.
