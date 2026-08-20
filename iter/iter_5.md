# Iteration 5 Plan: Capability-Constrained Cascade-Aware Quantization

## Working Title

**Capability-Constrained Cascade-Aware Quantization: Is There Useful Routing Signal Beyond Model Fidelity?**

Alternative titles:

- **Preserve Capability First: Cascade-Aware Search Within Safe Quantization Regions**
- **Does Cascade Utility Add Signal Beyond Accuracy-Oriented Quantization?**
- **Compression for Cascades Under a Capability Constraint**

---

# 1. Motivation

Iterations 1–4 progressively narrowed the hypothesis space.

### Iteration 1
Teacher-aware mixed-precision allocation did not reliably outperform student-only sensitivity, including under an oracle teacher-correct/student-wrong signal.

### Iteration 2
Probability-based trust-critical layers were not distinct from capability-critical layers.

### Iteration 3
Whole-model quantization substantially changed uncertainty rankings and delegation decisions, but routing drift alone was not consistently harmful.

### Iteration 4
Direct local cascade-utility sensitivity (CUAQ) failed.

At 4.0 average bits, CUAQ sacrificed too much standalone capability and the cascade could not recover the loss. Across compressed configurations, standalone SLM quality remained strongly associated with cascade utility.

The key design lesson is:

> **Cascade utility should refine a capability-preserving compression policy, not replace capability preservation as the primary objective.**

Iteration 5 therefore asks a narrower and more decisive question:

\[
\boxed{
\text{Once model capability is held approximately fixed,}
\quad
\text{does cascade utility provide useful additional signal?}
}
\]

Instead of searching the full mixed-precision space, begin from a strong KL-sensitive allocation and search only within a **capability-safe neighborhood**.

---

# 2. Main Research Questions

## RQ1 — Does cascade utility vary meaningfully among equally capable compressed models?

Given a set of quantization configurations:

\[
\mathcal{Q}_{safe}
\]

that satisfy approximately the same capability constraint, measure:

\[
Var\left(U_{\text{cascade}} \mid Q\in\mathcal{Q}_{safe}\right).
\]

### Hypothesis

If this variance is non-trivial, cascade-aware optimization may still be useful after capability is protected.

If the variance is tiny, standalone capability is effectively sufficient for static compression selection in this setup.

---

## RQ2 — Can capability-constrained search improve cascade utility over KL-sensitive allocation?

Let:

\[
Q_{KL}
\]

be the baseline KL-sensitive mixed-precision allocation.

Search for:

\[
Q^*\in\mathcal{Q}_{safe}
\]

such that:

\[
U_{\text{cascade}}(Q^*) > U_{\text{cascade}}(Q_{KL})
\]

while preserving approximately the same standalone capability.

This is the proposed method:

\[
\boxed{\text{C3Q: Capability-Constrained Cascade Quantization}}
\]

---

## RQ3 — If gains exist, where do they come from?

At matched capability, analyze whether utility gains are caused by:

- better uncertainty ranking;
- more useful fallback calls;
- fewer missed beneficial delegations;
- fewer harmful escalations;
- lower fallback rate at similar accuracy.

This separates routing quality from standalone model quality.

---

# 3. Core Method: C3Q

## 3.1 Base Quantization Policy

Start from the strongest conventional compression policy from Iteration 4:

\[
Q_{KL}.
\]

KL sensitivity:

\[
S^{KL}_{l,b}
=
\mathbb{E}_x\left[D_{KL}\left(p_{FP}(x)\Vert p_{Q_{l,b}}(x)\right)\right].
\]

Use the same multiple-choice knapsack allocation under the chosen bit budget.

Do not build a new layer sensitivity metric.

---

## 3.2 Capability Constraint

Define a candidate configuration \(Q\) as admissible only if it satisfies a capability guardrail.

### Guardrail A — KL constraint

\[
D_{KL}(Q)
\le
(1+\epsilon)D_{KL}(Q_{KL}).
\]

Suggested:

\[
\epsilon\in\{0.05,0.10,0.20\}.
\]

Primary condition:

\[
\epsilon=0.10.
\]

### Guardrail B — Calibration accuracy constraint

\[
Acc_{cal}(Q)
\ge
Acc_{cal}(Q_{KL})-\delta.
\]

Suggested:

\[
\delta\in\{0.01,0.02\}.
\]

Primary:

\[
\delta=0.01.
\]

### Joint Safe Set

Primary safe set:

\[
\mathcal{Q}_{safe}
=
\left\{
Q:
D_{KL}(Q)\le1.1D_{KL}(Q_{KL}),
\;Acc_{cal}(Q)\ge Acc_{cal}(Q_{KL})-0.01
\right\}.
\]

This prevents the Iteration-4 failure mode where a routing-oriented allocation destroyed basic SLM capability.

---

# 4. Candidate Generation

Do not search the entire:

\[
\{2,4,8\}^{L}
\]

space.

Search locally around \(Q_{KL}\).

## 4.1 Single-Swap Neighborhood

For layer pairs \((i,j)\), make budget-preserving swaps such as:

```text
Layer i: 2 → 4
Layer j: 8 → 4
```

or:

```text
Layer i: 4 → 8
Layer j: 4 → 2
```

Retain only exact-budget configurations.

## 4.2 Multi-Swap Neighborhood

Generate configurations with:

```text
2–3 simultaneous precision swaps
```

around the KL solution.

Use:

```text
100–300 total candidate configurations
```

per bit budget.

## 4.3 Random Safe Neighborhood

1. begin from \(Q_{KL}\);
2. randomly choose 2–4 layers;
3. perform budget-preserving swaps;
4. retain only candidates satisfying capability constraints.

This provides a less deterministic candidate pool.

---

# 5. Candidate Evaluation

For each candidate \(Q\), evaluate on **calibration data only**.

## Standalone capability

- accuracy;
- FP prediction agreement;
- output KL.

## Routing quality

- error AUROC;
- AURC;
- uncertainty rank correlation;
- matched-rate delegation behavior.

## Cascade behavior

At delegation targets:

\[
R=\{0.10,0.20,0.30,0.50\},
\]

compute:

- cascade accuracy;
- actual delegation rate;
- fallback correction rate;
- missed beneficial delegation;
- harmful escalation.

## Utility

\[
U(Q;r,\lambda)
=
Acc_{\text{cascade}}(Q;r)-\lambda DR(Q;r).
\]

Evaluate:

\[
\lambda\in\{0,0.01,0.025,0.05,0.1\}.
\]

Aggregate:

\[
\bar U(Q)
=
\frac{1}{|R||\Lambda|}
\sum_{r,\lambda}U(Q;r,\lambda).
\]

---

# 6. Search Objective

Among candidates satisfying the capability constraint, choose:

\[
Q^*
=
\arg\max_{Q\in\mathcal{Q}_{safe}}\bar U(Q).
\]

This differs fundamentally from Iteration 4.

### Iteration 4

\[
\text{cascade utility}
\rightarrow
\text{layer sensitivity}
\rightarrow
\text{allocation}.
\]

### Iteration 5

\[
\text{KL allocation}
\rightarrow
\text{capability-safe candidate set}
\rightarrow
\text{direct whole-model cascade search}.
\]

No additive local cascade-utility assumption is required.

---

# 7. Dataset

## 7.1 Primary Dataset — ARC-Challenge

### Calibration

```text
ARC-Challenge train
n ≈ 750
```

Use for:

- KL baseline construction;
- candidate capability filtering;
- candidate cascade-utility estimation;
- router threshold calibration;
- candidate selection.

### Evaluation

```text
ARC-Challenge validation
n = 299
```

Use only once candidate selection and hyperparameters are frozen.

This is the first go/no-go benchmark.

## 7.2 Secondary Dataset — MMLU-Pro

Run only if ARC is positive.

Purpose:

- larger evaluation scale;
- more reliable routing statistics;
- broader domain coverage;
- test candidate-selection generalization.

### Evaluation

Prefer:

```text
2,000–full test examples
```

if compute permits.

At minimum, reuse the deterministic stratified 1,000-example evaluation from Iteration 3.

---

# 8. Models

Primary cascade:

```text
SLM: Qwen2.5-1.5B-Instruct
Fallback: Qwen2.5-7B-Instruct
```

Keep this pair fixed throughout the ARC diagnostic.

---

# 9. Quantization Search Space

Use the Iteration-4 fake-quantization setup first.

Candidate precisions:

\[
b_l\in\{2,4,8\}.
\]

Primary budgets:

\[
\bar b\in\{3.5,4.0\}.
\]

Do not add more bitwidths until the constrained-search hypothesis is positive.

---

# 10. Baselines

## B0 — FP/BF16

```text
Qwen2.5-1.5B-Instruct
```

## B1 — Uniform Q4

Uniform 4-bit quantization.

## B2 — Random Mixed Precision

Use:

```text
10–20 exact-budget random allocations
```

per budget.

Report mean, standard deviation, and best random candidate.

## B3 — KL-Sensitive Mixed Precision

Primary baseline:

\[
Q_{KL}.
\]

## B4 — Accuracy-Sensitive Mixed Precision

Use calibration accuracy drop as a label-aware capability baseline.

## B5 — Local CUAQ

Retain Iteration-4 CUAQ as a negative baseline.

Do not spend new tuning effort on it.

## B6 — Best Safe Random Candidate

Among random candidates satisfying the same capability constraints, select the best calibration-utility candidate.

This is important because C3Q is also a search procedure.

## B7 — C3Q

\[
Q^*=\arg\max U_{\text{cascade}}
\]

subject to capability constraints.

---

# 11. Evaluation

## 11.1 Capability Metrics

### Accuracy

\[
Acc(S_Q)
\]

### Prediction agreement with FP

\[
P[\hat y_Q=\hat y_{FP}].
\]

### Output KL

\[
E[KL(p_{FP}\Vert p_Q)].
\]

## 11.2 Routing Metrics

- error AUROC;
- AUPRC;
- AURC;
- FP-to-Q uncertainty Spearman;
- delegation flip rate at matched delegation rate.

These are diagnostic.

## 11.3 Cascade Metrics

### Cascade accuracy

\[
Acc_{\text{cascade}}.
\]

### Delegation rate

\[
DR.
\]

### Fallback correction rate

\[
P[S_Q\neq y^*,L=y^*\mid D=1].
\]

### Missed beneficial delegation

\[
P[S_Q\neq y^*,L=y^*,D=0].
\]

### Harmful escalation

\[
P[S_Q=y^*,L\neq y^*,D=1].
\]

## 11.4 System Utility

\[
U=Acc_{\text{cascade}}-\lambda DR.
\]

Primary comparison:

\[
\Delta U
=
U(C3Q)-U(Q_{KL}).
\]

---

# 12. Critical Diagnostic: Does Cascade Utility Have Headroom?

For all:

\[
Q\in\mathcal{Q}_{safe},
\]

compute:

\[
Var(U_{\text{cascade}})
\]

and:

\[
Range(U_{\text{cascade}})=\max U-\min U.
\]

Also compute within the safe set:

\[
\rho(Acc(S_Q),U_{\text{cascade}}(Q)).
\]

This is the most important scientific diagnostic.

### Interpretation

If:

\[
Var(U_{\text{cascade}}\mid\mathcal{Q}_{safe})\approx0,
\]

then cascade-aware static compression has little room to improve after capability is controlled.

If variance remains substantial:

> cascade utility contains deployment-specific information beyond standalone capability.

---

# 13. Main Figures

## Figure 1 — Capability-Safe Candidate Cloud

x-axis:

\[
D_{KL}(Q)
\]

or calibration accuracy.

y-axis:

\[
U_{\text{cascade}}(Q).
\]

Highlight:

- KL baseline;
- random candidates;
- rejected candidates;
- safe candidates;
- C3Q winner.

This should be the central figure.

## Figure 2 — Standalone Accuracy vs Cascade Utility Within Safe Set

Only candidates satisfying capability constraints.

Purpose:

> determine whether utility still varies once capability is approximately controlled.

## Figure 3 — Cascade Accuracy vs Delegation Rate

Curves for:

- FP;
- uniform;
- KL;
- safe-random best;
- C3Q.

## Figure 4 — Bit Allocation Difference

Compare:

\[
Q_{KL}
\]

and:

\[
Q_{C3Q}.
\]

## Figure 5 — Delegation Outcome Difference

At:

\[
r=0.30
\]

compare:

- fallback corrections;
- harmful escalations;
- missed beneficial delegation.

---

# 14. Statistical Analysis

All held-out comparisons use identical evaluation examples.

### C3Q vs KL

Use:

- paired bootstrap CI for cascade-utility difference;
- paired bootstrap CI for cascade accuracy;
- McNemar test for final cascade correctness.

### Candidate-selection robustness

Repeat candidate generation/search with:

```text
3 random seeds
```

if stochastic.

Report:

- selected configurations;
- utility variance;
- overlap in modified layers.

---

# 15. Key Ablations

## A1 — KL-Only Constraint

\[
KL(Q)\le1.1KL(Q_{KL}).
\]

## A2 — Accuracy-Only Constraint

\[
Acc_{cal}(Q)\ge Acc_{cal}(Q_{KL})-0.01.
\]

## A3 — Joint Constraint

Primary:

\[
KL(Q)\le1.1KL(Q_{KL})
\]

and:

\[
Acc_{cal}(Q)\ge Acc_{cal}(Q_{KL})-0.01.
\]

Compare which constraint best predicts held-out capability preservation.

## A4 — Search Radius

Compare candidates differing from KL in:

```text
1 swap
2 swaps
3 swaps
```

## A5 — Utility Target

Compare:

### Mean frontier utility

\[
\bar U(Q)
\]

versus:

### One operating point

\[
U(Q;r=0.30,\lambda=0.025).
\]

The mean-frontier objective should be primary to avoid overfitting.

## A6 — Error vs Uplift Routing Analysis

Compare whether the selected configuration better prioritizes:

\[
S_Q\text{ errors}
\]

or specifically:

\[
S_Q\text{ wrong and }L\text{ correct}.
\]

The latter is the actual useful delegation target.

---

# 16. Success Criteria

## Strong Positive Result

Proceed to MMLU-Pro if:

1. C3Q remains inside the predefined capability-safe region;
2. C3Q improves held-out cascade utility over KL;
3. improvement exceeds the best safe-random candidate or is stable across search seeds;
4. gains come from better delegation outcomes rather than capability differences.

Ideal:

\[
|Acc(C3Q)-Acc(KL)|<1\text{ pp}
\]

but:

\[
U(C3Q)>U(KL)
\]

with a bootstrap CI excluding zero at one or more meaningful operating regions.

## Moderate Positive Result

If C3Q does not significantly beat KL but the safe candidate set shows meaningful utility spread, then conclude:

> cascade-aware static compression may be feasible, but the current local search is insufficient.

Possible next methods:

- evolutionary search;
- Bayesian optimization;
- context-aware/non-additive search.

Do not implement these before establishing headroom.

## Negative Result

Stop static cascade-aware quantization if:

1. utility variation within the capability-safe set is small;
2. C3Q does not outperform KL;
3. best safe random configurations perform similarly;
4. calibration utility rankings do not generalize to held-out evaluation.

Then conclude:

> **Once capability is controlled, static precision allocation provides little additional leverage for cascade optimization.**

Do not invent another static layer-allocation score.

---

# 17. Alternative Direction After Negative Result

If Iteration 5 is negative, pivot away from static mixed precision.

## A. Precision Cascade / Adaptive Precision

Choose at runtime:

\[
a(x)\in\{Q4,Q8,LLM\}.
\]

Use query difficulty or cross-precision disagreement to determine whether to:

- answer with Q4;
- rerun at Q8;
- delegate to 7B.

This moves heterogeneity from layers to inputs.

## B. Compression-Aware Router

Keep the strongest quantizer fixed and train:

\[
r_\theta(x,Q)
\rightarrow
P(L\text{ improves over }S_Q).
\]

This treats compression as a routing condition rather than making the compressor solve the routing problem.

---

# 18. Minimum Viable Iteration 5

## Models

```text
SLM: Qwen2.5-1.5B-Instruct
Fallback: Qwen2.5-7B-Instruct
```

## Dataset

```text
Calibration: ARC train (~750)
Held-out: ARC validation (299)
```

## Budgets

```text
3.5 bits
4.0 bits
```

## Search

```text
Start from KL allocation
100–300 local exact-budget candidates
1–3 layer swaps
```

## Capability Constraint

Primary:

```text
KL <= 1.10 × KL(KL-baseline)
Calibration accuracy >= KL-baseline accuracy - 1 pp
```

## Baselines

```text
Uniform
Random MP
KL-sensitive MPQ
Local CUAQ
Best safe random
C3Q
```

## Main Metrics

```text
SLM accuracy
output KL
cascade accuracy
delegation rate
cascade utility
fallback correction
missed beneficial delegation
harmful escalation
```

## Main Go/No-Go

Proceed beyond ARC only if:

> C3Q improves held-out cascade utility over KL while preserving approximately the same standalone capability.

---

# 19. Recommended Execution Order

## Stage 1 — Build Safe Candidate Set

- generate 100–300 KL-neighborhood configurations;
- evaluate calibration capability;
- reject unsafe candidates.

Before held-out evaluation, report:

\[
|\mathcal{Q}_{safe}|.
\]

If almost no alternatives survive:

> the KL allocation is locally dominant and the search space has little useful flexibility.

## Stage 2 — Measure Utility Headroom

Within safe candidates:

- evaluate calibration cascade utility;
- measure variance/range;
- inspect correlation with accuracy/KL.

If utility spread is negligible:

> STOP.

Do not consume held-out evaluation.

## Stage 3 — Select C3Q

Choose:

\[
Q^*=\arg\max_{safe}\bar U(Q).
\]

Freeze:

- candidate;
- thresholds;
- utility aggregation;
- capability criteria.

## Stage 4 — Held-Out ARC Test

Evaluate:

- KL;
- best safe random;
- C3Q;
- uniform;
- FP.

Use paired statistics.

## Stage 5 — MMLU-Pro

Only if ARC passes.

## Stage 6 — Real Quantization

Only if MMLU-Pro also passes.

Then test:

- AWQ/GPTQ or equivalent packed quantization;
- memory;
- latency;
- throughput.

---

# 20. Literature Positioning

Iteration 5 is motivated by two adjacent findings in current quantization work:

1. **Layerwise local sensitivity can fail to predict global compressed-model behavior**, especially under aggressive mixed precision, because of inter-layer interactions.
2. **Direct search over whole compression configurations** can outperform additive local approximations when the search space is structured carefully.

Iteration 5 uses those observations conservatively: rather than replacing KL sensitivity, it uses KL to define a capability-preserving region and searches for cascade utility only inside that region.

This differs from standard mixed-precision quantization, which optimizes the compressed model in isolation, and from Iteration-4 CUAQ, which attempted to make local cascade utility the primary layer sensitivity.

---

# 21. Intended Contribution

The paper should not claim:

> cascade utility replaces ordinary quantization sensitivity.

Iteration 4 argues against that.

The stronger principle is:

> **Cascade-aware compression should be capability constrained. Standalone fidelity defines the safe compression region; system utility can then choose among similarly capable compressed models.**

The key empirical question is:

\[
\boxed{
\text{Does such a useful safe region actually exist?}
}
\]

If yes, C3Q becomes a constructive compression method.

If no, the negative result is itself useful:

> **For static mixed-precision compression, standalone model fidelity largely determines cascade utility once compression severity is fixed.**

That result motivates adaptive precision or compression-aware routing rather than further static bit-allocation objectives.

---

# 22. Immediate Execution Checklist

- [ ] Reuse Qwen2.5-1.5B → Qwen2.5-7B ARC cascade.
- [ ] Recompute/freeze KL-sensitive allocations at 3.5 and 4.0 bits.
- [ ] Generate 100–300 budget-preserving KL-neighborhood candidates.
- [ ] Compute calibration accuracy, prediction agreement, and output KL.
- [ ] Apply KL capability constraint.
- [ ] Apply calibration-accuracy constraint.
- [ ] Count safe configurations.
- [ ] Measure cascade-utility variance within the safe set.
- [ ] Measure accuracy–utility correlation within safe candidates.
- [ ] Stop if utility headroom is negligible.
- [ ] Otherwise select highest-utility safe configuration.
- [ ] Run 3 candidate-generation seeds if needed.
- [ ] Freeze C3Q before held-out evaluation.
- [ ] Evaluate C3Q vs KL vs best-safe-random vs uniform on ARC validation.
- [ ] Bootstrap cascade-utility differences.
- [ ] Analyze fallback corrections, harmful escalations, and missed beneficial delegations.
- [ ] Make final static-compression go/no-go decision.
- [ ] Only after a positive ARC result, replicate on MMLU-Pro.
- [ ] Only after replication, validate using real packed quantization.
