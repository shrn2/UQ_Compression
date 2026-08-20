# Iteration 4 Plan: Cascade-Aware Compression for Small-to-Large LLM Systems

## Working Title

**Compress for the System, Not the Model: Cascade-Aware Quantization for Small-to-Large Language Model Routing**

Alternative titles:

- **The Best Compressed Model Is Not Necessarily the Best Cascade**
- **Cascade-Utility-Aware Quantization for Small Language Models**
- **Can Compression Improve Model Collaboration? Quantization for SLM-to-LLM Cascades**

---

# 1. Motivation

Iterations 1–3 progressively ruled out several simpler hypotheses:

1. **Teacher-aware mixed-precision allocation** did not reliably outperform student-only quantization sensitivity.
2. **Trust-critical layers** were not meaningfully distinct from capability-critical layers for probability-based uncertainty.
3. **Quantization-induced delegation drift is real**, but routing drift alone was not consistently harmful.

Iteration 3 produced the key observation motivating this iteration:

> Quantization can substantially change uncertainty rankings and delegation decisions, including on examples where the FP and quantized SLM predict the same answer, but the changed routing policy is not necessarily worse.

This means that preserving the full-precision routing policy is not automatically the right compression objective.

The stronger question is:

> **Can the compression policy itself be optimized for the utility of the complete SLM → LLM cascade?**

Conventional quantization typically optimizes or approximates quantities such as:

- weight reconstruction error;
- activation reconstruction;
- perplexity;
- output KL divergence;
- standalone task accuracy.

But the deployed object is not the SLM alone. It is:

\[
x
\rightarrow
S_Q(x)
\rightarrow
r_Q(x)
\rightarrow
\begin{cases}
S_Q(x), & \text{keep local}\\
L(x), & \text{delegate}
\end{cases}
\]

where:

- \(S_Q\): compressed small model;
- \(L\): stronger fallback model;
- \(r_Q(x)\): routing score;
- final utility depends on both predictive quality and fallback cost.

Iteration 4 therefore studies:

\[
\boxed{
\text{compression policy}
\rightarrow
\text{cascade behavior}
\rightarrow
\text{system utility}
}
\]

rather than:

\[
\text{compression policy}
\rightarrow
\text{standalone SLM accuracy}.
\]

---

# 2. Main Research Questions

## RQ1 — Does optimizing compression for cascade utility outperform accuracy-oriented compression?

At a fixed compression budget:

\[
B,
\]

compare compression policies optimized for:

\[
\text{standalone model quality}
\]

versus:

\[
\text{end-to-end cascade utility}.
\]

Main hypothesis:

> A compression scheme with slightly worse standalone SLM accuracy can still produce a better cascade if it preserves the examples and confidence geometry that matter for profitable delegation.

---

## RQ2 — Which cascade-aware compression objective works best?

Compare three methods:

1. **CUAQ — Cascade-Utility-Aware Quantization**
2. **DBAQ — Delegation-Boundary-Aware Quantization**
3. **CARQ — Compression-as-Routing-Regularization Quantization**

These represent increasingly aggressive uses of cascade information:

- CUAQ protects components that preserve end-to-end cascade utility;
- DBAQ focuses compression resources near consequential delegation boundaries;
- CARQ intentionally searches for quantization configurations that improve routing behavior, even if they do not preserve FP routing.

---

## RQ3 — Is the best compressed model for standalone accuracy also the best compressed model for a cascade?

For every compression configuration \(Q\), measure:

\[
Acc(S_Q)
\]

and:

\[
U_{\text{cascade}}(Q).
\]

Test whether their rankings agree:

\[
\rho
\left(
rank(Acc(S_Q)),
rank(U_{\text{cascade}}(Q))
\right).
\]

A low or imperfect correlation directly supports the central thesis:

> **compression objectives should be deployment-aware.**

---

# 3. Shared Cascade Setup

## Small Model

Primary:

```text
Qwen2.5-1.5B-Instruct
```

This model was selected in Iteration 3 after the 0.5B model failed the router-viability gate.

## Fallback Model

Use the same fallback across datasets where feasible:

```text
Qwen2.5-7B-Instruct
```

Reasons:

- reduces experimental confounds;
- creates more meaningful delegation value;
- makes cascade utility easier to interpret across tasks.

---

# 4. Core Quantity: Delegation Value

Compression should not optimize only whether the SLM is wrong.

The actual system needs to know whether the fallback provides enough benefit to justify delegation.

For labeled example \(x\), define:

\[
G_Q(x)
=
U_L(x)
-
U_{S_Q}(x)
-
\lambda_c C_L.
\]

A simple correctness-based version is:

\[
G_Q(x)
=
\mathbf{1}[L(x)=y^*]
-
\mathbf{1}[S_Q(x)=y^*]
-
\lambda_c.
\]

A loss-based version is:

\[
G_Q(x)
=
\ell(S_Q(x),y^*)
-
\ell(L(x),y^*)
-
\lambda_c.
\]

Interpretation:

- \(G_Q(x)>0\): delegation is beneficial;
- \(G_Q(x)\approx0\): delegation is marginal;
- \(G_Q(x)<0\): keeping the example local is preferable.

For the first implementation, use the correctness-based version because it is easy to audit.

---

# 5. Shared Cascade Utility

Primary system utility:

\[
U_{\text{cascade}}
=
Acc_{\text{cascade}}
-
\lambda_c DR,
\]

where:

\[
DR=P[\text{delegate}].
\]

Evaluate several values:

\[
\lambda_c
\in
\{0,0.01,0.025,0.05,0.1\}.
\]

Do not choose one \(\lambda_c\) and report only that condition.

Also report the raw:

\[
\text{cascade accuracy}
\quad \text{vs.} \quad
\text{delegation rate}
\]

frontier so conclusions are not dependent on the utility coefficient.

---

# 6. Quantization Search Space

## Phase A — Controlled Scientific Experiment

Use the same fake-quantization infrastructure first.

Candidate block precisions:

\[
b_l \in \{2,4,8\}.
\]

If intermediate precision is implemented reliably, expand to:

\[
b_l \in \{2,3,4,5,6,8\}.
\]

Recommended average budgets:

\[
\bar b
\in
\{3.0,3.5,4.0,5.0\}.
\]

The main comparison should use **exactly matched parameter-weighted average-bit budgets**.

## Phase B — Real Quantization

Only after a cascade-aware scheme is positive under controlled fake quantization.

Validate with real packed quantization using:

- GPTQ;
- AWQ;
- GGUF/llama.cpp where appropriate.

Then report:

- real model size;
- peak memory;
- latency;
- throughput;
- fallback-call cost.

---

# 7. Method 1 — CUAQ: Cascade-Utility-Aware Quantization

## 7.1 Core Idea

Instead of defining layer sensitivity as:

\[
C_{l,b}
=
D(
p_{FP},
p_{Q_{l,b}}
),
\]

define sensitivity by the **change in cascade utility** produced by quantizing that component.

For layer \(l\) and precision \(b\):

\[
S^{CUAQ}_{l,b}
=
U_{\text{cascade}}(Q_{\text{reference}})
-
U_{\text{cascade}}(Q_{l,b}).
\]

A large value means:

> quantizing this layer to \(b\) bits damages the deployed cascade.

A small or negative value means:

> the layer can be compressed aggressively without reducing cascade utility.

## 7.2 Layer-Probing Version

Start from an FP or high-precision reference model.

For every layer \(l\):

1. quantize only layer \(l\) to \(b\);
2. run the compressed SLM;
3. apply the fixed routing protocol;
4. invoke cached fallback outputs;
5. compute:
   - SLM accuracy;
   - delegation rate;
   - cascade accuracy;
   - cascade utility.

Then:

\[
S^{CUAQ}_{l,b}
=
U_{FP}
-
U_{l,b}.
\]

## 7.3 Allocation

Choose:

\[
b_1,\ldots,b_L
\]

to maximize:

\[
\sum_l
-
S^{CUAQ}_{l,b_l}
\]

subject to:

\[
\sum_l N_l b_l
\le
B.
\]

This can be solved as a multiple-choice knapsack problem.

## 7.4 Context-Aware Extension

If the simple layer-local CUAQ works, estimate marginal value under partially quantized contexts:

\[
S^{CUAQ-context}_{l,b}
=
\mathbb{E}_{Q_{-l}}
[
U(Q_{-l},b_{\text{high}})
-
U(Q_{-l},b)
].
\]

Procedure:

1. sample several partial quantization configurations;
2. modify precision of layer \(l\);
3. measure marginal cascade-utility change;
4. average over contexts.

Do not implement this extension before the layer-local version passes a positive gate.

## 7.5 CUAQ Hypothesis

At matched bits:

\[
U_{\text{cascade}}(CUAQ)
>
U_{\text{cascade}}(\text{KL-sensitive MPQ})
\]

even if:

\[
Acc(S_{CUAQ})
\le
Acc(S_{\text{KL-MPQ}}).
\]

That result would directly support deployment-aware compression.

---

# 8. Method 2 — DBAQ: Delegation-Boundary-Aware Quantization

## 8.1 Motivation

Not every input matters equally to routing.

Compression-induced score changes are especially operationally relevant when an example is near the accept/defer boundary.

However, boundary proximity alone is insufficient.

The example should also have meaningful delegation value.

Define a calibration weight:

\[
w(x)
=
v(x)\cdot k(x),
\]

where:

\[
v(x)=|G(x)|
\]

captures consequence and:

\[
k(x)
=
\exp
\left(
-\frac{|r_{FP}(x)-\tau|}{T}
\right)
\]

captures proximity to the delegation boundary.

Thus:

\[
w(x)
=
|G(x)|
\exp
\left(
-\frac{|r_{FP}(x)-\tau|}{T}
\right).
\]

## 8.2 Boundary-Aware Layer Sensitivity

For layer \(l\) and precision \(b\):

\[
S^{DBAQ}_{l,b}
=
\mathbb{E}_x
\left[
w(x)
D(
p_{FP}(x),
p_{Q_{l,b}}(x)
)
\right].
\]

Alternative direct decision version:

\[
S^{DBAQ-decision}_{l,b}
=
\mathbb{E}_x
\left[
|G(x)|
\mathbf{1}
[
D_{Q_{l,b}}(x)
\neq
D_{\text{target}}(x)
]
\right].
\]

Start with the distributional version because it is smoother.

Use the direct decision version as an ablation.

## 8.3 Multiple Operating Points

Because Iteration 3 evaluated:

\[
10\%,20\%,30\%,50\%
\]

delegation rates, DBAQ should not optimize one arbitrary threshold.

Define:

\[
S^{DBAQ}_{l,b}
=
\frac{1}{|R|}
\sum_{r\in R}
S^{DBAQ}_{l,b}(r),
\]

where:

\[
R=\{0.1,0.2,0.3,0.5\}.
\]

## 8.4 DBAQ Hypothesis

DBAQ should be especially competitive when:

- fallback calls are expensive;
- routing mistakes near the threshold are consequential;
- full cascade-utility probing is too expensive.

It is the lightweight approximation to CUAQ.

---

# 9. Method 3 — CARQ: Compression as Routing Regularization

## 9.1 Motivation

Iteration 3 showed that quantized rankings are not always inferior to FP rankings.

Therefore:

> FP routing behavior should not be treated as an oracle.

CARQ deliberately searches for compression configurations that improve the cascade rather than preserve the FP model.

## 9.2 Objective

For quantization configuration \(Q\):

\[
J(Q)
=
U_{\text{cascade}}(Q)
-
\lambda_b Cost_{\text{memory}}(Q).
\]

Equivalently:

\[
\max_Q
U_{\text{cascade}}(Q)
\]

subject to:

\[
Bits(Q)\le B.
\]

Conceptual distinction:

### CUAQ

Estimate which components are costly to compress, then protect them.

### CARQ

Directly search for the quantization configuration that produces the **best cascade**.

## 9.3 Search Procedure

Full exhaustive search is impossible.

Use a lightweight search.

### Stage 1 — Candidate Initialization

Generate configurations from:

- random mixed precision;
- KL-sensitive allocation;
- CUAQ allocation;
- DBAQ allocation.

### Stage 2 — Local Search

At each step:

1. select one or two layers;
2. increase/decrease precision while respecting budget;
3. compute calibration cascade utility;
4. keep the change if utility improves.

Suggested search budget:

```text
100–300 candidate configurations
```

per target bit budget.

## 9.4 Optional Evolutionary Search

If local search stalls:

- mutation = modify 1–3 layer precisions;
- selection = top cascade utility;
- crossover = optional.

Keep the method simple enough to reproduce.

## 9.5 CARQ Hypothesis

There may exist:

\[
Q^*
\]

such that:

\[
U_{\text{cascade}}(Q^*)
>
U_{\text{cascade}}(FP)
\]

at lower memory.

This is a stretch hypothesis.

A weaker positive result is sufficient:

\[
U_{\text{cascade}}(CARQ)
>
U_{\text{cascade}}(\text{accuracy-oriented MPQ})
\]

at the same bit budget.

---

# 10. Dataset

## 10.1 Primary Development Dataset — ARC-Challenge

Reuse the established pipeline.

### Calibration

```text
ARC-Challenge train
~750 examples
```

Use for:

- CUAQ sensitivity;
- DBAQ boundary weights;
- CARQ search;
- routing-threshold calibration.

### Evaluation

```text
ARC-Challenge validation
299 examples
```

Use only after compression configurations are frozen.

ARC is mainly a fast development benchmark.

## 10.2 Main Evaluation Dataset — MMLU-Pro

This should be the primary result if compute permits.

### Calibration

Current Iteration-3 setup used the validation split for calibration.

Because the validation set is small, improve this if possible by creating a larger calibration pool from public multiple-choice training data while keeping MMLU-Pro test untouched.

Minimum:

```text
MMLU-Pro validation for calibration
```

### Evaluation

Prefer:

```text
2,000–full MMLU-Pro test examples
```

rather than only the previous 1,000-example subset if compute permits.

At minimum, preserve the same deterministic category-stratified sample for direct comparison with Iteration 3.

## 10.3 Optional Hard Benchmark — GPQA Diamond

Only after CUAQ/DBAQ/CARQ shows a positive MMLU-Pro result.

Purpose:

- stronger fallback advantage;
- harder questions;
- test whether cascade-aware compression becomes more valuable when delegation has higher potential payoff.

---

# 11. Calibration Data Requirements

For every calibration example, cache:

\[
p_{FP}(x),
\quad
p_Q(x),
\quad
y^*,
\quad
L(x),
\quad
r(x).
\]

Also cache:

- FP correctness;
- quantized correctness;
- fallback correctness;
- delegation value \(G(x)\);
- router score;
- fallback uplift class.

Useful classes:

1. SLM correct / LLM correct;
2. SLM wrong / LLM correct;
3. SLM correct / LLM wrong;
4. SLM wrong / LLM wrong.

The most valuable delegation class is:

\[
S\text{ wrong},\quad L\text{ correct}.
\]

---

# 12. Routing Protocol

Use the same uncertainty/router signal for all compression methods during the first comparison.

Recommended starting point:

- calibration-selected margin or entropy;
- target rates:
  \[
  10\%,20\%,30\%,50\%.
  \]

Do **not** let each compression method use a different learned router initially.

The purpose is to isolate the value of the compression policy.

---

# 13. Baselines

## B0 — FP/BF16 Student

Reference:

```text
Qwen2.5-1.5B-Instruct
```

Report standalone and cascade performance.

## B1 — Uniform Quantization

At comparable precisions:

```text
uniform Q8
uniform Q4
uniform Q2
```

or nearest matched average-bit conditions.

## B2 — Random Mixed Precision

At each exact budget:

- sample at least 10 configurations;
- match parameter-weighted average bits exactly.

Report:

- mean;
- standard deviation;
- best random configuration.

Important because CARQ is itself a search method.

## B3 — KL-Sensitive Mixed Precision

Use conventional output-distribution sensitivity:

\[
S^{KL}_{l,b}
=
E[
KL(p_{FP}\Vert p_{Q_{l,b}})
].
\]

This is the most important internal baseline.

## B4 — Accuracy-Drop Mixed Precision

Use:

\[
S^{Acc}_{l,b}
=
Acc(S_{FP})
-
Acc(S_{Q_{l,b}}).
\]

This is a label-aware standalone-performance oracle baseline.

## B5 — UQ-Preserving Mixed Precision

Reuse the Iteration-2 idea:

\[
S^{UQ}_{l,b}
=
1-
\rho(
u_{FP},
u_{Q_{l,b}}
).
\]

Iteration 2 suggests this overlaps strongly with KL sensitivity, but it remains a useful negative baseline.

## B6 — Current External Mixed-Precision Baseline

Use one external adaptive mixed-precision method if implementation is feasible.

Preferred:

```text
MixQuant
```

Alternative:

```text
MXSens / SliM-LLM
```

Do not implement several complex external methods for a workshop submission.

## B7 — Oracle Cascade-Utility Allocation

If computationally feasible, evaluate a restricted search over bit allocations using calibration labels only.

Purpose:

> estimate how much headroom exists for a cascade-aware compression policy.

CARQ approximates this search.

---

# 14. Method Comparison Matrix

At each budget:

\[
\bar b
\in
\{3.0,3.5,4.0,5.0\},
\]

compare:

1. Uniform
2. Random MP
3. KL-sensitive MPQ
4. Accuracy-sensitive MPQ
5. UQ-sensitive MPQ
6. External MPQ baseline
7. **CUAQ**
8. **DBAQ**
9. **CARQ**

If runtime is limited, minimum paper comparison:

1. Uniform
2. Random
3. KL-sensitive
4. CUAQ
5. DBAQ
6. CARQ

---

# 15. Evaluation Metrics

## 15.1 Standalone SLM Metrics

### Accuracy

\[
Acc(S_Q)
\]

### Prediction agreement with FP

\[
P[
\hat y_Q=\hat y_{FP}
].
\]

### Output KL

\[
E[
KL(p_{FP}\Vert p_Q)
].
\]

## 15.2 Compression Metrics

Report:

- average bits/weight;
- estimated model size;
- compression ratio.

For real quantization:

- packed size;
- peak memory;
- tokens/sec;
- latency.

Do not claim runtime benefit from fake quantization.

## 15.3 Router Metrics

Report:

- error AUROC;
- AUPRC;
- AURC;
- uncertainty-rank correlation with FP;
- matched-rate delegation flip rate.

These are diagnostic rather than the final optimization target.

## 15.4 Cascade Metrics

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
P[
S_Q\neq y^*,
L=y^*
\mid
D=1
].
\]

### Harmful escalation

\[
P[
S_Q=y^*,
L\neq y^*
\mid
D=1
].
\]

### Missed beneficial delegation

\[
P[
S_Q\neq y^*,
L=y^*,
D=0
].
\]

## 15.5 Cascade Utility

For every cost coefficient:

\[
U
=
Acc_{\text{cascade}}
-
\lambda_c DR.
\]

Main comparison:

\[
\Delta U_{\text{method}}
=
U_{\text{method}}
-
U_{\text{KL baseline}}.
\]

---

# 16. Central Paper Plots

## Figure 1 — Standalone Accuracy vs Cascade Utility

Each point = compression configuration.

x-axis:

\[
Acc(S_Q)
\]

y-axis:

\[
U_{\text{cascade}}.
\]

This directly tests:

> Is the best standalone compressed model the best cascade component?

## Figure 2 — Cascade Accuracy vs Delegation Rate

Curves for:

- FP;
- uniform;
- KL-sensitive;
- CUAQ;
- DBAQ;
- CARQ.

## Figure 3 — Cascade Utility vs Average Bits

x-axis:

\[
\text{average bits/weight}
\]

y-axis:

\[
U_{\text{cascade}}.
\]

Goal:

> show whether cascade-aware compression improves the system-efficiency frontier.

## Figure 4 — Bit Allocation Across Layers

For:

- KL-sensitive;
- CUAQ;
- DBAQ;
- CARQ.

Question:

> Which layers does cascade-aware compression protect differently?

## Figure 5 — Where the Gains Come From

Compare:

- fallback corrections;
- unnecessary escalations;
- missed beneficial delegations.

---

# 17. Statistical Analysis

All methods must be evaluated on identical examples.

Use:

- paired bootstrap confidence intervals;
- McNemar test for paired final correctness;
- bootstrap differences in cascade utility.

For random mixed precision:

- at least 10 seeds;
- report mean ± SD;
- compare proposed methods with the random distribution.

For CARQ:

- use at least 3 search seeds if stochastic.

---

# 18. Important Ablations

## A1 — Remove Delegation Cost

Set:

\[
\lambda_c=0.
\]

Tests whether the method reduces to cascade-accuracy optimization.

## A2 — Error-Based vs Uplift-Based Utility

Compare:

\[
G_{\text{error}}
=
\mathbf1[S_Q\neq y^*]
\]

against:

\[
G_{\text{uplift}}
=
\mathbf1[L=y^*]
-
\mathbf1[S_Q=y^*].
\]

Hypothesis:

> fallback uplift should be a better system objective than SLM error alone.

## A3 — DBAQ Without Boundary Proximity

Use:

\[
w(x)=|G(x)|.
\]

Compare against:

\[
w(x)
=
|G(x)|
e^{-|r(x)-\tau|/T}.
\]

## A4 — DBAQ Without Utility Weight

Use only:

\[
w(x)
=
e^{-|r(x)-\tau|/T}.
\]

## A5 — CUAQ Local vs Context-Aware

If feasible:

\[
S_{l,b}^{local}
\]

vs.

\[
S_{l,b}^{context}.
\]

## A6 — CARQ Initialization

Initialize CARQ from:

- random;
- KL-sensitive;
- CUAQ;
- DBAQ.

Measure dependence on initialization.

---

# 19. Success Criteria

## Strong Positive Result

Iteration 4 is strongly positive if, at matched average bits:

1. at least one cascade-aware method significantly improves:

\[
U_{\text{cascade}}
\]

over KL-sensitive and random MP;

2. gains replicate on both ARC and MMLU-Pro;
3. the method changes allocation meaningfully rather than reproducing the KL ranking;
4. improvement comes from better delegation outcomes rather than simply more fallback calls.

Ideal result:

\[
Acc(S_{CUAQ})
<
Acc(S_{KL})
\]

but:

\[
U_{\text{cascade}}(CUAQ)
>
U_{\text{cascade}}(KL).
\]

## Very Strong Result

CARQ finds a compressed configuration:

\[
Q^*
\]

with:

\[
U_{\text{cascade}}(Q^*)
>
U_{\text{cascade}}(FP)
\]

while using substantially fewer SLM bits.

This would support:

> **compression as routing regularization.**

## Moderate Positive Result

If CUAQ/DBAQ achieve the same cascade quality at lower fallback rate or lower bit budget:

> frame the contribution as improving the **efficiency frontier**.

## Negative Result

Stop this direction if:

- cascade-aware allocations collapse to the same layer ranking as KL sensitivity;
- gains do not beat random mixed precision;
- CARQ search overfits calibration data;
- held-out cascade utility does not improve.

Then the strongest remaining result is the Iteration-3 **delegation stability audit**.

---

# 20. Recommended Execution Order

## Stage 1 — Cheapest Test

Implement **CUAQ local sensitivity only**.

Use:

```text
ARC
Qwen2.5-1.5B → Qwen2.5-7B
2/4/8-bit probes
3.5- and 4.0-bit budgets
```

Compare:

- KL-sensitive MPQ;
- random MP;
- CUAQ.

If CUAQ does not outperform these, do not immediately implement CARQ.

## Stage 2 — DBAQ

If CUAQ shows signal:

- compute delegation-value weights;
- add boundary proximity;
- compare DBAQ with CUAQ.

## Stage 3 — CARQ

Only if the search space shows meaningful variation in cascade utility.

Run:

- 100–300 candidate configurations;
- 3 search seeds;
- calibration-only selection.

Freeze the best configuration before evaluation.

## Stage 4 — MMLU-Pro

Replicate only methods that survive ARC.

## Stage 5 — Real Quantization

Only after positive MMLU-Pro result:

- GPTQ/AWQ;
- real size;
- memory;
- latency;
- throughput.

---

# 21. Minimum Viable Iteration 4

## Models

```text
SLM: Qwen2.5-1.5B-Instruct
Fallback: Qwen2.5-7B-Instruct
```

## Dataset

```text
ARC calibration: ~750
ARC evaluation: 299
```

## Bits

```text
{2,4,8}
```

## Budgets

```text
3.5 bits
4.0 bits
```

## Methods

```text
KL-sensitive
Random MP
CUAQ
DBAQ
CARQ
```

## Metrics

```text
SLM accuracy
cascade accuracy
delegation rate
cascade utility
fallback correction rate
missed beneficial delegation
average bits
```

## Main Go/No-Go

Proceed to MMLU-Pro only if at least one cascade-aware method beats KL-sensitive mixed precision and random MP on held-out ARC cascade utility.

---

# 22. Literature Positioning

## Standard / Mixed-Precision Quantization

Methods such as GPTQ, AWQ, MixQuant, MXSens, SliM-LLM, and KL-based sensitivity methods optimize or estimate standalone model distortion/performance under quantization.

Iteration 4 differs by making **downstream cascade utility** the compression objective.

## Compression and Uncertainty

Recent work shows that quantization can change uncertainty and calibration without those changes perfectly tracking accuracy.

Iterations 1–3 already established that this effect can propagate into changed routing decisions.

Iteration 4 asks whether that observation can be turned into a constructive compression algorithm.

## Uncertainty-Based Routing

Existing SLM→LLM routing work uses uncertainty, calibration, or learned routing to decide when to invoke a stronger model.

Iteration 4 differs by changing:

\[
\textbf{the compression policy itself}
\]

to optimize the downstream routing system.

## Cascade-Aware Training

Prior cascade-aware training motivates optimizing small models for their role inside a larger system rather than as isolated predictors.

Iteration 4 transfers this principle to **post-training compression**.

---

# 23. Intended Contribution

The central claim should be:

> **Conventional quantization optimizes the compressed model in isolation, but deployed SLMs increasingly operate inside cascades. We introduce cascade-aware compression objectives that allocate precision according to downstream delegation utility rather than standalone prediction distortion.**

The strongest conceptual result would be:

> **The compressed model with the highest standalone accuracy is not necessarily the compressed model that yields the best accuracy–cost cascade frontier.**

The three methods test complementary versions of this principle:

### CUAQ

> Preserve components whose compression damages end-to-end cascade utility.

### DBAQ

> Spend precision on consequential examples near delegation boundaries.

### CARQ

> Treat quantization itself as a controllable perturbation that may improve routing behavior.

---

# 24. Immediate Execution Checklist

- [ ] Standardize primary cascade to Qwen2.5-1.5B → Qwen2.5-7B.
- [ ] Cache fallback outputs for ARC calibration/evaluation.
- [ ] Define correctness-based delegation value \(G(x)\).
- [ ] Reuse 2/4/8-bit layer probes where possible.
- [ ] Compute KL-sensitive baseline allocations.
- [ ] Implement CUAQ layer-utility sensitivity.
- [ ] Solve matched-budget CUAQ allocations.
- [ ] Evaluate CUAQ vs KL-sensitive vs random MP.
- [ ] If positive, compute DBAQ utility/boundary weights.
- [ ] Evaluate DBAQ at the same budgets.
- [ ] Build CARQ local-search procedure.
- [ ] Use calibration data only for CARQ search.
- [ ] Freeze all configurations before held-out evaluation.
- [ ] Compare standalone accuracy vs cascade utility.
- [ ] Compare cascade accuracy vs delegation rate.
- [ ] Compare cascade utility vs average bits.
- [ ] Analyze fallback corrections and missed-benefit cases.
- [ ] Make ARC go/no-go decision.
- [ ] Replicate surviving methods on MMLU-Pro.
- [ ] Only after replication, test real GPTQ/AWQ deployment.
