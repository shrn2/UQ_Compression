# Research Plan: Teacher-Uncertainty-Aware Compression for Small Language Models

## 1. Working Title

**Teacher-Uncertainty-Aware Mixed-Precision Compression for Small Language Models**

Alternative:

**Preserve What the Teacher Knows: Uncertainty-Guided Compression of Small Language Models**

---

## 2. Motivation

Most post-training quantization and mixed-precision compression methods determine which parts of a model should retain higher precision using quantities derived from the **small model itself**, such as:

- weight magnitude,
- activation magnitude,
- Hessian/sensitivity estimates,
- output perturbation under quantization,
- layer-wise reconstruction error.

However, these signals do not directly indicate whether the capabilities being damaged by compression are actually important.

A stronger large language model can provide an additional semantic signal.

For an input \(x\), the large teacher may be:

- confident and correct,
- uncertain,
- confident while the small model disagrees,
- similarly uncertain to the small model.

The central hypothesis is:

> **Compression should preferentially preserve the parts of a small model responsible for examples where the stronger teacher has reliable knowledge that the small model does not yet represent well.**

Instead of treating all calibration examples equally, teacher uncertainty and teacher–student disagreement are used to determine the importance of quantization-induced perturbations.

The teacher is required only **offline during calibration/compression**, so the final compressed model retains the deployment advantages of a conventional small model.

---

# 3. Main Research Questions

## RQ1 — Can large-model uncertainty identify capabilities worth preserving during compression?

**Question**

Can uncertainty from a stronger teacher model help identify which quantization-induced changes in a smaller model are semantically important?

Specifically, compare conventional student-only sensitivity:

\[
I_l^{S}
=
\mathbb{E}_{x}
[
D(
p_S^{FP}(x),
p_S^{Q_l}(x)
)
]
\]

against teacher-aware sensitivity:

\[
I_l^{T}
=
\mathbb{E}_{x}
[
w_T(x)
D(
p_S^{FP}(x),
p_S^{Q_l}(x)
)
].
\]

Here:

- \(S^{FP}\): full-precision student,
- \(S^{Q_l}\): student with layer \(l\) temporarily quantized,
- \(D(\cdot,\cdot)\): output-distribution divergence,
- \(w_T(x)\): importance inferred from the teacher.

### Hypothesis

Teacher-aware sensitivity should identify a different subset of layers/capabilities than purely numerical student sensitivity.

---

## RQ2 — Can teacher-aware bit allocation improve the accuracy–reliability–memory frontier?

**Question**

At the same average bit budget, can teacher-informed mixed-precision quantization outperform:

- uniform quantization,
- standard PTQ,
- student-only mixed precision,
- random mixed precision,
- existing sensitivity-driven mixed-precision methods?

### Hypothesis

Teacher-aware allocation should preserve more useful capability per bit because higher precision is allocated to parameters whose perturbation affects examples where the teacher contains reliable knowledge.

Main comparison:

\[
\text{Task Performance}
\quad\text{vs}\quad
\text{Average Bits/Weight}.
\]

---

## RQ3 — Does teacher-aware compression better preserve uncertainty and selective behavior?

Compression may preserve raw accuracy while damaging uncertainty information.

Therefore evaluate whether teacher-guided compression preserves:

\[
U_{S_Q}(x)
\approx
U_{S_{FP}}(x)
\]

and whether the compressed model continues to distinguish correct from incorrect predictions.

### Hypothesis

Teacher-aware compression should better preserve:

- uncertainty ranking,
- calibration,
- error detection,
- teacher–student uncertainty structure,

than quantization methods optimized only for prediction reconstruction.

This also provides a connection to downstream applications such as:

- selective prediction,
- model delegation,
- SLM → LLM escalation,
- uncertainty-aware agents.

---

# 4. Proposed Method

## 4.1 Models

Use models from the same model family when possible.

Example:

```text
Teacher: Qwen 7B–14B+
Student: Qwen 1.5B–3B
```

Same-family models reduce confounding due to:

- tokenizer differences,
- architecture differences,
- unrelated training distributions.

Only the student is compressed.

The teacher is used offline.

---

# 5. Calibration Pipeline

For each calibration example \(x\):

## Step 1 — Run teacher

Obtain:

\[
p_T(y|x)
\]

and calculate teacher uncertainty:

\[
H_T(x)
=
-\sum_k p_T(k|x)\log p_T(k|x).
\]

Normalize:

\[
\tilde H_T(x)
=
\frac{H_T(x)}{H_{\max}}.
\]

Teacher confidence:

\[
C_T(x)=1-\tilde H_T(x).
\]

---

## Step 2 — Run full-precision student

Obtain:

\[
p_S(y|x).
\]

Calculate teacher–student disagreement:

\[
G(x)
=
D_{\mathrm{KL}}
(
p_T(y|x)
\Vert
p_S(y|x)
).
\]

---

## Step 3 — Construct example importance score

### Proposed main score

\[
w(x)
=
C_T(x)
\cdot
G(x).
\]

Therefore:

\[
w(x)
=
(1-\tilde H_T(x))
D_{\mathrm{KL}}
(
p_T
\Vert
p_S
).
\]

Interpretation:

### High weight

```text
Teacher confident
+
Student disagrees
```

→ likely useful capability that should be protected.

### Low weight

```text
Teacher uncertain
```

or

```text
Teacher and student already agree
```

→ less reason to spend additional precision.

---

# 6. Layer-Wise Quantization Sensitivity

For every student layer \(l\):

1. keep all other layers at full precision;
2. temporarily quantize layer \(l\);
3. evaluate calibration examples;
4. calculate output perturbation.

For example:

\[
\Delta_l(x)
=
D_{\mathrm{KL}}
(
p_S^{FP}(x)
\Vert
p_S^{Q_l}(x)
).
\]

Teacher-aware layer importance:

\[
I_l
=
\mathbb{E}_{x}
[
w(x)\Delta_l(x)
].
\]

Large \(I_l\):

> Quantizing this layer damages behavior on examples considered important by the teacher.

Small \(I_l\):

> This layer is a better candidate for aggressive quantization.

---

# 7. Mixed-Precision Bit Allocation

Possible precision choices:

\[
b_l\in\{2,3,4,6,8\}.
\]

A simpler initial setup:

\[
b_l\in\{3,4,8\}.
\]

Optimize:

\[
\min_{\{b_l\}}
\sum_l I_l E_l(b_l)
\]

subject to:

\[
\sum_l N_l b_l \le B,
\]

where:

- \(N_l\): number of parameters in layer \(l\),
- \(B\): total memory/bit budget,
- \(E_l(b_l)\): estimated quantization error at bit width \(b_l\).

Target average precision levels:

```text
4.0 bits/weight
3.5 bits/weight
3.0 bits/weight
```

This generates a full compression frontier rather than one operating point.

---

# 8. Dataset Design

Two separate dataset roles should be maintained.

## 8.1 Generic Quantization Calibration Data

Use conventional text data such as:

**C4**

Suggested size:

```text
128–256 sequences
~2048 tokens per sequence
```

Purpose:

- GPTQ/AWQ calibration,
- activation collection,
- conventional quantization baseline.

---

## 8.2 Teacher-Uncertainty Calibration Dataset

Use only training/development splits.

Suggested mixture:

| Dataset | Approx. calibration examples | Purpose |
|---|---:|---|
| MMLU / MMLU-Pro development data | 500–750 | Broad knowledge/reasoning |
| GSM8K train | 300–500 | Mathematical reasoning |
| ARC-Challenge train/dev | 250–300 | Science reasoning |
| Optional commonsense set | 200–300 | Additional diversity |

Total:

\[
\approx 1,000-2,000
\]

examples.

For a minimum experiment:

\[
500-1,000
\]

may be sufficient.

No test examples should influence bit allocation.

---

# 9. Evaluation Datasets

## 9.1 MMLU-Pro

**Primary benchmark.**

Advantages:

- difficult,
- broad domains,
- multiple-choice probabilities available,
- easy uncertainty computation.

Metrics:

- accuracy,
- ECE,
- Brier score,
- AUROC for error detection.

---

## 9.2 ARC-Challenge

Tests science reasoning and cross-domain generalization.

Useful because it differs structurally from MMLU-Pro.

Metrics:

- accuracy,
- uncertainty/error-detection AUROC.

---

## 9.3 GPQA / GPQA Diamond

Hard reasoning benchmark.

Especially useful for studying cases where:

\[
U_T(x)
\]

is itself large.

Questions:

- Does teacher confidence weighting fail when the teacher is uncertain?
- Does uncertainty-aware compression remain useful on difficult/OOD samples?

Primary metric:

- accuracy.

Secondary:

- calibration / error ranking where feasible.

---

## 9.4 GSM8K

Tests free-form generative reasoning.

Main metric:

\[
\text{Exact Match}.
\]

Important because the compression policy should not only improve multiple-choice tasks.

---

# 10. Minimum Dataset Configuration

For an initial workshop experiment:

### Calibration

```text
500–1,000 mixed reasoning examples
```

### Evaluation

```text
MMLU-Pro
ARC-Challenge
GSM8K
```

Add GPQA only after the main pipeline works.

---

# 11. Baselines

## Baseline 0 — Full-Precision Student

\[
S_{FP16/BF16}
\]

Purpose:

Upper-bound reference for:

- accuracy,
- calibration,
- uncertainty.

---

## Baseline 1 — Uniform Quantization

All layers use the same bit width.

Examples:

```text
W4A16
W3A16
```

Purpose:

Determine whether mixed precision is necessary.

---

## Baseline 2 — GPTQ

Standard post-training quantization baseline.

Compare at equivalent memory/average-bit budgets.

---

## Baseline 3 — AWQ

Activation-aware weight quantization.

Important conceptual comparison:

```text
AWQ:
student activation → importance

Proposed:
teacher knowledge → importance
```

---

## Baseline 4 — Random Mixed Precision

Randomly assign:

\[
b_l\in\{3,4,8\}
\]

while maintaining exactly the same average memory budget as the proposed method.

Run several random seeds.

Purpose:

Demonstrate that any improvements do not arise simply from allocating some layers extra bits.

---

# 12. Most Important Baseline: Student-Only Sensitivity

Use exactly the proposed compression framework but remove teacher information.

Define:

\[
I_l^{student}
=
\mathbb{E}_{x}
[
D(
p_S^{FP}(x),
p_S^{Q_l}(x)
)
].
\]

Then allocate bits using \(I_l^{student}\).

This baseline isolates the central contribution.

Comparison:

\[
\boxed{
\text{student-only sensitivity}
\quad\text{vs}\quad
\text{teacher-aware sensitivity}
}
\]

If the proposed method does not outperform this baseline, the teacher may not actually be contributing useful information.

---

# 13. Existing Mixed-Precision Baseline

Include at least one strong mixed-precision quantization method if implementation permits.

Possible candidates:

- MixQuant,
- SliM-LLM,
- another recent sensitivity-driven mixed-precision PTQ method.

This evaluates whether teacher-aware importance provides additional benefit over current mixed-precision heuristics.

---

# 14. Signal Ablations

These are critical.

The method uses:

\[
w(x)
=
(1-\tilde H_T(x))
D_{\mathrm{KL}}(p_T||p_S).
\]

Test the following.

---

## A1. No Teacher

\[
w(x)=1.
\]

Equivalent to student-only sensitivity.

---

## A2. Teacher Confidence Only

\[
w(x)
=
1-\tilde H_T(x).
\]

Tests whether teacher confidence alone is sufficient.

---

## A3. Teacher Uncertainty Only

\[
w(x)
=
\tilde H_T(x).
\]

Tests the opposite hypothesis:

> difficult/uncertain teacher examples deserve more capacity.

---

## A4. Teacher–Student Disagreement Only

\[
w(x)
=
D_{\mathrm{KL}}(p_T||p_S).
\]

Tests whether uncertainty itself contributes beyond teacher/student disagreement.

---

## A5. Proposed Combined Score

\[
w(x)
=
(1-\tilde H_T(x))
D_{\mathrm{KL}}(p_T||p_S).
\]

Interpretation:

> Protect examples where the teacher knows something that the student does not.

---

# 15. Task Performance Metrics

For classification/reasoning:

\[
Accuracy.
\]

For GSM8K:

\[
Exact\ Match.
\]

Report performance difference from full precision:

\[
\Delta Acc
=
Acc(S_Q)-Acc(S_{FP}).
\]

---

# 16. Compression Metrics

Report:

### Average bits per parameter

\[
\bar b
=
\frac{\sum_l N_lb_l}
{\sum_lN_l}.
\]

### Model storage

```text
GB
```

### Compression ratio

\[
CR
=
\frac{Size_{FP}}
{Size_Q}.
\]

### Runtime

If the implementation supports true mixed-precision execution:

- tokens/sec,
- latency,
- peak memory.

If not, avoid claiming theoretical bit savings automatically imply latency improvements.

---

# 17. Reliability / Uncertainty Metrics

## Expected Calibration Error

\[
ECE.
\]

---

## Brier Score

Measures quality of probabilistic predictions.

---

## Error-Detection AUROC

Treat uncertainty as a detector of incorrect predictions:

\[
AUROC(U(x),\mathbf 1[\hat y\neq y]).
\]

This may be more important than ECE for future delegation/selective prediction.

---

## Risk–Coverage Curve

Sort predictions by uncertainty.

As uncertain samples are rejected, evaluate accuracy/risk on remaining samples.

Report:

- AURC,
- selective accuracy.

---

# 18. Uncertainty Preservation Metrics

Measure how compression changes uncertainty relative to the full-precision student.

### Per-example distortion

\[
\Delta U(x)
=
|U_{S_Q}(x)-U_{S_{FP}}(x)|.
\]

Report:

\[
\mathbb{E}[\Delta U].
\]

---

## Ranking Preservation

Calculate:

\[
\rho(
U_{S_Q},
U_{S_{FP}}
)
\]

using Spearman correlation.

This determines whether the compressed model preserves the ordering of easy and difficult samples.

---

## Teacher–Student Uncertainty Alignment

Measure:

\[
\rho(
U_T,
U_{S_Q}
).
\]

Compare with:

\[
\rho(
U_T,
U_{S_{FP}}
).
\]

Question:

> Does teacher-aware compression preserve the uncertainty structure associated with the stronger model better than conventional quantization?

---

# 19. Main Experimental Table

Example structure:

| Method | Avg bits | MMLU-Pro | ARC-C | GSM8K | ECE ↓ | Error AUROC ↑ | UQ correlation ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 16 | — | — | — | — | — | — |
| Uniform INT4 | 4.0 | | | | | | |
| GPTQ | 4.0 | | | | | | |
| AWQ | 4.0 | | | | | | |
| Random MP | 3.5 | | | | | | |
| Student sensitivity | 3.5 | | | | | | |
| Existing MPQ | 3.5 | | | | | | |
| Teacher confidence | 3.5 | | | | | | |
| Teacher disagreement | 3.5 | | | | | | |
| **Proposed** | **3.5** | | | | | | |

---

# 20. Primary Plot

The most important figure:

## Accuracy vs Compression

\[
x=\text{Average bits/weight}
\]

\[
y=\text{MMLU-Pro accuracy}
\]

Plot:

- uniform quantization,
- GPTQ/AWQ,
- student-only mixed precision,
- proposed method.

Desired result:

```text
          Proposed ●
                 /
 Student-only ●
             /
       AWQ ●
          /
 Uniform ●
────────────────────────
      Avg bits/weight
```

The proposed curve should dominate particularly around aggressive compression.

---

# 21. Second Key Plot

## Reliability vs Compression

\[
x=\text{Average bits/weight}
\]

\[
y=\text{Error-detection AUROC}
\]

Desired finding:

> Standard quantization can retain much of the task accuracy while rapidly degrading uncertainty quality, whereas teacher-aware compression preserves both.

This provides the uncertainty-focused contribution.

---

# 22. Diagnostic Analysis

Study which layers receive additional precision.

Questions:

- Do early layers remain aggressively quantized?
- Are attention layers more protected?
- Are MLP layers more sensitive?
- Do teacher-aware and student-only methods rank layers differently?

Calculate:

\[
\rho(
I_l^{student},
I_l^{teacher}
).
\]

Low/moderate correlation would indicate that the teacher provides genuinely different information.

---

# 23. Example-Level Analysis

Partition calibration/evaluation examples based on teacher/student behavior.

### Group A

Teacher correct + confident  
Student correct

### Group B

Teacher correct + confident  
Student wrong

### Group C

Teacher uncertain

### Group D

Teacher confident + wrong

The proposed method should especially preserve behavior on **Group B**.

This directly tests the interpretation:

> Preserve capabilities that exist in the teacher but are weak in the student.

---

# 24. Failure Analysis

Important failure modes to investigate:

### Teacher overconfidence

A confidently wrong teacher may assign high importance to undesirable behavior.

### Teacher/student correlated errors

Teacher confidence may provide little additional information when both models fail similarly.

### Domain shift

A bit allocation obtained from reasoning benchmarks may not generalize to unrelated tasks.

### Model-size gap

A teacher too similar in capability to the student may provide weak signals.

### Extremely aggressive quantization

At 2-bit or below, numerical distortion may dominate semantic allocation.

---

# 25. Possible Extensions

Not necessary for the initial submission.

## Extension A — Downstream Delegation

Take the compressed SLM and use:

\[
U_{S_Q}(x)
\]

to decide whether to call the teacher.

Then test whether teacher-aware compression preserves the quality of the escalation policy.

This connects:

\[
\text{Teacher uncertainty}
\rightarrow
\text{compression}
\rightarrow
\text{student uncertainty}
\rightarrow
\text{delegation}.
\]

---

## Extension B — Multiple Teachers

Test whether teacher signals are model-specific.

For example:

```text
Teacher A → compression policy A
Teacher B → compression policy B
```

Compare layer allocations.

---

## Extension C — Domain-Aware Compression

Use separate teacher-calibration distributions:

```text
math
science
general knowledge
```

and determine whether domain-specific compression policies emerge.

---

# 26. Minimum Viable Experiment

To obtain the first meaningful result:

## Models

```text
1 teacher
1 student (~1.5B–3B)
```

## Calibration

```text
500–1,000 examples
```

## Evaluation

```text
MMLU-Pro
ARC-Challenge
GSM8K
```

## Precision budgets

```text
4.0 bits
3.5 bits
3.0 bits
```

## Baselines

1. FP16 student
2. uniform quantization
3. GPTQ
4. AWQ
5. random mixed precision
6. student-only sensitivity
7. proposed teacher-aware sensitivity

## Ablations

1. teacher confidence only,
2. teacher–student disagreement only,
3. combined signal.

---

# 27. Success Criteria

The direction is promising if at least one of the following holds consistently.

### Outcome 1 — Better task preservation

At matched bit budget:

\[
Acc_{\text{ours}}
>
Acc_{\text{student sensitivity}}.
\]

### Outcome 2 — Better uncertainty preservation

At similar accuracy:

\[
AUROC_{\text{ours}}
>
AUROC_{\text{AWQ/GPTQ}}.
\]

### Outcome 3 — Better aggressive-compression robustness

The advantage increases as:

\[
4\text{ bit}
\rightarrow
3.5\text{ bit}
\rightarrow
3\text{ bit}.
\]

This would be especially compelling because the teacher signal helps determine **what should survive when capacity becomes scarce**.

---

# 28. Intended Contribution

The strongest framing is not:

> Teacher uncertainty improves quantization.

Instead:

> **We introduce a teacher-aware principle for mixed-precision compression in which precision is allocated according to the semantic value of preserving model behavior, rather than only numerical quantization sensitivity. Large-model uncertainty and teacher–student disagreement identify capabilities that the compressed student should preferentially retain.**

The paper then evaluates whether this principle improves:

1. predictive accuracy,
2. reliability and uncertainty preservation,
3. compression efficiency,

under fixed deployment budgets.