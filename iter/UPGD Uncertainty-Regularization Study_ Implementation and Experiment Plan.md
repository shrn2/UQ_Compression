# UPGD Uncertainty-Regularization Study: Implementation and Experiment Plan

## 1. Objective

The goal of this study is to determine **how explicitly preserving teacher uncertainty affects structured FFN pruning**, beyond the benefit obtained from ordinary KL-based distillation.

The current results indicate that joint learned gate optimization is beneficial, but the additional uncertainty-preservation terms are not uniformly beneficial across model scales and datasets. For example, on 1.5B HellaSwag, KL-only improves over one-shot pruning by +3.2 points and full UPGD by +4.4 points, whereas on 3B HellaSwag KL-only is approximately neutral and full UPGD is −2.6 points.

The proposed study therefore treats uncertainty preservation as a **controllable regularization strength**, rather than assuming that the current fixed full-UPGD objective is universally optimal.

---

# 2. Research Questions

### RQ1 — Does learned gate optimization improve structured pruning?

Compare:

- Random 50% pruning
- One-shot activation-importance pruning
- Learned gates using KL-only distillation

This isolates whether optimizing the mask through the frozen network improves over heuristic channel ranking.

---

### RQ2 — Does explicit uncertainty preservation provide benefits beyond KL-only distillation?

Compare:

\[
\mathcal{L}_{KL}
\]

against

\[
\mathcal{L}_{KL}+\lambda\mathcal{L}_{UQ}.
\]

The main question is whether uncertainty-aware objectives improve:

- uncertainty fidelity to the dense model;
- calibration/selective prediction;
- predictive accuracy;
- or the trade-off between these quantities.

---

### RQ3 — Is there an accuracy–uncertainty Pareto frontier?

Determine whether intermediate uncertainty weights preserve substantially more dense-model uncertainty while retaining most of the predictive performance of KL-only pruning.

---

### RQ4 — Are the effects stable across model scale and task?

Evaluate the same experimental protocol on:

- 1.5B BF16 model
- 3B NF4 model

and:

- HellaSwag
- ARC-Challenge
- MMLU auxiliary

The goal is **not** to claim model-invariant superiority unless the results support it. Existing experiments already indicate substantial task/model dependence.

---

# 3. Models

## 3.1 1.5B model

Use the existing 1.5B BF16 setup:

- 28 transformer layers
- 8,960 FFN channels per layer
- BF16 weights
- 50% FFN-channel retention
- frozen model weights
- learned gate logits
- physical channel export available

The existing implementation uses 600 gate-optimization steps.

---

## 3.2 3B model

Use the existing 3B NF4 setup:

- 36 transformer layers
- 11,008 FFN channels per layer
- 4-bit NF4 quantization
- 50% FFN-channel retention
- frozen model weights
- learned gate logits
- logical hook masks for quality evaluation

The current implementation uses 300 optimization steps, which creates a confound with the 600-step 1.5B experiment.

### Step-budget control

The primary study should use a **matched number of optimization steps across models**.

Preferred:

```text
1.5B: 600 steps
3B:   600 steps
```

If 600 steps for the 3B model is infeasible:

```text
1.5B: 300 steps
3B:   300 steps
```

and separately run a 1.5B 600-step reference to quantify the effect of the shorter schedule.

Do not interpret differences between a 1.5B/600-step run and a 3B/300-step run as evidence of scaling behavior.

---

# 4. Datasets

Use three task families:

| Dataset | Role |
|---|---|
| HellaSwag | commonsense completion |
| ARC-Challenge | scientific reasoning |
| MMLU auxiliary | broad knowledge/reasoning |

Existing evaluation sizes can be retained:

- HellaSwag: 500 examples
- ARC-Challenge: 299 examples
- MMLU auxiliary: 600 examples

The final evaluation examples must **not appear in gate optimization or hyperparameter selection**.

---

# 5. Calibration Dataset

Rather than learning a separate pruning mask for every benchmark, the primary experiment should learn a **single multi-task mask per model, seed, and \(\lambda\)**.

This makes the experiment considerably stronger because the resulting pruning mask is not tuned specifically to the test benchmark.

## 5.1 Multi-task calibration mixture

Construct a calibration set containing approximately:

```text
256 HellaSwag examples
256 ARC examples
256 MMLU auxiliary examples
----------------------------
768 total calibration examples
```

Use labeled training/development examples that do not overlap with final evaluation examples.

---

## 5.2 Entropy balancing

Run the dense model over the calibration pool and compute predictive entropy:

\[
H(p_d)
=
-\sum_c p_d(c)\log p_d(c).
\]

Within each dataset, divide examples into:

- low dense entropy;
- middle dense entropy;
- high dense entropy.

Sample approximately equally from each stratum.

This retains the uncertainty-stratified design of the current UPGD implementation, which currently draws examples across three dense-entropy strata.

Calibration batches should therefore be balanced approximately across both:

```text
dataset
×
uncertainty stratum
```

rather than being dominated by one benchmark or one confidence region.

---

# 6. UPGD Objective

The existing full objective is:

\[
\mathcal L
=
KL(p_d\Vert p_m)
+
(H_d-H_m)^2
+
0.5(M_d-M_m)^2
+
0.5L_{\mathrm{worst}}.
\]



Rewrite this as:

\[
\boxed{
\mathcal L(\lambda)
=
KL(p_d\Vert p_m)
+
\lambda\mathcal L_{UQ}
}
\]

where

\[
\mathcal L_{UQ}
=
(H_d-H_m)^2
+
0.5(M_d-M_m)^2
+
0.5L_{\mathrm{worst}}.
\]

This makes uncertainty preservation an explicit experimental variable.

---

# 7. Primary \(\lambda\) Sweep

Evaluate:

\[
\boxed{
\lambda\in
\{0,\;0.25,\;0.5,\;1.0\}
}
\]

Interpretation:

| λ | Objective |
|---:|---|
| 0 | KL-only learned pruning |
| 0.25 | weak UQ regularization |
| 0.50 | moderate UQ regularization |
| 1.00 | current full UPGD |

Thus:

\[
\lambda=0
\]

is the critical control for learned-mask optimization without explicit uncertainty preservation.

And:

\[
\lambda=1
\]

recovers the existing full UPGD formulation.

Do **not** modify the entropy, margin, or worst-stratum coefficients independently during this experiment. The purpose of this study is to vary one controlled quantity.

---

# 8. Fixed Training Configuration

Everything except \(\lambda\) should remain fixed.

Use the current UPGD mechanics:

- model weights frozen;
- one learnable gate logit per FFN channel;
- exact 50% retained-channel budget;
- stochastic mask exploration;
- annealed logistic noise;
- exact top-\(k\) selection;
- straight-through gradient estimator;
- deterministic retained-channel indices after optimization.

The current implementation anneals temperature from 0.5 to 0.1 and mask noise from 0.5 to 0.05.

Freeze:

```yaml
retention_ratio: 0.50

temperature:
  start: 0.50
  end: 0.10

mask_noise:
  start: 0.50
  end: 0.05

entropy_weight_base: 1.0
margin_weight_base: 0.5
robust_weight_base: 0.5

lambda_uq:
  - 0.0
  - 0.25
  - 0.5
  - 1.0
```

The effective uncertainty coefficients are multiplied by `lambda_uq`.

---

# 9. Seeds

Run **three independent learned-mask seeds** for every \(\lambda\).

The seed must control:

- gate initialization;
- stochastic mask noise;
- calibration-example ordering;
- any other stochastic training component.

Use:

```text
seed = 1
seed = 2
seed = 3
```

The random-mask experiments already demonstrate that arbitrary 50% masks can perform substantially worse than learned/importance-based masks. On HellaSwag, random masks average below one-shot for both model scales.

The important new reproducibility test is therefore **variation between independently learned masks**.

---

# 10. Core Run Matrix

The primary learned-mask experiment becomes:

\[
2\text{ models}
\times
4\lambda
\times
3\text{ seeds}
=
24
\]

learned masks.

Each mask is evaluated independently on:

\[
3\text{ datasets}.
\]

Therefore:

\[
24\times3=72
\]

learned-mask evaluation slices.

This is preferable to independently optimizing a different mask for every test dataset because it directly evaluates task transfer.

---

# 11. Baseline Conditions

Every model/dataset evaluation should contain:

### Dense

Unpruned reference model.

### Random-50%

Three independently sampled exact-50% channel masks.

Purpose:

> establish a pruning sanity floor.

### One-shot

Existing activation-importance structured pruning baseline.

Purpose:

> determine whether optimization is better than simple ranking.

### KL-only

\[
\lambda=0.
\]

Purpose:

> isolate joint learned pruning.

### UQ-regularized UPGD

\[
\lambda=\{0.25,0.5,1.0\}.
\]

Purpose:

> isolate the contribution of explicit uncertainty preservation.

The conceptual comparison is therefore:

```text
Dense
   ↓
Random pruning
   ↓
One-shot importance pruning
   ↓
KL learned gates
   ↓
KL + controlled UQ regularization
```

---

# 12. Quality-Evaluation Protocol

For each final evaluation example, save:

```json
{
  "example_id": "...",
  "gold_label": "...",
  "dense_prediction": "...",
  "masked_prediction": "...",
  "dense_probabilities": [],
  "masked_probabilities": [],
  "dense_entropy": 0.0,
  "masked_entropy": 0.0,
  "dense_margin": 0.0,
  "masked_margin": 0.0,
  "correct_dense": true,
  "correct_masked": true
}
```

Saving per-example outputs is essential because all important comparisons should be paired.

---

# 13. Primary Metrics

## 13.1 Predictive quality

Report:

\[
\text{Accuracy}
\]

and:

\[
\Delta Acc
=
Acc_{\mathrm{masked}}
-
Acc_{\mathrm{dense}}.
\]

Also report accuracy relative to one-shot:

\[
Acc_{\mathrm{method}}
-
Acc_{\mathrm{one-shot}}.
\]

---

## 13.2 Dense-model fidelity

### KL divergence

\[
KL(p_d\Vert p_m)
\]

### Dense agreement

Fraction of examples where:

\[
\arg\max p_d
=
\arg\max p_m.
\]

---

# 14. Uncertainty-Preservation Metrics

The main uncertainty-fidelity measurements should be:

### Entropy MAE

\[
\frac{1}{N}
\sum_i
|H_d^{(i)}-H_m^{(i)}|.
\]

This should be one of the primary metrics because it directly corresponds to the proposed mechanism.

---

### Entropy rank correlation

Spearman correlation:

\[
\rho(H_d,H_m).
\]

This measures whether examples considered uncertain by the dense model remain relatively uncertain after pruning.

---

### Margin MAE

\[
\frac{1}{N}
\sum_i
|M_d^{(i)}-M_m^{(i)}|.
\]

---

### ECE

Report expected calibration error as a secondary metric.

---

### AURC

Report area under the risk-coverage curve to evaluate selective prediction.

The existing report already measures KL, dense agreement, entropy correlation, AURC, and ECE, so the new experiment should preserve these metrics for continuity.

---

# 15. Uncertainty-Stratified Evaluation

For every model/dataset pair, divide the final evaluation examples according to **dense-model entropy**:

```text
Low uncertainty
Middle uncertainty
High uncertainty
```

For each stratum report:

- accuracy;
- accuracy difference vs one-shot;
- accuracy difference vs KL-only;
- entropy MAE;
- dense agreement;
- AURC where meaningful.

Do not claim that the current worst-stratum objective reliably improves high-uncertainty examples unless the new results demonstrate it. Existing results show that uncertainty-stratum behavior is unstable across slices.

---

# 16. Primary Statistical Comparisons

The most important comparison is no longer only:

\[
UPGD-OneShot.
\]

For each \(\lambda>0\), compute:

\[
\boxed{
UPGD_\lambda
-
UPGD_{\lambda=0}
}
\]

on exactly the same evaluation examples.

This directly answers:

> Does uncertainty-aware optimization improve anything beyond KL-based learned pruning?

Use paired bootstrap confidence intervals or an equivalent paired resampling method.

Report:

```text
mean difference
95% CI
```

for:

- accuracy;
- entropy MAE;
- margin MAE;
- KL;
- AURC;
- ECE.

---

# 17. Seed Aggregation

For each condition report:

\[
\text{mean across 3 learned-mask seeds}
\]

and:

\[
\text{standard deviation across seeds}.
\]

The main paper table should therefore contain entries such as:

| Model | Dataset | λ | Accuracy | Entropy MAE | KL | AURC |
|---|---|---:|---:|---:|---:|---:|
| 1.5B | HellaSwag | 0 | mean ± SD | ... | ... | ... |
| 1.5B | HellaSwag | .25 | mean ± SD | ... | ... | ... |
| 1.5B | HellaSwag | .50 | mean ± SD | ... | ... | ... |
| 1.5B | HellaSwag | 1.0 | mean ± SD | ... | ... | ... |

Repeat for all six model/dataset combinations.

---

# 18. Primary Pareto Analysis

The key visualization should show the trade-off between predictive quality and uncertainty fidelity.

For every learned mask, plot:

\[
x=
Acc_{\mathrm{dense}}
-
Acc_{\mathrm{masked}}
\]

against:

\[
y=
\text{Entropy MAE}.
\]

Lower is better on both axes.

Include:

```text
one-shot
λ = 0
λ = 0.25
λ = 0.5
λ = 1.0
```

with error bars across seeds.

A second version can use:

\[
x=\text{Accuracy}
\]

and:

\[
y=\rho(H_d,H_m).
\]

The central question is whether an intermediate \(\lambda\) lies on a better Pareto frontier than both:

- KL-only;
- full UPGD.

---

# 19. Hyperparameter Selection

Do not select \(\lambda\) directly from the final evaluation results.

Create a held-out validation mixture containing examples from:

- HellaSwag;
- ARC;
- MMLU auxiliary.

Select the preferred \(\lambda\) using the validation set.

A simple predeclared rule is:

> Select the largest \(\lambda\) that meaningfully improves uncertainty fidelity while reducing aggregate validation accuracy by no more than 1 percentage point relative to KL-only.

Then report its performance on the untouched final evaluation sets.

The full sweep should still be reported for scientific analysis.

---

# 20. Model-Scale Comparison

Do not primarily compare:

```text
1.5B accuracy
vs
3B accuracy
```

because the dense models themselves differ substantially.

Instead compare **relative effects**:

\[
\Delta Acc_\lambda
=
Acc_\lambda-Acc_{KL}.
\]

Likewise:

\[
\Delta EntropyMAE_\lambda
=
EntropyMAE_\lambda
-
EntropyMAE_{KL}.
\]

Then ask whether uncertainty regularization has similar directional effects at the two model scales.

---

# 21. Quality vs Deployment Evaluation

For the quality study, use equivalent logical masks whenever possible so that pruning semantics remain comparable.

Separately, evaluate physical deployment.

## 21.1 1.5B physical model

The current 1.5B pipeline can physically remove channels.

For the selected mask report:

- original parameters;
- remaining parameters;
- percentage parameter reduction;
- model size;
- peak VRAM;
- tokens/second;
- latency/token;
- FLOPs/token if available.

Verify that physical export reproduces the logical-mask outputs within numerical tolerance.

---

## 21.2 3B NF4

Keep logical masking for the main quality experiment unless packed NF4 physical pruning is implemented reliably.

Do not claim actual 3B inference acceleration from logical-hook masking.

Report this limitation explicitly.

---

# 22. Implementation Structure

Suggested experiment structure:

```text
experiments/
│
├── configs/
│   ├── model_1p5b.yaml
│   ├── model_3b_nf4.yaml
│   ├── lambda_000.yaml
│   ├── lambda_025.yaml
│   ├── lambda_050.yaml
│   └── lambda_100.yaml
│
├── calibration/
│   ├── build_multitask_calibration.py
│   ├── entropy_stratify.py
│   └── calibration_manifest.json
│
├── pruning/
│   ├── train_gates.py
│   ├── build_mask.py
│   ├── random_mask.py
│   └── one_shot_mask.py
│
├── evaluation/
│   ├── evaluate_hellaswag.py
│   ├── evaluate_arc.py
│   ├── evaluate_mmlu_aux.py
│   └── uncertainty_metrics.py
│
├── analysis/
│   ├── paired_bootstrap.py
│   ├── aggregate_seeds.py
│   ├── pareto_analysis.py
│   └── generate_tables.py
│
└── outputs/
    ├── masks/
    ├── predictions/
    ├── metrics/
    ├── statistics/
    └── figures/
```

---

# 23. Run Manifest

Every experiment should have a row in a machine-readable manifest:

```csv
run_id,model,lambda,seed,steps,retention_ratio,calibration_set,mask_path,status
```

Example:

```text
1p5b_l025_s1,1.5B,0.25,1,600,0.50,multitask_v1,...,complete
```

Record:

- git commit;
- model checkpoint;
- dataset revision;
- random seed;
- hyperparameters;
- hardware;
- runtime;
- peak memory.

This will make the final paper substantially easier to reproduce.

---

# 24. Required Result Tables

## Table 1 — Main pruning results

```text
Dense
Random
One-shot
KL-only
λ=.25
λ=.50
λ=1.0
```

Report accuracy across:

```text
1.5B × {HellaSwag, ARC, MMLU-aux}
3B   × {HellaSwag, ARC, MMLU-aux}
```

---

## Table 2 — Uncertainty fidelity

Report:

- KL
- entropy MAE
- entropy rho
- margin MAE
- ECE
- AURC

---

## Table 3 — Direct UQ contribution

For:

\[
\lambda=.25,.5,1
\]

report paired differences against:

\[
\lambda=0.
\]

This is the most important ablation table for the scientific claim.

---

## Table 4 — Deployment

For 1.5B:

```text
Dense
One-shot physical
Selected UPGD physical
```

Report parameter count, storage, VRAM, latency, and throughput.

---

# 25. Required Figures

### Figure 1 — Method

```text
Dense model
      │
      ├── p_dense
      ├── entropy_dense
      └── margin_dense
             │
             ▼
    UPGD gate optimization
             │
   KL + λ × UQ preservation
             │
             ▼
     exact 50% FFN mask
             │
             ▼
       compressed model
```

---

### Figure 2 — Accuracy vs λ

For each model/dataset:

\[
Accuracy(\lambda).
\]

---

### Figure 3 — Uncertainty preservation vs λ

For example:

\[
EntropyMAE(\lambda).
\]

---

### Figure 4 — Pareto frontier

\[
\text{accuracy loss}
\quad\text{vs}\quad
\text{entropy deviation}.
\]

This should be the main result figure if an intermediate \(\lambda\) produces a useful trade-off.

---

# 26. Interpretation Rules

The conclusions should be determined by the results rather than requiring full UPGD to win.

### Outcome A — Intermediate λ is best

If:

\[
\lambda=.25\text{ or }.5
\]

preserves uncertainty substantially better than KL-only with little or no accuracy degradation, the main conclusion becomes:

> Moderate uncertainty regularization produces a better accuracy–uncertainty trade-off during structured pruning than either KL-only distillation or strong fixed uncertainty regularization.

This is the strongest likely outcome.

---

### Outcome B — Full UPGD wins consistently

Then the original UPGD claim becomes substantially stronger.

---

### Outcome C — KL-only dominates accuracy, UQ improves uncertainty only

This is still useful.

The conclusion becomes:

> Explicit uncertainty preservation provides controllable uncertainty fidelity at an accuracy cost.

The method is then best understood as a **trustworthiness/behavior-preservation control** rather than an accuracy-improving pruning method.

---

### Outcome D — λ effects remain highly task-dependent

Do not hide this result.

The paper becomes:

> Uncertainty-aware structured pruning exhibits task-dependent trade-offs; fixed global uncertainty penalties are insufficient, motivating adaptive uncertainty-preservation objectives.

For a LIGHT workshop paper, this remains a coherent empirical contribution.

---

# 27. Experiment Priority

## Phase 1 — Mandatory

1. Build disjoint multi-task calibration mixture.
2. Match optimization budgets across 1.5B and 3B.
3. Implement `lambda_uq`.
4. Run:
   - λ=0
   - λ=.25
   - λ=.5
   - λ=1
5. Use 3 learned-mask seeds.
6. Evaluate every learned mask on:
   - HellaSwag
   - ARC-Challenge
   - MMLU auxiliary.
7. Compute direct paired comparisons against KL-only.
8. Generate Pareto analysis.

---

## Phase 2 — Mandatory for LIGHT-quality submission

9. Evaluate dense, random, and one-shot anchors.
10. Measure physical 1.5B deployment efficiency.
11. Produce aggregate tables and confidence intervals.
12. Freeze the experimental protocol.

---

## Phase 3 — Only if time remains

13. Add a stronger external structured-pruning baseline such as FLAP.
14. Test one additional pruning ratio, e.g. 70% retention or 30% retention.
15. Investigate adaptive or example-conditioned uncertainty weighting.

These should not delay the core \(\lambda\)-sweep.

---

# 28. Target Paper Claim

The study should initially be framed around the conservative claim:

> **Joint gate distillation enables exact-budget structured pruning while uncertainty regularization provides explicit control over how much of the dense model's uncertainty behavior is preserved. Across 1.5B and 3B models and three reasoning benchmarks, we study the resulting predictive-accuracy versus uncertainty-fidelity trade-off.**

Do not claim that uncertainty regularization universally improves accuracy unless the new experiments demonstrate it.

The current evidence already shows that joint learned pruning is useful while the effects of additional uncertainty terms vary by model and dataset.

---

# 29. Minimal Completion Criterion for the LIGHT Submission

The experiment is ready to freeze for writing once the following are complete:

- both 1.5B and 3B evaluated;
- HellaSwag, ARC-Challenge, and MMLU auxiliary evaluated;
- λ = 0, .25, .5, 1 completed;
- three learned-mask seeds completed;
- one-shot and random controls completed;
- direct λ-vs-KL paired statistics completed;
- uncertainty-fidelity metrics completed;
- accuracy–uncertainty Pareto plot completed;
- 1.5B physical deployment numbers completed;
- all runs reproducible from saved configs/manifests.

At that point, additional experiments should be added only if they answer a specific reviewer-facing question.