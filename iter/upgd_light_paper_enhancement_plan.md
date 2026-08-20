# UPGD Paper Enhancement — Implementation and Experiment Plan

## 0. Goal

Upgrade the current UPGD study into a clean, scale-controlled **uncertainty-aware structured compression** paper for the LIGHT workshop.

The enhanced paper should answer:

1. **Does structured pruning distort uncertainty differently from task accuracy?**
2. **Does uncertainty-aware gate distillation reduce that distortion beyond KL-only gate learning?**
3. **How does the trade-off change as pruning becomes more aggressive?**
4. **Does the effect transfer across model scale?**
5. **Do the learned masks produce real deployment benefits after physical pruning?**

Target story:

> Structured pruning induces different degradation trajectories for predictive accuracy, dense-teacher uncertainty fidelity, and downstream calibration. UPGD explicitly controls this trade-off through uncertainty-aware gate distillation under exact structured pruning budgets.

---

# 1. Freeze the Current Iteration 16 State

Before modifying the codebase, preserve the current experiment exactly.

Current evidence already includes:

- 1.5B BF16 experiments.
- 3B NF4 experiments.
- exact 50% FFN-channel retention.
- KL-only versus uncertainty-regularized gate learning.
- `lambda_uq ∈ {0, 0.25, 0.5, 1.0}`.
- three learned-mask seeds in the latest lambda study.
- HellaSwag, ARC-Challenge, and MMLU auxiliary.
- teacher-fidelity metrics: KL, agreement, entropy MAE/rank correlation, margin MAE.
- downstream metrics: accuracy, ECE, AURC.
- prior random and one-shot pruning controls.
- physical export for 1.5B.
- logical masking for the existing 3B NF4 model.

Create an immutable snapshot:

```text
git tag upgd-iter16-frozen
```

Archive:

```text
artifacts/iter16_frozen/
├── configs/
├── masks/
├── predictions/
├── metrics/
├── manifests/
├── logs/
└── report/
```

Do not overwrite Iteration 16 outputs.

---

# 2. Definitive Study Design

## Models

Primary:

```text
1.5B BF16
3B BF16
```

Optional confirmation:

```text
7B BF16
```

## Datasets

```text
HellaSwag
ARC-Challenge
MMLU auxiliary
```

## FFN retention ratios

```text
75%
50%
25%
```

Equivalent pruning severities:

```text
25% FFN channels pruned
50% FFN channels pruned
75% FFN channels pruned
```

## Uncertainty regularization

Primary grid:

```text
lambda_uq = 0.0   # KL-only
lambda_uq = 0.5   # moderate UQ preservation
lambda_uq = 1.0   # full UPGD
```

Keep `lambda_uq = 0.25` as an exploratory/legacy point unless extra compute is available.

## Seeds

```text
1
2
3
```

## Gate optimization budget

Use:

```text
600 steps
```

for both 1.5B and 3B.

The definitive paper should not compare a 600-step 1.5B condition against a 300-step 3B condition.

---

# 3. Refactor Experiment Configuration

Expose model, pruning ratio, objective, and seed as independent configuration dimensions.

Recommended schema:

```yaml
model:
  name: ...
  dtype: bfloat16

pruning:
  retention_ratio: 0.50
  mask_mode: physical
  structured_unit: ffn_channel

training:
  steps: 600
  seed: 1

loss:
  lambda_uq: 0.5
  entropy_weight: 1.0
  margin_weight: 0.5
  robust_weight: 0.5
```

Do not hard-code:

- model-specific step counts;
- 50% retention;
- NF4 for the 3B model;
- fixed lambda;
- physical/logical mask behavior by model name.

Recommended configuration tree:

```text
configs/
├── models/
│   ├── model_1p5b_bf16.yaml
│   ├── model_3b_bf16.yaml
│   └── model_7b_bf16.yaml
├── pruning/
│   ├── retain_075.yaml
│   ├── retain_050.yaml
│   └── retain_025.yaml
├── objectives/
│   ├── kl_only.yaml
│   ├── uq_050.yaml
│   └── uq_100.yaml
├── datasets/
│   └── multitask_calibration.yaml
└── experiments/
    └── definitive_grid.yaml
```

---

# 4. Move the 3B Model to BF16

This is the highest-priority technical change.

## 4.1 Dense BF16 model path

Implement a non-quantized 3B path:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

Remove from the definitive path:

- NF4 loading;
- bitsandbytes `Linear4bit`;
- quantization-specific masking hooks.

## 4.2 Establish a new 3B BF16 dense reference

Evaluate the unpruned 3B BF16 model on:

```text
HellaSwag
ARC-Challenge
MMLU auxiliary
```

Store:

```text
dense_3b_bf16/
├── predictions.jsonl
├── probabilities.npz
└── metrics.json
```

Do not require BF16 predictions to reproduce the prior NF4 outputs exactly. BF16 becomes the new reference for the definitive study.

---

# 5. Generalize Physical FFN Pruning

Generalize the physical export path from 1.5B to 3B BF16.

## 5.1 Gated FFN pruning

For a SwiGLU-like block:

```text
hidden
  ├── gate_proj ─┐
  └── up_proj   ─┴─ elementwise product
                      │
                  down_proj
                      │
                    hidden
```

For retained channel indices `I`:

```python
gate_proj.weight = gate_proj.weight[I, :]
up_proj.weight   = up_proj.weight[I, :]
down_proj.weight = down_proj.weight[:, I]
```

Prune corresponding biases consistently if present.

## 5.2 Invariants

For every layer:

```text
retained gate_proj rows
=
retained up_proj rows
=
retained down_proj columns
```

and:

```text
new_intermediate_size
=
round(original_intermediate_size * retention_ratio)
```

## 5.3 Logical-versus-physical equivalence

For every exported model:

1. run deterministic logical masking;
2. export the physically pruned model with the same retained indices;
3. run both on identical examples;
4. compare logits.

Acceptance criterion:

```text
max absolute logit difference <= BF16 numerical tolerance
```

Add:

```text
test_physical_export_matches_logical_mask
```

---

# 6. Build a Clean Multi-Task Calibration Protocol

The paper should not learn a different mask for each final benchmark.

## 6.1 Calibration pool

Use training-side examples from:

```text
HellaSwag
ARC-Challenge
MMLU auxiliary
```

Suggested balanced pool:

```text
256 HellaSwag
256 ARC
256 MMLU auxiliary
-------------------
768 examples
```

If ARC split constraints make 256 inconvenient, use the largest clean balanced count that preserves disjoint final evaluation.

## 6.2 Dense-entropy stratification

For every candidate example:

```python
H_dense = -sum(p * log(p))
```

Within each task, divide examples into:

```text
low entropy
middle entropy
high entropy
```

Construct batches balanced approximately across:

```text
dataset × entropy stratum
```

That gives nine cells:

```text
HellaSwag × {Low, Mid, High}
ARC       × {Low, Mid, High}
MMLU      × {Low, Mid, High}
```

---

# 7. Create a Three-Way Data Split

The enhanced study needs distinct data for:

1. gate optimization;
2. lambda selection;
3. final evaluation.

Use:

```text
Calibration/train set
    -> optimize gate logits

Selection/validation set
    -> select lambda_uq

Final evaluation set
    -> paper results
```

Do not choose lambda using final test/evaluation results.

---

# 8. Predeclare Lambda Selection

For each model and retention ratio, choose lambda using the held-out validation mixture.

Recommended rule:

> Choose the largest lambda that improves aggregate teacher-uncertainty fidelity while reducing validation accuracy by no more than 1 percentage point relative to KL-only.

Operational form:

```python
eligible = accuracy(lambda) >= accuracy(lambda=0) - 0.01
selected = eligible lambda with minimum entropy_mae
```

Use entropy MAE as the primary teacher-uncertainty fidelity metric; margin MAE is secondary.

This selected configuration becomes the **Selected UPGD** row in the main paper table.

The complete lambda sweep remains an ablation.

---

# 9. Definitive Learned-Mask Grid

Run:

```text
2 models
× 3 retention ratios
× 3 lambda values
× 3 seeds
=
54 learned masks
```

Grid:

```text
model ∈ {1.5B BF16, 3B BF16}

retention ∈ {0.75, 0.50, 0.25}

lambda_uq ∈ {0.0, 0.5, 1.0}

seed ∈ {1, 2, 3}
```

Every run uses:

```text
600 gate-optimization steps
```

## 9.1 Objective

Use:

\[
L(\lambda)
=
KL(p_d\Vert p_m)
+
\lambda L_{UQ}
\]

with:

\[
L_{UQ}
=
(H_d-H_m)^2
+
0.5(M_d-M_m)^2
+
0.5L_{\mathrm{worst}}
\]

Interpretation:

```text
lambda=0
```

is a KL-only **learned-gate control**, not one-shot pruning.

## 9.2 Match all other settings

For matched lambda runs, keep fixed:

```text
calibration examples
example ordering
optimizer
learning rate
step count
temperature schedule
mask-noise schedule
pruning budget
```

---

# 10. Baselines

The paper should compare heuristic pruning, learned KL pruning, and uncertainty-aware learned pruning.

## 10.1 Dense

For each model:

```text
100% FFN retention
```

## 10.2 Random structured pruning

For every model and retention ratio:

```text
3 random masks
```

Total:

```text
2 × 3 × 3 = 18 random masks
```

## 10.3 One-shot activation pruning

Run:

```text
2 models × 3 retention ratios = 6 one-shot masks
```

Use the same calibration pool when the baseline requires activations.

## 10.4 FLAP

Use FLAP as the principal strong external structured-pruning baseline.

Run:

```text
2 models × 3 retention ratios = 6 conditions
```

Match:

- model checkpoint;
- retention ratio;
- evaluation set;
- calibration data when applicable.

Do not add recovery fine-tuning to only one method unless reported as a separate condition.

## 10.5 Optional secondary baseline

Only if implementation is simple:

```text
structured Wanda
```

or another closely matched structured baseline.

Do not delay the core experiment for many weakly matched baselines.

---

# 11. Final Evaluation

Evaluate every learned mask on:

```text
HellaSwag
ARC-Challenge
MMLU auxiliary
```

Thus:

```text
54 learned masks × 3 tasks = 162 learned-mask evaluation slices
```

## 11.1 Save per-example outputs

For each method:

```json
{
  "example_id": "...",
  "gold": 0,
  "dense_prediction": 0,
  "masked_prediction": 1,
  "dense_probabilities": [],
  "masked_probabilities": [],
  "dense_entropy": 0.0,
  "masked_entropy": 0.0,
  "dense_margin": 0.0,
  "masked_margin": 0.0,
  "dense_correct": true,
  "masked_correct": false
}
```

This is necessary for paired statistical comparisons.

---

# 12. Metric Design

Separate the metrics into three distinct concepts.

## 12.1 Task utility

Primary:

```text
accuracy
```

Also report:

```text
accuracy drop     = pruned_accuracy - dense_accuracy
accuracy retained = pruned_accuracy / dense_accuracy
```

## 12.2 Dense-teacher predictive fidelity

### KL

\[
KL(p_d\Vert p_m)
\]

### Dense agreement

\[
P(\arg\max p_d = \arg\max p_m)
\]

## 12.3 Dense-teacher uncertainty fidelity

Primary:

### Entropy MAE

\[
\frac{1}{N}\sum_i |H_d^{(i)}-H_m^{(i)}|
\]

Secondary:

### Entropy rank correlation

\[
\rho(H_d,H_m)
\]

### Margin MAE

\[
\frac{1}{N}\sum_i |M_d^{(i)}-M_m^{(i)}|
\]

Use the phrase:

```text
teacher-uncertainty fidelity
```

rather than equating entropy matching with calibration.

## 12.4 Ground-truth calibration/selective prediction

Report separately:

```text
ECE
AURC
```

Add if straightforward:

```text
NLL
Brier score
```

A key analysis should explicitly test whether better teacher-uncertainty fidelity corresponds to better or worse ground-truth calibration.

---

# 13. Statistical Analysis

## 13.1 Across-seed aggregation

For every learned condition:

```text
mean ± standard deviation across 3 seeds
```

## 13.2 Paired example-level bootstrap

Critical comparisons:

```text
KL-only vs one-shot
UPGD lambda=.5 vs KL-only
UPGD lambda=1 vs KL-only
Selected UPGD vs one-shot
Selected UPGD vs FLAP
```

Report:

```text
mean difference
95% bootstrap CI
```

for:

- accuracy;
- KL;
- entropy MAE;
- margin MAE;
- ECE;
- AURC.

## 13.3 Compression-severity trends

For each method:

```text
75% -> 50% -> 25% retention
```

Analyze changes in:

```text
accuracy
KL
entropy MAE
ECE
AURC
```

Primary question:

> Does uncertainty fidelity deteriorate at a different rate than predictive accuracy as pruning becomes more aggressive?

---

# 14. Pareto Analysis

## 14.1 Accuracy vs teacher-uncertainty fidelity

For every model/task:

```text
x = accuracy loss from dense
y = entropy MAE from dense
```

Lower-left is preferable.

Show:

```text
one-shot
FLAP
KL-only
lambda=.5
lambda=1
```

and encode retention ratio.

## 14.2 Compression vs accuracy

```text
x = actual retained parameters / FFN retention
y = accuracy
```

## 14.3 Compression vs uncertainty fidelity

```text
x = actual retained parameters / FFN retention
y = entropy MAE
```

This is one of the most important new figures because it tests whether uncertainty degrades differently from accuracy.

---

# 15. Physical Deployment Benchmark

This is mandatory for the enhanced LIGHT story.

Benchmark physically exported:

```text
1.5B BF16
3B BF16
```

at:

```text
75%
50%
25% FFN retention
```

Primary methods:

```text
Dense
One-shot
Selected UPGD
```

Add:

```text
FLAP
KL-only
```

if feasible.

## 15.1 Actual model compression

Report:

```text
total parameter count
FFN parameter count
checkpoint size
percentage total parameter reduction
```

Do not describe 50% FFN retention as 50% whole-model compression.

## 15.2 GPU memory

Measure:

```text
peak allocated memory
peak reserved memory
steady-state inference memory
```

Useful APIs:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

## 15.3 Throughput and latency

Use fixed benchmark conditions.

Suggested prompt lengths:

```text
128 tokens
512 tokens
```

Generated length:

```text
128 new tokens
```

Batch sizes:

```text
1
4
```

Procedure:

1. warm up;
2. synchronize CUDA;
3. run repeated measurements;
4. report median plus dispersion.

Report:

```text
tokens/s
end-to-end latency
time to first token, if available
decode latency/token
```

## 15.4 Benchmark controls

Keep identical:

```text
GPU
CUDA
PyTorch
attention backend
batch size
prompt length
generation length
sampling settings
```

---

# 16. Optional 7B Scaling Extension

Only start this after the 1.5B/3B definitive study is complete.

Purpose:

> Test whether the uncertainty/compression trade-off continues to change with model scale.

Use a reduced grid:

```text
retention ∈ {0.50, 0.25}
lambda_uq ∈ {0, 1}
seed ∈ {1, 2, 3}
```

Total:

```text
12 gate-training runs
```

Evaluate on the same three datasets.

If 7B requires substantial architecture-specific work, skip it. A clean 1.5B/3B BF16 study has higher value.

---

# 17. Run Management

## 17.1 Manifest

Every run should have:

```csv
run_id,model,dtype,retention,lambda_uq,seed,steps,calibration_manifest,mask_path,status
```

Example:

```text
3b_bf16_r050_l050_s2
```

## 17.2 Save metadata

Record:

```text
git commit
model revision
dataset revision
GPU
CUDA version
PyTorch version
seed
training steps
learning rate
retention
lambda_uq
mask checksum
runtime
peak memory
```

## 17.3 Output structure

```text
outputs/
├── dense/
├── random/
├── one_shot/
├── flap/
├── upgd/
│   ├── 1p5b/
│   ├── 3b/
│   └── 7b/
├── physical_models/
├── predictions/
├── metrics/
├── bootstrap/
├── deployment/
└── figures/
```

---

# 18. Required Automated Tests

Add checks for:

```text
exact retention ratio
FFN index alignment
logical/physical equivalence
deterministic final mask
train/eval disjointness
identical evaluation IDs across methods
identical calibration data across lambda conditions
same 600-step budget for 1.5B and 3B
seed propagation
metric reproducibility
```

Recommended tests:

```text
test_exact_retention_budget
test_ffn_index_alignment
test_physical_export_matches_logical
test_dataset_disjointness
test_lambda_zero_is_kl_only
test_seed_reproducibility
test_eval_pairing
```

---

# 19. Paper Figures

## Figure 1 — UPGD method

```text
Dense model
   ↓
teacher probabilities + uncertainty
   ↓
joint FFN gate optimization
   ↓
KL + lambda × uncertainty preservation
   ↓
exact structured mask
   ↓
physically compressed model
```

## Figure 2 — Compression severity vs accuracy

x-axis:

```text
75%, 50%, 25% retention
```

curves:

```text
one-shot
FLAP
KL-only
Selected UPGD
```

for both model scales.

## Figure 3 — Compression severity vs uncertainty fidelity

x-axis:

```text
FFN retention
```

y-axis:

```text
entropy MAE to dense teacher
```

## Figure 4 — Accuracy/uncertainty Pareto frontier

```text
x = accuracy loss
y = entropy MAE
```

## Figure 5 — Deployment frontier

Possible versions:

```text
throughput vs accuracy
parameter count vs accuracy
throughput vs entropy MAE
```

## Figure 6 — Teacher uncertainty fidelity is not calibration

Use the clearest task, likely MMLU auxiliary, to compare:

```text
lambda
vs
entropy MAE
ECE
KL
accuracy
```

The point is that teacher-uncertainty fidelity and ground-truth calibration are distinct.

---

# 20. Paper Tables

## Table 1 — Main results

Columns:

```text
Model
Retention
Method
Accuracy
KL
Entropy MAE
AURC
ECE
```

Rows:

```text
Dense
Random
One-shot
FLAP
KL-only
Selected UPGD
```

## Table 2 — Lambda ablation

For both models and every retention ratio:

```text
lambda=0
lambda=.5
lambda=1
```

Report mean ± SD.

## Table 3 — Direct UQ contribution

Report:

```text
UPGD - KL-only
```

with 95% paired CIs for:

```text
accuracy
KL
entropy MAE
ECE
AURC
```

## Table 4 — Deployment

Columns:

```text
Model
Retention
Method
Parameters
Checkpoint size
Peak VRAM
Tokens/s
Latency
```

---

# 21. Mechanistic Analysis

Run only after the primary study is complete.

## 21.1 Mask overlap

For masks \(A\) and \(B\):

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Compute per-layer Jaccard similarity for:

```text
KL-only vs lambda=.5
KL-only vs lambda=1
lambda=.5 vs lambda=1
```

## 21.2 Layerwise sensitivity

Plot:

```text
layer index
vs
mask overlap
```

This tests whether UQ regularization changes primarily early, middle, or late FFN channels.

## 21.3 Uncertainty-stratum behavior

Split final examples by dense entropy:

```text
Low
Middle
High
```

Report:

```text
accuracy drop
entropy MAE
dense agreement
```

for each retention ratio.

Supporting question:

> Does aggressive compression disproportionately damage high-uncertainty examples, and does UPGD reduce that damage?

---

# 22. Interpretation Rules

## Core claim

Target:

> Structured pruning affects predictive performance, dense-teacher uncertainty fidelity, and downstream calibration differently. UPGD exposes and controls this trade-off by adding uncertainty-preservation objectives to exact-budget joint gate distillation.

## Stronger claim only if supported

If UPGD becomes more advantageous as retention decreases:

> Explicit uncertainty preservation becomes more valuable under aggressive structured compression.

Do not assume this before running the retention sweep.

## If results stay task-dependent

Use:

> The optimal uncertainty-preservation strength is model-, task-, and compression-dependent; fixed uncertainty regularization is not universally optimal.

This remains a valid workshop contribution.

## Avoid

Do not claim:

```text
UPGD universally improves accuracy.
UPGD universally improves calibration.
50% FFN pruning means 50% smaller model.
logical masking provides inference acceleration.
teacher entropy matching guarantees calibrated probabilities.
```

---

# 23. Priority Checklist

## Priority A — Must complete

- [ ] Freeze Iteration 16.
- [ ] Add 3B BF16 model path.
- [ ] Generalize physical FFN export to 3B.
- [ ] Verify physical/logical equivalence.
- [ ] Build disjoint multi-task calibration data.
- [ ] Build held-out lambda-selection data.
- [ ] Standardize both models to 600 gate steps.
- [ ] Implement retention `{0.75, 0.50, 0.25}`.
- [ ] Run lambda `{0, 0.5, 1}`.
- [ ] Run three learned-mask seeds.
- [ ] Evaluate HellaSwag, ARC-Challenge, and MMLU auxiliary.
- [ ] Run matched random controls.
- [ ] Run matched one-shot controls.
- [ ] Compute task utility, teacher fidelity, and calibration metrics.
- [ ] Compute paired bootstrap CIs.
- [ ] Generate compression/accuracy/uncertainty Pareto plots.
- [ ] Physically benchmark 1.5B and 3B.
- [ ] Report real parameter, memory, latency, and throughput changes.

## Priority B — Strongly recommended

- [ ] Implement FLAP for all retention ratios.
- [ ] Use held-out lambda selection in the main reported pipeline.
- [ ] Add NLL and Brier score.
- [ ] Analyze seed stability.
- [ ] Analyze mask overlap across lambda values.
- [ ] Analyze layerwise channel-selection changes.

## Priority C — Only after the core paper is complete

- [ ] Add 7B BF16 confirmation.
- [ ] Add structured Wanda or another second external baseline.
- [ ] Add another retention ratio.
- [ ] Study adaptive/per-layer lambda.
- [ ] Study task-conditioned lambda selection.
- [ ] Add energy measurements if reliable.

---

# 24. Experiment Freeze Criteria

Stop adding experiments once:

- [ ] 1.5B and 3B both use BF16.
- [ ] both use 600 gate steps.
- [ ] both support physical FFN pruning.
- [ ] 75%, 50%, and 25% retention are complete.
- [ ] lambda `{0, .5, 1}` is complete.
- [ ] three learned-mask seeds are complete.
- [ ] all three core datasets are complete.
- [ ] random and one-shot controls are complete.
- [ ] FLAP is complete or clearly documented as infeasible.
- [ ] held-out lambda selection is complete.
- [ ] paired statistics are complete.
- [ ] deployment benchmarks are complete.
- [ ] figures/tables regenerate automatically.
- [ ] final run IDs/configs are frozen.

Then move entirely to paper writing.

---

# 25. Target Final Story

The enhanced paper should read as a **study of what structured compression preserves and destroys**, with UPGD as the proposed control mechanism.

Narrative:

1. Structured pruning reduces computation but can distort predictive uncertainty.
2. Accuracy, teacher-distribution fidelity, teacher-uncertainty fidelity, and calibration are not equivalent.
3. KL-only joint gate learning preserves teacher predictions but does not directly control uncertainty distortion.
4. UPGD explicitly regularizes uncertainty preservation under exact structured pruning budgets.
5. Across model scale and compression severity, the study characterizes the accuracy–uncertainty trade-off.
6. Physical pruning determines whether those quality trade-offs translate into real parameter, memory, and inference-efficiency gains.
7. The final contribution is therefore not merely a new channel-selection heuristic, but an empirical and methodological study of **what LLM compression preserves, what it silently destroys, and when uncertainty-aware pruning is worthwhile**.
