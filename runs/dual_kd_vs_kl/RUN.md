# Dual-KD vs vanilla KL: full supported grid

Executed 2026-08-24/25 on Nova. Complete grid over the supported configuration
in `configs/experiment.yaml`, extended with a third model.

## Scope

- Models: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct (all BF16)
- Datasets: ARC-Challenge, MMLU-AUX, MMLU-Pro, HellaSwag (official splits)
- Methods: `kl_only`, `dual_kl`
- Retentions: 0.75, 0.50, 0.25
- 72 masks, one seed, two exact without-replacement epochs

`validate_results.py` passed: 12 calibrations, 72 training conditions, 144
prediction files, dataset partition counts and ID disjointness verified.

## Result: the entropy constraint binds; accuracy pays for it

**Uncertainty fidelity.** Entropy MAE to the dense teacher improves in 30 of 36
cells, scaling with compression pressure. On MMLU-AUX at 25% retention:
1.5B 0.6324 -> 0.2842 (-55%), 3B 0.8238 -> 0.1733 (-79%), 7B 0.6413 -> 0.1547 (-76%).

HellaSwag is the control: vanilla KL already reaches 0.02-0.09 entropy error
there, so the constraint has nothing to close and Dual-KD changes nothing
(delta within +/-0.006). The effect appears only where the fidelity gap is wide.

**Accuracy.** Dual-KD improves held-out accuracy in only 8 of 36 cells, and
loses most where the entropy gains are largest: 7B MMLU-AUX at 25% retention
falls 0.3816 -> 0.2689 (-11.3 points). ECE tracks accuracy rather than fidelity,
also improving in only 8 of 36.

**Selective prediction.** Of 72 paired bootstraps (5,000 replicates) on error
AUROC and AUPRC, 52 intervals cross zero; of the 20 that resolve, 13 favour
`kl_only` and 7 favour `dual_kl`. The clearest exception is the most aggressive
setting: 7B MMLU-AUX at 25% retention gains +0.1123 AUPRC [+0.0976, +0.1274].

**Threshold compatibility.** Applying the dense model's entropy threshold unchanged
to the compressed model, Dual-KD holds the intended operating point in 143 of 144
paired conditions and lowers escalation disagreement in 142. Under vanilla KL the
threshold collapses at aggressive compression, escalating 100% of inputs instead of
10%. See the section below.

The warranted conclusion is a trade-off with one clear operational winner: Dual-KD
costs capability nearly everywhere the constraint binds, but it is what allows a
dense-calibrated decision threshold to survive compression at all.

## Configuration changes

- Added `7b_bf16: Qwen/Qwen2.5-7B-Instruct` to the model grid (48 -> 72 cells).
  `config.py` validates methods and datasets exactly but leaves `models` open,
  so this is a supported extension.
- Raised HellaSwag `choice_microbatch_size` from 2 to 4. Measured 0.52 -> 0.37
  s/step at equal peak memory. Microbatching is gradient accumulation within a
  step. It is neutral in expectation, but a later distributed experiment showed
  that re-splitting a step's batch perturbs bf16 reduction order enough to flip
  channels at the exact-top-k boundary, so it is not bit-neutral. It was applied
  uniformly to every cell before any training ran, so the grid stays internally
  consistent and every method pair is matched.

## Limitations

**Single seed.** `training.seed: 1`. The paired bootstrap resamples examples,
not training runs, so intervals capture example-level uncertainty only. Treat
every comparison as seed-sensitive.

**3B / MMLU-AUX has been corrected.** Those three cells originally trained
`kl_only` on A100 (sm_80) and `dual_kl` on RTX PRO 6000 (sm_120) after an
out-of-memory reschedule. They were retrained on matched Blackwell hardware and
re-analysed; the numbers below are the corrected ones. The effect was small and
directional: KL entropy MAE moved 0.3193 -> 0.3301, 0.4909 -> 0.5134, 0.8238 ->
0.8237, leaving the -79% headline intact. All 36 method pairs are now matched.

**Claim boundary.** Masks are exact top-k logical channel gates applied to frozen
BF16 weights. These establish model quality only; no packed-kernel latency,
memory, or energy claim follows.

## Memory notes

Peak training memory is driven by prompt length, not choice count, and cannot be
reduced by `choice_microbatch_size` because the autograd graph retains every
microbatch until backward.

- 1.5B fits every card, including 40GB A100-PCIE.
- 3B needs 80GB+ for MMLU-AUX and MMLU-Pro.
- 7B needs 96GB for MMLU-AUX and MMLU-Pro; its prompt tail (p99 811, max 1367
  tokens) exceeds 80GB. ARC and HellaSwag fit in 80GB.

## Comparison table

All 36 cells. Accuracy is held-out exact-choice accuracy; entropy MAE is mean
absolute normalised-entropy error against the dense teacher (lower is better).
`d` columns are `dual_kl - kl_only`. Error-AUROC `d` carries the paired
bootstrap 95% interval; **bold** marks intervals excluding zero.
Generated from `outputs/downstream/evaluation_summary.csv` and
`bootstrap_comparisons.csv`.

### Qwen2.5-1.5B

| dataset | ret | dense | acc KL | acc Dual | d acc | entMAE KL | entMAE Dual | d entMAE | d errAUROC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | 0.75 | 0.4693 | 0.4130 | 0.4147 | +0.0017 | 0.0758 | 0.0743 | -0.0015 | +0.0039 [-0.0215, +0.0293] |
| ARC-Challenge | 0.5 | 0.4693 | 0.3123 | 0.2918 | -0.0205 | 0.1063 | 0.1009 | -0.0055 | +0.0220 [-0.0148, +0.0567] |
| ARC-Challenge | 0.25 | 0.4693 | 0.2210 | 0.2235 | +0.0026 | 0.1544 | 0.1339 | -0.0205 | -0.0170 [-0.0612, +0.0277] |
| MMLU-AUX | 0.75 | 0.5927 | 0.4880 | 0.4890 | +0.0011 | 0.2820 | 0.2163 | -0.0657 | +0.0068 [-0.0031, +0.0166] |
| MMLU-AUX | 0.5 | 0.5927 | 0.3997 | 0.3843 | -0.0154 | 0.4228 | 0.2450 | -0.1779 | **-0.0195 [-0.0311, -0.0078]** |
| MMLU-AUX | 0.25 | 0.5927 | 0.2505 | 0.2465 | -0.0039 | 0.6324 | 0.2842 | -0.3482 | +0.0008 [-0.0140, +0.0159] |
| MMLU-Pro | 0.75 | 0.2826 | 0.2406 | 0.2244 | -0.0162 | 0.1771 | 0.1538 | -0.0233 | +0.0099 [-0.0183, +0.0385] |
| MMLU-Pro | 0.5 | 0.2826 | 0.1999 | 0.1929 | -0.0071 | 0.2533 | 0.2150 | -0.0382 | +0.0268 [-0.0061, +0.0594] |
| MMLU-Pro | 0.25 | 0.2826 | 0.1072 | 0.1106 | +0.0033 | 0.3396 | 0.2743 | -0.0653 | +0.0025 [-0.0197, +0.0250] |
| HellaSwag | 0.75 | 0.6397 | 0.6007 | 0.5987 | -0.0020 | 0.0236 | 0.0233 | -0.0003 | +0.0036 [-0.0064, +0.0139] |
| HellaSwag | 0.5 | 0.6397 | 0.5361 | 0.5268 | -0.0093 | 0.0362 | 0.0359 | -0.0002 | +0.0064 [-0.0052, +0.0184] |
| HellaSwag | 0.25 | 0.6397 | 0.3601 | 0.3389 | -0.0212 | 0.0552 | 0.0583 | +0.0031 | **-0.0149 [-0.0291, -0.0004]** |

### Qwen2.5-3B

| dataset | ret | dense | acc KL | acc Dual | d acc | entMAE KL | entMAE Dual | d entMAE | d errAUROC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | 0.75 | 0.4616 | 0.4437 | 0.4608 | +0.0171 | 0.1109 | 0.0961 | -0.0148 | -0.0093 [-0.0307, +0.0124] |
| ARC-Challenge | 0.5 | 0.4616 | 0.3532 | 0.3780 | +0.0247 | 0.1420 | 0.1253 | -0.0167 | -0.0030 [-0.0329, +0.0273] |
| ARC-Challenge | 0.25 | 0.4616 | 0.2696 | 0.2449 | -0.0247 | 0.1966 | 0.1785 | -0.0181 | -0.0178 [-0.0636, +0.0284] |
| MMLU-AUX | 0.75 | 0.6540 | 0.5480 | 0.5326 | -0.0154 | 0.3301 | 0.1625 | -0.1676 | **-0.0219 [-0.0312, -0.0122]** |
| MMLU-AUX | 0.5 | 0.6540 | 0.4393 | 0.3955 | -0.0438 | 0.5134 | 0.1741 | -0.3393 | **-0.0460 [-0.0575, -0.0345]** |
| MMLU-AUX | 0.25 | 0.6540 | 0.2474 | 0.2295 | -0.0179 | 0.8237 | 0.1733 | -0.6504 | **+0.0158 [+0.0004, +0.0305]** |
| MMLU-Pro | 0.75 | 0.3387 | 0.2943 | 0.2926 | -0.0017 | 0.2578 | 0.1936 | -0.0642 | -0.0046 [-0.0297, +0.0212] |
| MMLU-Pro | 0.5 | 0.3387 | 0.2419 | 0.2307 | -0.0112 | 0.3972 | 0.2408 | -0.1564 | -0.0204 [-0.0534, +0.0132] |
| MMLU-Pro | 0.25 | 0.3387 | 0.1168 | 0.1126 | -0.0042 | 0.5649 | 0.2583 | -0.3066 | **+0.0451 [+0.0062, +0.0849]** |
| HellaSwag | 0.75 | 0.7229 | 0.6954 | 0.6902 | -0.0052 | 0.0433 | 0.0435 | +0.0002 | -0.0035 [-0.0148, +0.0076] |
| HellaSwag | 0.5 | 0.7229 | 0.6388 | 0.6256 | -0.0132 | 0.0604 | 0.0605 | +0.0001 | -0.0001 [-0.0124, +0.0123] |
| HellaSwag | 0.25 | 0.7229 | 0.4704 | 0.3820 | -0.0884 | 0.0864 | 0.0921 | +0.0057 | **-0.0179 [-0.0326, -0.0035]** |

### Qwen2.5-7B

| dataset | ret | dense | acc KL | acc Dual | d acc | entMAE KL | entMAE Dual | d entMAE | d errAUROC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | 0.75 | 0.5341 | 0.5444 | 0.5358 | -0.0085 | 0.1283 | 0.1139 | -0.0144 | +0.0100 [-0.0122, +0.0330] |
| ARC-Challenge | 0.5 | 0.5341 | 0.4522 | 0.4718 | +0.0196 | 0.1615 | 0.1385 | -0.0230 | +0.0272 [-0.0015, +0.0554] |
| ARC-Challenge | 0.25 | 0.5341 | 0.3259 | 0.3157 | -0.0102 | 0.2155 | 0.1852 | -0.0302 | +0.0153 [-0.0272, +0.0570] |
| MMLU-AUX | 0.75 | 0.6851 | 0.6100 | 0.6077 | -0.0023 | 0.2682 | 0.1473 | -0.1209 | **-0.0248 [-0.0341, -0.0157]** |
| MMLU-AUX | 0.5 | 0.6851 | 0.4974 | 0.4699 | -0.0274 | 0.4684 | 0.1582 | -0.3102 | **-0.0300 [-0.0404, -0.0192]** |
| MMLU-AUX | 0.25 | 0.6851 | 0.3816 | 0.2689 | -0.1127 | 0.6413 | 0.1547 | -0.4866 | **+0.0225 [+0.0097, +0.0357]** |
| MMLU-Pro | 0.75 | 0.3965 | 0.3533 | 0.3504 | -0.0029 | 0.2047 | 0.1547 | -0.0499 | -0.0195 [-0.0393, +0.0011] |
| MMLU-Pro | 0.5 | 0.3965 | 0.2884 | 0.2693 | -0.0191 | 0.3038 | 0.2142 | -0.0896 | -0.0138 [-0.0410, +0.0141] |
| MMLU-Pro | 0.25 | 0.3965 | 0.2086 | 0.1654 | -0.0432 | 0.4807 | 0.2874 | -0.1933 | **-0.0415 [-0.0798, -0.0030]** |
| HellaSwag | 0.75 | 0.7806 | 0.7650 | 0.7667 | +0.0017 | 0.0458 | 0.0461 | +0.0003 | -0.0007 [-0.0115, +0.0101] |
| HellaSwag | 0.5 | 0.7806 | 0.7280 | 0.7248 | -0.0032 | 0.0651 | 0.0647 | -0.0003 | **-0.0131 [-0.0251, -0.0008]** |
| HellaSwag | 0.25 | 0.7806 | 0.6003 | 0.5822 | -0.0182 | 0.0904 | 0.0923 | +0.0019 | **-0.0133 [-0.0264, -0.0006]** |

## Threshold compatibility

`scripts/check_threshold_compatibility.py` applies the **dense** model's entropy
threshold, unchanged, to the compressed model. `student_rate` is the fraction the
compressed model then escalates; it should match the target. This is the
operational consequence of conserved uncertainty, and it is where Dual-KD's
advantage is largest.

| model | dataset | ret | student rate KL | student rate Dual | disagreement KL | disagreement Dual |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | ARC-Challenge | 0.75 | 0.1280 | 0.1365 | 0.1041 | 0.1041 |
| Qwen2.5-1.5B | ARC-Challenge | 0.5 | 0.1664 | 0.1382 | 0.1681 | 0.1468 |
| Qwen2.5-1.5B | ARC-Challenge | 0.25 | 0.3891 | 0.1220 | 0.3396 | 0.1510 |
| Qwen2.5-1.5B | MMLU-AUX | 0.75 | 0.2902 | 0.1101 | 0.2474 | 0.1570 |
| Qwen2.5-1.5B | MMLU-AUX | 0.5 | 0.5904 | 0.0040 | 0.5220 | 0.1019 |
| Qwen2.5-1.5B | MMLU-AUX | 0.25 | 1.0000 | 0.0000 | 0.8999 | 0.1001 |
| Qwen2.5-1.5B | MMLU-Pro | 0.75 | 0.2140 | 0.1002 | 0.1945 | 0.1313 |
| Qwen2.5-1.5B | MMLU-Pro | 0.5 | 0.3978 | 0.1043 | 0.3749 | 0.1663 |
| Qwen2.5-1.5B | MMLU-Pro | 0.25 | 0.5707 | 0.0121 | 0.5362 | 0.1106 |
| Qwen2.5-1.5B | HellaSwag | 0.75 | 0.1207 | 0.1012 | 0.0797 | 0.0741 |
| Qwen2.5-1.5B | HellaSwag | 0.5 | 0.1359 | 0.1098 | 0.1257 | 0.1140 |
| Qwen2.5-1.5B | HellaSwag | 0.25 | 0.2458 | 0.1095 | 0.2436 | 0.1606 |
| Qwen2.5-3B | ARC-Challenge | 0.75 | 0.1817 | 0.1468 | 0.1220 | 0.1109 |
| Qwen2.5-3B | ARC-Challenge | 0.5 | 0.2338 | 0.1451 | 0.1945 | 0.1314 |
| Qwen2.5-3B | ARC-Challenge | 0.25 | 0.4275 | 0.2108 | 0.3797 | 0.2159 |
| Qwen2.5-3B | MMLU-AUX | 0.75 | 0.4748 | 0.0261 | 0.4024 | 0.1130 |
| Qwen2.5-3B | MMLU-AUX | 0.5 | 0.7419 | 0.0063 | 0.6545 | 0.1041 |
| Qwen2.5-3B | MMLU-AUX | 0.25 | 1.0000 | 0.0000 | 0.8999 | 0.1001 |
| Qwen2.5-3B | MMLU-Pro | 0.75 | 0.2631 | 0.0453 | 0.2336 | 0.1189 |
| Qwen2.5-3B | MMLU-Pro | 0.5 | 0.5744 | 0.0100 | 0.5191 | 0.1093 |
| Qwen2.5-3B | MMLU-Pro | 0.25 | 0.9672 | 0.0004 | 0.8753 | 0.1006 |
| Qwen2.5-3B | HellaSwag | 0.75 | 0.1261 | 0.1115 | 0.0853 | 0.0826 |
| Qwen2.5-3B | HellaSwag | 0.5 | 0.1414 | 0.1069 | 0.1312 | 0.1174 |
| Qwen2.5-3B | HellaSwag | 0.25 | 0.2350 | 0.1112 | 0.2315 | 0.1654 |
| Qwen2.5-7B | ARC-Challenge | 0.75 | 0.1826 | 0.1587 | 0.1297 | 0.1143 |
| Qwen2.5-7B | ARC-Challenge | 0.5 | 0.2312 | 0.1459 | 0.1903 | 0.1323 |
| Qwen2.5-7B | ARC-Challenge | 0.25 | 0.3695 | 0.1365 | 0.3302 | 0.1536 |
| Qwen2.5-7B | MMLU-AUX | 0.75 | 0.3645 | 0.0386 | 0.3043 | 0.1143 |
| Qwen2.5-7B | MMLU-AUX | 0.5 | 0.6596 | 0.0142 | 0.5752 | 0.1095 |
| Qwen2.5-7B | MMLU-AUX | 0.25 | 0.8837 | 0.0000 | 0.7882 | 0.1001 |
| Qwen2.5-7B | MMLU-Pro | 0.75 | 0.2452 | 0.0844 | 0.1783 | 0.1072 |
| Qwen2.5-7B | MMLU-Pro | 0.5 | 0.4335 | 0.0278 | 0.3558 | 0.1172 |
| Qwen2.5-7B | MMLU-Pro | 0.25 | 0.7988 | 0.0067 | 0.7136 | 0.1068 |
| Qwen2.5-7B | HellaSwag | 0.75 | 0.1151 | 0.1012 | 0.0768 | 0.0723 |
| Qwen2.5-7B | HellaSwag | 0.5 | 0.1377 | 0.1119 | 0.1148 | 0.1044 |
| Qwen2.5-7B | HellaSwag | 0.25 | 0.1875 | 0.1115 | 0.1889 | 0.1514 |

Target escalation rate 0.10; ideal `student_rate` is 0.1000.

| target rate | mean abs rate error KL | Dual-KD | mean disagreement KL | Dual-KD |
|---|---|---|---|---|
| 0.05 | 0.2509 | **0.0292** | 0.2865 | **0.0693** |
| 0.10 | 0.2951 | **0.0487** | 0.3475 | **0.1214** |
| 0.20 | 0.3346 | **0.0842** | 0.3991 | **0.2072** |
| 0.30 | 0.3491 | **0.1072** | 0.4081 | **0.2724** |

Dual-KD transfers the dense threshold better in **143/144** paired conditions and
lowers escalation disagreement in **142/144**. Under vanilla KL the dense threshold
collapses at aggressive compression: at 25% retention on MMLU-AUX it escalates
100% of inputs instead of 10%, because the compressed entropy distribution has
shifted wholesale past the threshold. Dual-KD sometimes undershoots instead, which
is a far more benign failure than escalating all traffic.

