# Iteration 17 7B follow-up

This isolated follow-up applies the Iteration 17 paper-enhancement grid to
`Qwen/Qwen2.5-7B-Instruct` in BF16 with no quantization.

- 768 gate-training, 384 selection, and 1536 final-evaluation examples.
- Three per-layer FFN-channel retentions: 0.75, 0.50, and 0.25.
- UPGD lambda values 0, 0.5, and 1.0, with three seeds per cell: 27 learned masks.
- One-shot, fixed-budget FLAP WIFV, and random controls at each retention.
- Physical export/equivalence, fixed-prompt deployment benchmarks, and independent validation. The 7B results are merged into the canonical Iteration 17 report rather than rendered as a separate follow-up report.

The run is intentionally separate from `runs/iter17_paper_enhancement`.
Its final report surface is the existing `runs/iter17_paper_enhancement/report/` artifact.
