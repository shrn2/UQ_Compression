"""Analyze paired transfer artifacts for both model masks."""

from __future__ import annotations

import json
from pathlib import Path

from tuac.stochastic_gate_pruning import analyze_upgd_comparison


def main() -> int:
    root = Path("runs/iter13_transfer")
    datasets = ("commonsenseqa", "hellaswag", "openbookqa", "winogrande")
    rows = []
    for model_index, model_key in enumerate(("1p5b", "3b_nf4")):
        for dataset_index, dataset in enumerate(datasets):
            reference = root / f"{model_key}_{dataset}_reference.npz"
            upgd = root / f"{model_key}_{dataset}_upgd.npz"
            output = root / f"{model_key}_{dataset}_analysis.json"
            result = analyze_upgd_comparison(
                reference_artifact_path=reference,
                upgd_artifact_path=upgd,
                output_path=output,
                strata=3,
                conformal_alpha=0.1,
                conformal_fraction=0.5,
                split_seed=20260860 + model_index * 100 + dataset_index,
                bootstrap_replicates=5000,
                bootstrap_seed=20260870 + model_index * 100 + dataset_index,
            )
            paired = result["paired_comparison"]
            rows.append({
                "model_key": model_key, "dataset": dataset, "examples": result["scope"]["examples"],
                "difference": paired["difference"], "ci_low": paired["bootstrap_95_ci"][0],
                "ci_high": paired["bootstrap_95_ci"][1], "gate": result["gate"]["passed"],
                "analysis": str(output.resolve()),
            })
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    (root / "analysis_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
