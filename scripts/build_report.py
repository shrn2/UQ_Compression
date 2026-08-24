#!/usr/bin/env python3
"""Create a self-contained KL versus Dual-KD HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dualkd.config import SUPPORTED_METHODS, load_config


METHOD_LABELS = {"kl_only": "Vanilla KL", "dual_kl": "Dual-KD"}
METRICS = ("accuracy", "entropy_mae", "margin_mae", "ece", "error_auroc", "error_auprc", "aurc")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in METRICS + ("retention", "seed", "examples", "epochs"):
            if key in row and row[key] not in {"", None}:
                row[key] = float(row[key])
    return rows


def fmt(metric: str, value: float) -> str:
    if metric == "accuracy":
        return f"{100 * value:.1f}%"
    return f"{value:.3f}"


def macro_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)
    result = []
    for (dataset, method), values in groups.items():
        result.append(
            {
                "dataset": dataset,
                "method": method,
                **{metric: float(np.mean([row[metric] for row in values])) for metric in METRICS},
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--downstream", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.root or Path(config["output_root"])
    downstream = args.downstream or root / "outputs/downstream"
    report = args.report_dir or root / "report"
    report.mkdir(parents=True, exist_ok=True)
    evaluation = read_csv(downstream / "evaluation_summary.csv")
    validation = read_csv(downstream / "validation_summary.csv")
    training = read_csv(downstream / "training_audit.csv")
    bootstrap = read_csv(downstream / "bootstrap_comparisons.csv")
    cascade = read_csv(downstream / "cascade_curves.csv")
    methods = tuple(dict.fromkeys(row["method"] for row in evaluation))
    if methods != SUPPORTED_METHODS and set(methods) != set(SUPPORTED_METHODS):
        raise ValueError(f"report accepts only {SUPPORTED_METHODS}, found {methods}")
    macro = macro_rows(evaluation)
    macro_by_key = {(row["dataset"], row["method"]): row for row in macro}
    datasets = [config["datasets"][key]["label"] for key in config["datasets"]]

    paired = []
    grouped = defaultdict(dict)
    for row in evaluation:
        grouped[(row["dataset"], row["model"], row["retention"])][row["method"]] = row
    for key, values in grouped.items():
        if set(values) != set(SUPPORTED_METHODS):
            raise ValueError(f"incomplete matched condition {key}")
        paired.append(
            {
                "dataset": key[0],
                "model": key[1],
                "retention": key[2],
                "delta_entropy_mae": values["dual_kl"]["entropy_mae"] - values["kl_only"]["entropy_mae"],
                "delta_accuracy": values["dual_kl"]["accuracy"] - values["kl_only"]["accuracy"],
            }
        )
    paired.sort(key=lambda row: (datasets.index(row["dataset"]), row["model"], row["retention"]))
    entropy_wins = sum(row["delta_entropy_mae"] < 0 for row in paired)
    accuracy_wins = sum(row["delta_accuracy"] > 0 for row in paired)
    scale = max(abs(row["delta_entropy_mae"]) for row in paired) or 1.0

    macro_table = []
    for dataset in datasets:
        kl = macro_by_key[(dataset, "kl_only")]
        dual = macro_by_key[(dataset, "dual_kl")]
        macro_table.append(
            "<tr>"
            f"<th>{html.escape(dataset)}</th>"
            + "".join(
                f"<td>{fmt(metric, kl[metric])} / {fmt(metric, dual[metric])}</td>"
                for metric in ("accuracy", "entropy_mae", "ece", "error_auroc", "error_auprc", "aurc")
            )
            + "</tr>"
        )
    condition_table = []
    for row in evaluation:
        condition_table.append(
            "<tr>"
            f"<td>{html.escape(row['dataset'])}</td><td>{html.escape(row['model'])}</td>"
            f"<td>{100 * row['retention']:.0f}%</td><td>{METHOD_LABELS[row['method']]}</td>"
            + "".join(f"<td>{fmt(metric, row[metric])}</td>" for metric in ("accuracy", "entropy_mae", "margin_mae", "ece", "error_auroc", "error_auprc", "aurc"))
            + "</tr>"
        )
    delta_rows = []
    for row in paired:
        width = 48 * abs(row["delta_entropy_mae"]) / scale
        side = "left" if row["delta_entropy_mae"] < 0 else "right"
        delta_rows.append(
            "<div class='delta-row'>"
            f"<span>{html.escape(row['dataset'])} · {html.escape(row['model'])} · {100 * row['retention']:.0f}%</span>"
            "<div class='track'><i class='zero'></i>"
            f"<b class='{side}' style='width:{width:.2f}%'></b></div>"
            f"<code>{row['delta_entropy_mae']:+.3f}</code></div>"
        )
    significant = [
        row for row in bootstrap
        if float(row["ci_low"]) > 0 or float(row["ci_high"]) < 0
    ]
    generated = datetime.now(timezone.utc).isoformat()
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-KD versus Vanilla KL</title>
<style>
:root{{--blue:#0072b2;--orange:#d55e00;--ink:#17202a;--muted:#5f6b76;--line:#dce2e7;--paper:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:#f4f6f8;color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;background:var(--paper);padding:42px 52px;box-shadow:0 0 30px #ccd2d8}}
h1{{font-size:32px;margin:0 0 4px}} h2{{margin-top:38px;border-bottom:2px solid var(--ink);padding-bottom:6px}}
.lede{{font-size:18px;max-width:900px}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:15px}} .card strong{{display:block;font-size:26px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}} thead th{{position:sticky;top:0;background:#eef2f5}} .scroll{{overflow:auto;max-height:560px}}
.delta-row{{display:grid;grid-template-columns:250px 1fr 60px;gap:12px;align-items:center;margin:7px 0;font-size:13px}}
.track{{height:14px;background:#f1f3f5;position:relative}} .zero{{position:absolute;left:50%;height:100%;border-left:1px solid #111}}
.track b{{position:absolute;height:8px;top:3px}} .track .left{{right:50%;background:var(--blue)}} .track .right{{left:50%;background:var(--orange)}}
code{{text-align:right}} footer{{margin-top:42px;color:var(--muted);font-size:12px}} @media(max-width:760px){{main{{padding:24px}}.cards{{grid-template-columns:1fr 1fr}}.delta-row{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Vanilla KL versus Dual-constrained KD</h1>
<p class="lede">Matched structured FFN gate compression on four datasets, two Qwen2.5 model sizes, and three retention levels. Teacher-uncertainty fidelity is reported separately from task accuracy, calibration, error detection, and selective prediction.</p>
<div class="cards"><div class="card"><strong>{len(paired)}</strong>matched conditions</div><div class="card"><strong>{entropy_wins}/{len(paired)}</strong>Dual-KD entropy-MAE wins</div><div class="card"><strong>{accuracy_wins}/{len(paired)}</strong>Dual-KD accuracy wins</div><div class="card"><strong>{len(significant)}/{len(bootstrap)}</strong>ranking CIs exclude zero</div></div>
<h2>Dataset macro results</h2><p>Cells are Vanilla KL / Dual-KD. Accuracy is a percentage; all other values are proportions or areas.</p>
<table><thead><tr><th>Dataset</th><th>Accuracy</th><th>Entropy MAE</th><th>ECE</th><th>Error AUROC</th><th>Error AUPRC</th><th>AURC</th></tr></thead><tbody>{''.join(macro_table)}</tbody></table>
<h2>Change in teacher-entropy fidelity</h2><p>Dual-KD minus KL normalized entropy MAE. Blue bars and negative values favor Dual-KD; orange bars favor KL.</p>{''.join(delta_rows)}
<h2>All evaluation conditions</h2><div class="scroll"><table><thead><tr><th>Dataset</th><th>Model</th><th>Retention</th><th>Method</th><th>Accuracy</th><th>Entropy MAE</th><th>Margin MAE</th><th>ECE</th><th>Error AUROC</th><th>Error AUPRC</th><th>AURC</th></tr></thead><tbody>{''.join(condition_table)}</tbody></table></div>
<h2>Interpretation boundary</h2><p>Lower entropy MAE means closer agreement with the dense teacher. It does not by itself establish better calibration to labels, better error ranking, better cascade utility, or realized hardware speedup. Results use one gate-training seed.</p>
<footer>Generated {html.escape(generated)} from CSV artifacts in {html.escape(str(downstream))}. No external JavaScript or report service is required.</footer>
</main></body></html>"""
    (report / "report.html").write_text(document, encoding="utf-8")
    artifact = {
        "study": config["study"],
        "generated_at": generated,
        "methods": list(SUPPORTED_METHODS),
        "evaluation": evaluation,
        "validation": validation,
        "training": training,
        "bootstrap": bootstrap,
        "cascade": cascade,
        "paired": paired,
        "macro": macro,
    }
    (report / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report / "report.html"), "conditions": len(paired)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
