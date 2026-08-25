#!/usr/bin/env python3
"""Compute dense-to-compressed entropy-threshold compatibility metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dualkd.config import SUPPORTED_METHODS, load_config
from dualkd.evaluation import TEACHER_ESCALATION_RATES, threshold_compatibility


Condition = tuple[str, str, float, str, int]
# ponytail: defaults reproduce Figure 1; use the plot selectors for another condition.
FIGURE_1_CONDITION = ("ARC-Challenge", "3b_bf16", 0.25, 0.10)


def load_conditions(path: Path) -> dict[Condition, dict[str, tuple[float, float]]]:
    conditions: dict[Condition, dict[str, tuple[float, float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            try:
                condition = (
                    str(record["dataset"]),
                    str(record["model"]),
                    float(record["retention"]),
                    str(record["method"]),
                    int(record["seed"]),
                )
                example_id = str(record["example_id"])
                entropies = (
                    float(record["dense_entropy"]),
                    float(record["compressed_entropy"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}") from exc
            if example_id in conditions[condition]:
                raise ValueError(f"duplicate example {example_id!r} in condition {condition}")
            conditions[condition][example_id] = entropies
    if not conditions:
        raise ValueError(f"no per-example records in {path}")
    return dict(conditions)


def compute_rows(
    conditions: dict[Condition, dict[str, tuple[float, float]]],
    rates: tuple[float, ...],
) -> list[dict[str, object]]:
    references: dict[tuple[str, str, int], tuple[list[str], np.ndarray]] = {}
    output = []
    for condition, examples in sorted(conditions.items()):
        dataset, model, retention, method, seed = condition
        ids = sorted(examples)
        teacher = np.asarray([examples[example_id][0] for example_id in ids])
        student = np.asarray([examples[example_id][1] for example_id in ids])
        reference_key = (dataset, model, seed)
        if reference_key in references:
            reference_ids, reference_teacher = references[reference_key]
            if ids != reference_ids or not np.array_equal(teacher, reference_teacher):
                raise ValueError(f"dense examples are not aligned for {dataset}/{model}/seed_{seed}")
        else:
            references[reference_key] = (ids, teacher)

        for metrics in threshold_compatibility(teacher, student, rates):
            output.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "retention": retention,
                    "method": method,
                    "seed": seed,
                    "examples": len(ids),
                    **metrics,
                }
            )
    return output


def plot_condition(
    conditions: dict[Condition, dict[str, tuple[float, float]]],
    path: Path,
    *,
    dataset: str,
    model: str,
    retention: float,
    seed: int,
    rate: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional local tooling
        raise RuntimeError("--plot requires matplotlib") from exc

    selected = {
        method: conditions.get((dataset, model, retention, method, seed))
        for method in SUPPORTED_METHODS
    }
    missing = [method for method, examples in selected.items() if examples is None]
    if missing:
        raise ValueError(
            f"plot condition {dataset}/{model}/{retention}/seed_{seed} "
            f"is missing methods: {', '.join(missing)}"
        )

    ids = sorted(selected[SUPPORTED_METHODS[0]])
    teacher = np.asarray([selected[SUPPORTED_METHODS[0]][example_id][0] for example_id in ids])
    students = {}
    for method, examples in selected.items():
        method_ids = sorted(examples)
        method_teacher = np.asarray([examples[example_id][0] for example_id in method_ids])
        if method_ids != ids or not np.array_equal(method_teacher, teacher):
            raise ValueError(f"dense examples are not aligned for plot method {method}")
        students[method] = np.asarray([examples[example_id][1] for example_id in ids])

    metrics = {
        method: threshold_compatibility(teacher, student, [rate])[0]
        for method, student in students.items()
    }
    threshold = metrics[SUPPORTED_METHODS[0]]["threshold"]
    lo = float(min(teacher.min(), *(student.min() for student in students.values())))
    hi = float(max(teacher.max(), *(student.max() for student in students.values())))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, 56)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def density(values: np.ndarray) -> np.ndarray:
        return np.histogram(values, bins=edges, density=True)[0]

    colors = {"kl_only": "#0072B2", "dual_kl": "#D55E00"}
    labels = {"kl_only": "Vanilla KL", "dual_kl": "Dual-KD"}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axvspan(threshold, hi, color="black", alpha=0.05, linewidth=0)
    ax.fill_between(
        centers,
        density(teacher),
        color="#555555",
        alpha=0.22,
        linewidth=0,
        label=f"Dense teacher: {100 * metrics[SUPPORTED_METHODS[0]]['teacher_rate']:.0f}%",
    )
    for index, method in enumerate(SUPPORTED_METHODS):
        ax.plot(
            centers,
            density(students[method]),
            color=colors[method],
            linestyle="-" if index == 0 else "--",
            linewidth=1.6,
            label=f"{labels[method]}: {100 * metrics[method]['student_rate']:.0f}%",
        )
    ax.axvline(threshold, color="#111111", linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(threshold, 1.01, r"$\tau$", transform=ax.get_xaxis_transform(), ha="right")
    ax.set(
        title=f"{dataset}, {model}, {retention:.0%} retained",
        xlabel="Normalized predictive entropy",
        ylabel="Density",
        xlim=(lo, hi),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=TEACHER_ESCALATION_RATES,
        help="teacher escalation fractions used to define entropy thresholds",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        metavar="PATH",
        help="also render the Figure 1 entropy panel (default: beside the CSV)",
    )
    parser.add_argument("--plot-dataset", default=FIGURE_1_CONDITION[0])
    parser.add_argument("--plot-model", default=FIGURE_1_CONDITION[1])
    parser.add_argument("--plot-retention", type=float, default=FIGURE_1_CONDITION[2])
    parser.add_argument("--plot-rate", type=float, default=FIGURE_1_CONDITION[3])
    args = parser.parse_args()

    config = load_config(args.config)
    downstream = Path(config["output_root"]) / "outputs/downstream"
    input_path = args.input or downstream / "per_example.jsonl"
    output_path = args.output or downstream / "threshold_compatibility.csv"
    conditions = load_conditions(input_path)
    rows = compute_rows(conditions, tuple(args.rates))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} threshold-compatibility rows to {output_path}")
    if args.plot:
        plot_path = output_path.with_suffix(".png") if args.plot == "auto" else Path(args.plot)
        plot_condition(
            conditions,
            plot_path,
            dataset=args.plot_dataset,
            model=args.plot_model,
            retention=args.plot_retention,
            seed=int(config["training"]["seed"]),
            rate=args.plot_rate,
        )
        print(f"wrote threshold-compatibility plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
