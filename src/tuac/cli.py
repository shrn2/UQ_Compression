"""Command-line interface for calibration collection and offline analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import CalibrationArtifact, MultiBitCalibrationArtifact
from .cascade import (
    CascadeProbabilityArtifact,
    analyze_cascade,
    analyze_cascade_drift_checkpoint,
    collect_cascade_probabilities,
    evaluate_router_gate,
    replace_cascade_fallback,
    score_router_gate,
)
from .cascade_compression import (
    analyze_cascade_compression,
    benchmark_cascade_compression,
    collect_cascade_layer_probes,
)
from .capability_search import benchmark_c3q, search_c3q
from .capacity_contrast import analyze_capacity_contrast, collect_capacity_contrast
from .capacity_contrast_routing import (
    analyze_ccr_system,
    collect_ccr_evaluation,
    freeze_iteration8_linear_ccr,
    prepare_deduplicated_evaluation,
)
from .counterfactual_routing import (
    analyze_counterfactual_routing,
    analyze_oracle_headroom,
    collect_capacity_hierarchy,
    collect_capacity_representations,
)
from .experiment import (
    analyze_artifact,
    benchmark_policies,
    collect_calibration,
    evaluate_policy,
    save_report,
)
from .metrics import classification_metrics
from .iteration import analyze_iteration, benchmark_iteration, collect_multibit_calibration
from .structured_sparsity import (
    analyze_densification,
    benchmark_densification,
    collect_sparse_hierarchy,
    score_sparse_hierarchy,
)
from .stochastic_gate_pruning import (
    analyze_upgd_comparison,
    score_pruned_mask,
    train_upgd_masks,
)
from .trust import analyze_trust_layers, benchmark_trust_policies, build_trust_policies
from .uncertainty_robust_pruning import (
    analyze_ursa_comparison,
    collect_ursa_masks,
    score_ursa_comparison,
)


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tuac", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect teacher/student/layer calibration outputs")
    collect.add_argument("--data", required=True, help="multiple-choice JSONL input")
    collect.add_argument("--teacher", required=True, help="teacher Hugging Face model ID or path")
    collect.add_argument("--student", required=True, help="student Hugging Face model ID or path")
    collect.add_argument("--output", required=True, help="destination .npz artifact")
    collect.add_argument("--probe-bits", type=int, default=4)
    collect.add_argument("--group-size", type=int, default=128)
    collect.add_argument("--limit", type=int)
    collect.add_argument("--device-map", default="auto")
    collect.add_argument("--dtype", default="auto")
    collect.add_argument("--trust-remote-code", action="store_true")

    analyze = commands.add_parser("analyze", help="derive ablations and bit policies from an artifact")
    analyze.add_argument("--artifact", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--budgets", type=_comma_floats, default=(4.0, 3.5, 3.0))
    analyze.add_argument("--bits", type=_comma_ints, default=(3, 4, 8))

    metrics = commands.add_parser("metrics", help="print FP/probe classification metrics")
    metrics.add_argument("--artifact", required=True)

    evaluate = commands.add_parser("evaluate", help="apply and evaluate one mixed-precision policy")
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--report", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--student", required=True)
    evaluate.add_argument("--mode", default="combined")
    evaluate.add_argument("--budget", type=float, required=True)
    evaluate.add_argument("--output", required=True, help="destination probability .npz")
    evaluate.add_argument("--device-map", default="auto")
    evaluate.add_argument("--dtype", default="auto")
    evaluate.add_argument("--trust-remote-code", action="store_true")

    benchmark = commands.add_parser("benchmark", help="evaluate FP, uniform, random, and sensitivity policies")
    benchmark.add_argument("--artifact", required=True)
    benchmark.add_argument("--report", required=True)
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--student", required=True)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--modes", default="student_only,combined")
    benchmark.add_argument("--budgets", type=_comma_floats, default=(4.0, 3.5, 3.0))
    benchmark.add_argument("--random-seeds", type=_comma_ints, default=(0, 1, 2))
    benchmark.add_argument("--device-map", default="auto")
    benchmark.add_argument("--dtype", default="auto")
    benchmark.add_argument("--trust-remote-code", action="store_true")
    benchmark.add_argument("--skip-baselines", action="store_true")

    collect_i1 = commands.add_parser("collect-iteration", help="collect resumable multi-bit iteration probes")
    collect_i1.add_argument("--data", required=True)
    collect_i1.add_argument("--teacher", required=True)
    collect_i1.add_argument("--student", required=True)
    collect_i1.add_argument("--output", required=True)
    collect_i1.add_argument("--bits", type=_comma_ints, default=(2, 4, 8))
    collect_i1.add_argument("--group-size", type=int, default=128)
    collect_i1.add_argument("--batch-size", type=int, default=32)
    collect_i1.add_argument("--device-map", default="auto")
    collect_i1.add_argument("--dtype", default="auto")
    collect_i1.add_argument("--trust-remote-code", action="store_true")

    analyze_i1 = commands.add_parser("analyze-iteration", help="derive interaction policies for five calibration seeds")
    analyze_i1.add_argument("--artifact", required=True)
    analyze_i1.add_argument("--output", required=True)
    analyze_i1.add_argument("--subset-size", type=int, default=500)
    analyze_i1.add_argument("--calibration-seeds", type=_comma_ints, default=(11, 23, 37, 53, 71))
    analyze_i1.add_argument("--budgets", type=_comma_floats, default=(4.0, 3.5, 3.0))
    analyze_i1.add_argument("--interaction-lambda", type=float, default=1.0)

    benchmark_i1 = commands.add_parser("benchmark-iteration", help="evaluate all iteration policies on held-out data")
    benchmark_i1.add_argument("--artifact", required=True)
    benchmark_i1.add_argument("--report", required=True)
    benchmark_i1.add_argument("--data", required=True)
    benchmark_i1.add_argument("--teacher", required=True)
    benchmark_i1.add_argument("--student", required=True)
    benchmark_i1.add_argument("--output-dir", required=True)
    benchmark_i1.add_argument("--batch-size", type=int, default=32)
    benchmark_i1.add_argument("--device-map", default="auto")
    benchmark_i1.add_argument("--dtype", default="auto")
    benchmark_i1.add_argument("--trust-remote-code", action="store_true")

    analyze_i2 = commands.add_parser(
        "analyze-trust", help="compare capability-critical and trust-critical layer rankings"
    )
    analyze_i2.add_argument("--artifact", required=True)
    analyze_i2.add_argument("--output", required=True)
    analyze_i2.add_argument("--bootstrap-replicates", type=int, default=1000)
    analyze_i2.add_argument("--bootstrap-seed", type=int, default=20260817)

    policies_i2 = commands.add_parser(
        "build-trust-policies", help="allocate capability-aware, trust-aware, and joint policies"
    )
    policies_i2.add_argument("--artifact", required=True)
    policies_i2.add_argument("--analysis", required=True)
    policies_i2.add_argument("--output", required=True)
    policies_i2.add_argument("--budgets", type=_comma_floats, default=(4.0, 3.5, 3.0))

    benchmark_i2 = commands.add_parser(
        "benchmark-trust", help="evaluate trust-aware and capability-aware policies"
    )
    benchmark_i2.add_argument("--artifact", required=True)
    benchmark_i2.add_argument("--report", required=True)
    benchmark_i2.add_argument("--data", required=True)
    benchmark_i2.add_argument("--student", required=True)
    benchmark_i2.add_argument("--output-dir", required=True)
    benchmark_i2.add_argument("--batch-size", type=int, default=128)
    benchmark_i2.add_argument("--bootstrap-replicates", type=int, default=2000)
    benchmark_i2.add_argument("--device-map", default="auto")
    benchmark_i2.add_argument("--dtype", default="auto")
    benchmark_i2.add_argument("--trust-remote-code", action="store_true")

    gate_i3 = commands.add_parser(
        "cascade-gate", help="test whether an FP model's uncertainty is viable for routing"
    )
    gate_i3.add_argument("--artifact", required=True)
    gate_i3.add_argument("--role", choices=("student", "teacher"), required=True)
    gate_i3.add_argument("--model", required=True)
    gate_i3.add_argument("--output", required=True)
    gate_i3.add_argument("--threshold", type=float, default=0.60)

    collect_i3 = commands.add_parser(
        "collect-cascade", help="collect uniform-precision SLM and fallback probabilities"
    )
    collect_i3.add_argument("--calibration-data", required=True)
    collect_i3.add_argument("--evaluation-data", required=True)
    collect_i3.add_argument("--slm", required=True)
    collect_i3.add_argument("--fallback", required=True)
    collect_i3.add_argument("--output", required=True)
    collect_i3.add_argument("--bits", type=_comma_ints, default=(8, 4, 2))
    collect_i3.add_argument("--group-size", type=int, default=128)
    collect_i3.add_argument("--batch-size", type=int, default=128)
    collect_i3.add_argument("--fallback-batch-size", type=int, default=64)
    collect_i3.add_argument("--max-batch-tokens", type=int, default=4096)
    collect_i3.add_argument("--fallback-max-batch-tokens", type=int, default=2048)
    collect_i3.add_argument("--fallback-checkpoint-size", type=int, default=25)
    collect_i3.add_argument("--reuse-calibration-artifact")
    collect_i3.add_argument("--reuse-evaluation-probabilities")
    collect_i3.add_argument("--reuse-key", default="teacher")
    collect_i3.add_argument("--device-map", default="auto")
    collect_i3.add_argument("--dtype", default="auto")
    collect_i3.add_argument("--trust-remote-code", action="store_true")

    analyze_i3 = commands.add_parser(
        "analyze-cascade", help="evaluate frozen, recalibrated, and matched-rate cascade routers"
    )
    analyze_i3.add_argument("--artifact", required=True)
    analyze_i3.add_argument("--output", required=True)
    analyze_i3.add_argument("--target-rates", type=_comma_floats, default=(0.10, 0.20, 0.30, 0.50))
    analyze_i3.add_argument("--bootstrap-replicates", type=int, default=2000)
    analyze_i3.add_argument("--bootstrap-seed", type=int, default=20260818)

    replace_i3 = commands.add_parser(
        "replace-cascade-fallback", help="rescore only the fallback in a cascade artifact"
    )
    replace_i3.add_argument("--artifact", required=True)
    replace_i3.add_argument("--calibration-data", required=True)
    replace_i3.add_argument("--evaluation-data", required=True)
    replace_i3.add_argument("--fallback", required=True)
    replace_i3.add_argument("--batch-size", type=int, default=16)
    replace_i3.add_argument("--max-batch-tokens", type=int, default=512)
    replace_i3.add_argument("--device-map", default="auto")
    replace_i3.add_argument("--dtype", default="auto")
    replace_i3.add_argument("--trust-remote-code", action="store_true")

    score_gate_i3 = commands.add_parser(
        "score-cascade-gate", help="score a calibration file and test router viability"
    )
    score_gate_i3.add_argument("--data", required=True)
    score_gate_i3.add_argument("--model", required=True)
    score_gate_i3.add_argument("--output", required=True)
    score_gate_i3.add_argument("--probability-output", required=True)
    score_gate_i3.add_argument("--threshold", type=float, default=0.60)
    score_gate_i3.add_argument("--batch-size", type=int, default=64)
    score_gate_i3.add_argument("--max-batch-tokens", type=int, default=2048)
    score_gate_i3.add_argument("--device-map", default="auto")
    score_gate_i3.add_argument("--dtype", default="auto")
    score_gate_i3.add_argument("--trust-remote-code", action="store_true")

    drift_i3 = commands.add_parser(
        "analyze-cascade-drift", help="analyze routing drift from an SLM-only checkpoint"
    )
    drift_i3.add_argument("--checkpoint", required=True)
    drift_i3.add_argument("--output", required=True)
    drift_i3.add_argument("--target-rates", type=_comma_floats, default=(0.10, 0.20, 0.30, 0.50))
    drift_i3.add_argument("--bootstrap-replicates", type=int, default=2000)
    drift_i3.add_argument("--bootstrap-seed", type=int, default=20260818)

    collect_i4 = commands.add_parser(
        "collect-cascade-probes", help="collect one-layer probes for cascade-aware compression"
    )
    collect_i4.add_argument("--cascade-artifact", required=True)
    collect_i4.add_argument("--calibration-data", required=True)
    collect_i4.add_argument("--slm", required=True)
    collect_i4.add_argument("--output", required=True)
    collect_i4.add_argument("--bits", type=_comma_ints, default=(2, 4, 8))
    collect_i4.add_argument("--group-size", type=int, default=128)
    collect_i4.add_argument("--batch-size", type=int, default=128)
    collect_i4.add_argument("--device-map", default="auto")
    collect_i4.add_argument("--dtype", default="auto")
    collect_i4.add_argument("--trust-remote-code", action="store_true")

    analyze_i4 = commands.add_parser(
        "analyze-cascade-compression", help="derive KL/CUAQ/DBAQ matched-budget allocations"
    )
    analyze_i4.add_argument("--artifact", required=True)
    analyze_i4.add_argument("--output", required=True)
    analyze_i4.add_argument("--router-signal", choices=("entropy", "max_probability", "margin"), default="margin")
    analyze_i4.add_argument("--target-rates", type=_comma_floats, default=(0.1, 0.2, 0.3, 0.5))
    analyze_i4.add_argument("--cost-coefficients", type=_comma_floats, default=(0.0, 0.01, 0.025, 0.05, 0.1))
    analyze_i4.add_argument("--budgets", type=_comma_floats, default=(3.5, 4.0))
    analyze_i4.add_argument("--random-seeds", type=_comma_ints, default=tuple(range(10)))

    benchmark_i4 = commands.add_parser(
        "benchmark-cascade-compression", help="evaluate frozen cascade-aware compression policies"
    )
    benchmark_i4.add_argument("--data", required=True)
    benchmark_i4.add_argument("--cascade-artifact", required=True)
    benchmark_i4.add_argument("--probe-artifact", required=True)
    benchmark_i4.add_argument("--policies", required=True)
    benchmark_i4.add_argument("--slm", required=True)
    benchmark_i4.add_argument("--output-dir", required=True)
    benchmark_i4.add_argument("--methods", default="kl,cuaq")
    benchmark_i4.add_argument("--batch-size", type=int, default=128)
    benchmark_i4.add_argument("--bootstrap-replicates", type=int, default=2000)
    benchmark_i4.add_argument("--bootstrap-seed", type=int, default=20260819)
    benchmark_i4.add_argument("--device-map", default="auto")
    benchmark_i4.add_argument("--dtype", default="auto")
    benchmark_i4.add_argument("--trust-remote-code", action="store_true")

    search_i5 = commands.add_parser(
        "search-c3q", help="search a capability-safe neighborhood around KL allocations"
    )
    search_i5.add_argument("--calibration-data", required=True)
    search_i5.add_argument("--cascade-artifact", required=True)
    search_i5.add_argument("--probe-artifact", required=True)
    search_i5.add_argument("--policies", required=True)
    search_i5.add_argument("--slm", required=True)
    search_i5.add_argument("--output-dir", required=True)
    search_i5.add_argument("--budgets", type=_comma_floats, default=(3.5, 4.0))
    search_i5.add_argument("--candidates-per-budget", type=int, default=100)
    search_i5.add_argument("--random-candidates", type=int, default=15)
    search_i5.add_argument("--generation-seeds", type=_comma_ints, default=(11, 23, 37))
    search_i5.add_argument("--router-signal", choices=("entropy", "max_probability", "margin"), default="margin")
    search_i5.add_argument("--target-rates", type=_comma_floats, default=(0.1, 0.2, 0.3, 0.5))
    search_i5.add_argument("--cost-coefficients", type=_comma_floats, default=(0.0, 0.01, 0.025, 0.05, 0.1))
    search_i5.add_argument("--kl-tolerance", type=float, default=0.10)
    search_i5.add_argument("--accuracy-tolerance", type=float, default=0.01)
    search_i5.add_argument("--headroom-threshold", type=float, default=0.005)
    search_i5.add_argument("--screen-with-layer-probes", action="store_true")
    search_i5.add_argument("--exact-finalists", type=int, default=6)
    search_i5.add_argument("--batch-size", type=int, default=128)
    search_i5.add_argument("--device-map", default="auto")
    search_i5.add_argument("--dtype", default="auto")
    search_i5.add_argument("--trust-remote-code", action="store_true")

    benchmark_i5 = commands.add_parser(
        "benchmark-c3q", help="run the gated held-out C3Q ARC comparison"
    )
    benchmark_i5.add_argument("--evaluation-data", required=True)
    benchmark_i5.add_argument("--cascade-artifact", required=True)
    benchmark_i5.add_argument("--probe-artifact", required=True)
    benchmark_i5.add_argument("--search-results", required=True)
    benchmark_i5.add_argument("--slm", required=True)
    benchmark_i5.add_argument("--output-dir", required=True)
    benchmark_i5.add_argument("--batch-size", type=int, default=128)
    benchmark_i5.add_argument("--bootstrap-replicates", type=int, default=2000)
    benchmark_i5.add_argument("--bootstrap-seed", type=int, default=20260820)
    benchmark_i5.add_argument("--device-map", default="auto")
    benchmark_i5.add_argument("--dtype", default="auto")
    benchmark_i5.add_argument("--trust-remote-code", action="store_true")

    collect_i6 = commands.add_parser(
        "collect-sparse-hierarchy", help="collect activation-aware nested sparse-model outputs"
    )
    collect_i6.add_argument("--calibration-data", required=True)
    collect_i6.add_argument("--cascade-artifact", required=True)
    collect_i6.add_argument("--slm", required=True)
    collect_i6.add_argument("--output", required=True)
    collect_i6.add_argument("--masks-output", required=True)
    collect_i6.add_argument("--importance-output", required=True)
    collect_i6.add_argument("--medium-sparsity", type=float, default=0.25)
    collect_i6.add_argument("--aggressive-sparsity", type=float, default=0.50)
    collect_i6.add_argument("--collapse-margin", type=float, default=0.02)
    collect_i6.add_argument("--batch-size", type=int, default=128)
    collect_i6.add_argument("--device-map", default="auto")
    collect_i6.add_argument("--dtype", default="auto")
    collect_i6.add_argument("--trust-remote-code", action="store_true")

    analyze_i6 = commands.add_parser(
        "analyze-densification", help="cross-validate sparse-to-dense uplift estimators"
    )
    analyze_i6.add_argument("--artifact", required=True)
    analyze_i6.add_argument("--output", required=True)
    analyze_i6.add_argument("--probe-output", required=True)
    analyze_i6.add_argument("--folds", type=int, default=5)
    analyze_i6.add_argument("--bootstrap-replicates", type=int, default=2000)
    analyze_i6.add_argument("--bootstrap-seed", type=int, default=20260821)
    analyze_i6.add_argument("--target-rates", type=_comma_floats, default=(0.1, 0.2, 0.3, 0.5))

    score_i6 = commands.add_parser(
        "score-sparse-hierarchy", help="score frozen nested sparse models on held-out data"
    )
    score_i6.add_argument("--data", required=True)
    score_i6.add_argument("--cascade-artifact", required=True)
    score_i6.add_argument("--masks", required=True)
    score_i6.add_argument("--slm", required=True)
    score_i6.add_argument("--output", required=True)
    score_i6.add_argument("--batch-size", type=int, default=128)
    score_i6.add_argument("--device-map", default="auto")
    score_i6.add_argument("--dtype", default="auto")
    score_i6.add_argument("--trust-remote-code", action="store_true")

    benchmark_i6 = commands.add_parser(
        "benchmark-densification", help="evaluate the frozen two-level U-Dense policy"
    )
    benchmark_i6.add_argument("--analysis", required=True)
    benchmark_i6.add_argument("--probe", required=True)
    benchmark_i6.add_argument("--evaluation-artifact", required=True)
    benchmark_i6.add_argument("--output-dir", required=True)
    benchmark_i6.add_argument("--random-seeds", type=_comma_ints, default=tuple(range(10)))
    benchmark_i6.add_argument("--bootstrap-replicates", type=int, default=2000)
    benchmark_i6.add_argument("--bootstrap-seed", type=int, default=20260822)

    collect_i7 = commands.add_parser(
        "collect-capacity-hierarchy", help="add S40 to the nested ARC sparse hierarchy"
    )
    collect_i7.add_argument("--iteration6-artifact", required=True)
    collect_i7.add_argument("--importance", required=True)
    collect_i7.add_argument("--data", required=True)
    collect_i7.add_argument("--slm", required=True)
    collect_i7.add_argument("--output", required=True)
    collect_i7.add_argument("--masks-output", required=True)
    collect_i7.add_argument("--batch-size", type=int, default=128)
    collect_i7.add_argument("--device-map", default="auto")
    collect_i7.add_argument("--dtype", default="auto")
    collect_i7.add_argument("--trust-remote-code", action="store_true")

    oracle_i7 = commands.add_parser(
        "analyze-capacity-headroom", help="select a sparse start from oracle headroom"
    )
    oracle_i7.add_argument("--hierarchy", required=True)
    oracle_i7.add_argument("--output", required=True)
    oracle_i7.add_argument("--target-rates", type=_comma_floats, default=(0.1, 0.2, 0.3, 0.5))
    oracle_i7.add_argument("--minimum-gain-at-30", type=float, default=0.10)

    representations_i7 = commands.add_parser(
        "collect-capacity-representations", help="cache sparse prompt residual states"
    )
    representations_i7.add_argument("--hierarchy", required=True)
    representations_i7.add_argument("--oracle-analysis", required=True)
    representations_i7.add_argument("--masks", required=True)
    representations_i7.add_argument("--data", required=True)
    representations_i7.add_argument("--slm", required=True)
    representations_i7.add_argument("--output", required=True)
    representations_i7.add_argument("--layer-indices", type=_comma_ints, default=(6, 13, 20, 27))
    representations_i7.add_argument("--batch-size", type=int, default=128)
    representations_i7.add_argument("--device-map", default="auto")
    representations_i7.add_argument("--dtype", default="auto")
    representations_i7.add_argument("--trust-remote-code", action="store_true")

    analyze_i7 = commands.add_parser(
        "analyze-counterfactual-routing", help="cross-validate capacity-advantage routers"
    )
    analyze_i7.add_argument("--hierarchy", required=True)
    analyze_i7.add_argument("--oracle-analysis", required=True)
    analyze_i7.add_argument("--representations", required=True)
    analyze_i7.add_argument("--output", required=True)
    analyze_i7.add_argument("--probe-output", required=True)
    analyze_i7.add_argument("--folds", type=int, default=5)
    analyze_i7.add_argument("--ridge-alpha", type=float, default=1.0)
    analyze_i7.add_argument("--target-rates", type=_comma_floats, default=(0.1, 0.2, 0.3, 0.5))
    analyze_i7.add_argument("--random-seeds", type=_comma_ints, default=tuple(range(10)))
    analyze_i7.add_argument("--bootstrap-replicates", type=int, default=2000)
    analyze_i7.add_argument("--bootstrap-seed", type=int, default=20260824)

    collect_i8 = commands.add_parser(
        "collect-capacity-contrast", help="cache paired S50/S25 capacity-contrast supervision"
    )
    collect_i8.add_argument("--data", required=True)
    collect_i8.add_argument("--masks", required=True)
    collect_i8.add_argument("--slm", required=True)
    collect_i8.add_argument("--output", required=True)
    collect_i8.add_argument("--layer-indices", type=_comma_ints, default=(6, 13, 20, 27))
    collect_i8.add_argument("--batch-size", type=int, default=64)
    collect_i8.add_argument("--device-map", default="auto")
    collect_i8.add_argument("--dtype", default="auto")
    collect_i8.add_argument("--trust-remote-code", action="store_true")

    analyze_i8 = commands.add_parser(
        "analyze-capacity-contrast", help="train and gate Capacity-Contrast Distillation routers"
    )
    analyze_i8.add_argument("--contrast-artifact", required=True)
    analyze_i8.add_argument("--arc-hierarchy", required=True)
    analyze_i8.add_argument("--arc-representations", required=True)
    analyze_i8.add_argument("--iteration7-probes", required=True)
    analyze_i8.add_argument("--output", required=True)
    analyze_i8.add_argument("--scores-output", required=True)
    analyze_i8.add_argument("--models-output", required=True)
    analyze_i8.add_argument("--validation-fraction", type=float, default=0.2)
    analyze_i8.add_argument("--training-seed", type=int, default=20260818)
    analyze_i8.add_argument("--target-rates", type=_comma_floats, default=(0.1, 0.2, 0.3, 0.5))
    analyze_i8.add_argument("--random-seeds", type=_comma_ints, default=tuple(range(10)))
    analyze_i8.add_argument("--bootstrap-replicates", type=int, default=2000)
    analyze_i8.add_argument("--bootstrap-seed", type=int, default=20260825)
    analyze_i8.add_argument("--device", default="auto")

    freeze_i9 = commands.add_parser(
        "freeze-ccr", help="export and verify the pre-registered Iteration 8 linear CCR router"
    )
    freeze_i9.add_argument("--paired-artifact", required=True)
    freeze_i9.add_argument("--iteration8-models", required=True)
    freeze_i9.add_argument("--iteration8-scores", required=True)
    freeze_i9.add_argument("--iteration8-analysis", required=True)
    freeze_i9.add_argument("--output", required=True)

    dedup_i9 = commands.add_parser(
        "prepare-ccr-evaluation", help="freeze an outcome-blind de-duplicated evaluation cohort"
    )
    dedup_i9.add_argument("--training-data", required=True)
    dedup_i9.add_argument("--evaluation-data", required=True)
    dedup_i9.add_argument("--output", required=True)
    dedup_i9.add_argument("--manifest", required=True)

    collect_i9 = commands.add_parser(
        "collect-ccr-evaluation", help="cache sequential S50, dense, and diagnostic S25 evaluation outputs"
    )
    collect_i9.add_argument("--data", required=True)
    collect_i9.add_argument("--masks", required=True)
    collect_i9.add_argument("--frozen-router", required=True)
    collect_i9.add_argument("--slm", required=True)
    collect_i9.add_argument("--output", required=True)
    collect_i9.add_argument("--batch-size", type=int, default=64)
    collect_i9.add_argument("--device-map", default="auto")
    collect_i9.add_argument("--dtype", default="auto")
    collect_i9.add_argument("--trust-remote-code", action="store_true")
    collect_i9.add_argument("--skip-s25", action="store_true")

    analyze_i9 = commands.add_parser(
        "analyze-ccr-system", help="evaluate frozen CCR at exact matched dense rates"
    )
    analyze_i9.add_argument("--evaluation-artifact", required=True)
    analyze_i9.add_argument("--paired-artifact", required=True)
    analyze_i9.add_argument("--iteration8-scores", required=True)
    analyze_i9.add_argument("--output", required=True)
    analyze_i9.add_argument("--scores-output", required=True)
    analyze_i9.add_argument("--target-rates", type=_comma_floats, default=(0.1, 0.2, 0.3, 0.5))
    analyze_i9.add_argument("--random-seeds", type=_comma_ints, default=tuple(range(20)))
    analyze_i9.add_argument("--bootstrap-replicates", type=int, default=5000)
    analyze_i9.add_argument("--bootstrap-seed", type=int, default=20260826)

    collect_i10 = commands.add_parser(
        "collect-ursa-masks", help="build matched mean-activation and uncertainty-robust FFN masks"
    )
    collect_i10.add_argument("--data", required=True)
    collect_i10.add_argument("--slm", required=True)
    collect_i10.add_argument("--output", required=True)
    collect_i10.add_argument("--baseline-masks-output", required=True)
    collect_i10.add_argument("--ursa-masks-output", required=True)
    collect_i10.add_argument("--index-artifact")
    collect_i10.add_argument("--index-key", default="train_indices")
    collect_i10.add_argument("--strata", type=int, default=3)
    collect_i10.add_argument("--temperature", type=float, default=0.05)
    collect_i10.add_argument("--refresh-interval", type=int, default=64)
    collect_i10.add_argument("--medium-sparsity", type=float, default=0.25)
    collect_i10.add_argument("--aggressive-sparsity", type=float, default=0.50)
    collect_i10.add_argument("--batch-size", type=int, default=64)
    collect_i10.add_argument("--device-map", default="auto")
    collect_i10.add_argument("--dtype", default="auto")
    collect_i10.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    collect_i10.add_argument("--bnb-4bit-compute-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    collect_i10.add_argument("--trust-remote-code", action="store_true")

    score_i10 = commands.add_parser(
        "score-ursa-comparison", help="score matched baseline and URSA sparse models"
    )
    score_i10.add_argument("--data", required=True)
    score_i10.add_argument("--slm", required=True)
    score_i10.add_argument("--baseline-masks", required=True)
    score_i10.add_argument("--ursa-masks", required=True)
    score_i10.add_argument("--output", required=True)
    score_i10.add_argument("--index-artifact")
    score_i10.add_argument("--index-key", default="validation_indices")
    score_i10.add_argument("--batch-size", type=int, default=64)
    score_i10.add_argument("--device-map", default="auto")
    score_i10.add_argument("--dtype", default="auto")
    score_i10.add_argument("--trust-remote-code", action="store_true")

    analyze_i10 = commands.add_parser(
        "analyze-ursa-comparison", help="compare URSA with matched structured-pruning masks"
    )
    analyze_i10.add_argument("--artifact", required=True)
    analyze_i10.add_argument("--output", required=True)
    analyze_i10.add_argument("--strata", type=int, default=3)
    analyze_i10.add_argument("--conformal-alpha", type=float, default=0.1)
    analyze_i10.add_argument("--conformal-fraction", type=float, default=0.5)
    analyze_i10.add_argument("--split-seed", type=int, default=20260827)
    analyze_i10.add_argument("--bootstrap-replicates", type=int, default=5000)
    analyze_i10.add_argument("--bootstrap-seed", type=int, default=20260828)

    train_i11 = commands.add_parser(
        "train-upgd-masks", help="learn exact-budget stochastic FFN gates with uncertainty distillation"
    )
    train_i11.add_argument("--data", required=True)
    train_i11.add_argument("--calibration-artifact", required=True)
    train_i11.add_argument("--slm", required=True)
    train_i11.add_argument("--output", required=True)
    train_i11.add_argument("--masks-output", required=True)
    train_i11.add_argument("--sparsity", type=float, default=0.50)
    train_i11.add_argument("--steps", type=int, default=600)
    train_i11.add_argument("--per-stratum", type=int, default=1)
    train_i11.add_argument("--learning-rate", type=float, default=0.05)
    train_i11.add_argument("--start-temperature", type=float, default=0.5)
    train_i11.add_argument("--end-temperature", type=float, default=0.1)
    train_i11.add_argument("--start-noise", type=float, default=0.5)
    train_i11.add_argument("--end-noise", type=float, default=0.05)
    train_i11.add_argument("--entropy-weight", type=float, default=1.0)
    train_i11.add_argument("--margin-weight", type=float, default=0.5)
    train_i11.add_argument("--robust-weight", type=float, default=0.5)
    train_i11.add_argument("--robust-temperature", type=float, default=0.1)
    train_i11.add_argument("--objective-name", default="full_upgd")
    train_i11.add_argument("--seed", type=int, default=20260829)
    train_i11.add_argument("--device-map", default="auto")
    train_i11.add_argument("--dtype", default="auto")
    train_i11.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    train_i11.add_argument("--bnb-4bit-compute-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    train_i11.add_argument("--trust-remote-code", action="store_true")

    score_i11 = commands.add_parser(
        "score-pruned-mask", help="physically apply and score one learned FFN-channel mask"
    )
    score_i11.add_argument("--data", required=True)
    score_i11.add_argument("--slm", required=True)
    score_i11.add_argument("--masks", required=True)
    score_i11.add_argument("--mask-key", default="retained")
    score_i11.add_argument("--output", required=True)
    score_i11.add_argument("--index-artifact")
    score_i11.add_argument("--index-key", default="validation_indices")
    score_i11.add_argument("--batch-size", type=int, default=64)
    score_i11.add_argument("--device-map", default="auto")
    score_i11.add_argument("--dtype", default="auto")
    score_i11.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    score_i11.add_argument("--bnb-4bit-compute-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    score_i11.add_argument("--mask-mode", choices=("physical", "hook"), default="physical")
    score_i11.add_argument("--trust-remote-code", action="store_true")

    analyze_i11 = commands.add_parser(
        "analyze-upgd-comparison", help="evaluate UPGD against the matched physical-pruning baseline"
    )
    analyze_i11.add_argument("--reference-artifact", required=True)
    analyze_i11.add_argument("--baseline-artifact")
    analyze_i11.add_argument("--upgd-artifact", required=True)
    analyze_i11.add_argument("--output", required=True)
    analyze_i11.add_argument("--strata", type=int, default=3)
    analyze_i11.add_argument("--conformal-alpha", type=float, default=0.1)
    analyze_i11.add_argument("--conformal-fraction", type=float, default=0.5)
    analyze_i11.add_argument("--split-seed", type=int, default=20260827)
    analyze_i11.add_argument("--bootstrap-replicates", type=int, default=5000)
    analyze_i11.add_argument("--bootstrap-seed", type=int, default=20260830)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "collect":
        artifact = collect_calibration(
            data_path=args.data,
            teacher_model=args.teacher,
            student_model=args.student,
            output_path=args.output,
            probe_bits=args.probe_bits,
            group_size=args.group_size,
            limit=args.limit,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"saved {len(artifact.layer_names)} layers to {args.output}")
    elif args.command == "analyze":
        artifact = CalibrationArtifact.load(args.artifact)
        report = analyze_artifact(artifact, budgets=args.budgets, bit_choices=args.bits)
        save_report(report, args.output)
        print(f"saved report to {args.output}")
    elif args.command == "metrics":
        artifact = CalibrationArtifact.load(args.artifact)
        if (artifact.labels < 0).any():
            raise SystemExit("metrics require an answer on every JSONL example")
        output = {"full_precision": classification_metrics(artifact.student_probabilities, artifact.labels)}
        for index, name in enumerate(artifact.layer_names):
            output[f"probe:{name}"] = classification_metrics(
                artifact.quantized_probabilities[index], artifact.labels
            )
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=True))
    elif args.command == "evaluate":
        artifact = CalibrationArtifact.load(args.artifact)
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        result = evaluate_policy(
            data_path=args.data,
            student_model=args.student,
            artifact=artifact,
            report=report,
            mode=args.mode,
            budget=args.budget,
            output_path=args.output,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    elif args.command == "benchmark":
        artifact = CalibrationArtifact.load(args.artifact)
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        result = benchmark_policies(
            data_path=args.data,
            student_model=args.student,
            artifact=artifact,
            report=report,
            output_dir=args.output_dir,
            modes=tuple(item.strip() for item in args.modes.split(",") if item.strip()),
            budgets=args.budgets,
            random_seeds=args.random_seeds,
            include_baselines=not args.skip_baselines,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    elif args.command == "collect-iteration":
        artifact = collect_multibit_calibration(
            data_path=args.data,
            teacher_model=args.teacher,
            student_model=args.student,
            output_path=args.output,
            probe_bits=args.bits,
            group_size=args.group_size,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"saved {len(artifact.probe_bits)} widths x {len(artifact.layer_names)} layers to {args.output}")
    elif args.command == "analyze-iteration":
        artifact = MultiBitCalibrationArtifact.load(args.artifact)
        result = analyze_iteration(
            artifact,
            output_path=args.output,
            subset_size=args.subset_size,
            calibration_seeds=args.calibration_seeds,
            budgets=args.budgets,
            interaction_lambda=args.interaction_lambda,
        )
        print(json.dumps({"output": args.output, "calibration_seeds": result["calibration_seeds"]}, indent=2))
    elif args.command == "benchmark-iteration":
        artifact = MultiBitCalibrationArtifact.load(args.artifact)
        policies = json.loads(Path(args.report).read_text(encoding="utf-8"))
        result = benchmark_iteration(
            data_path=args.data,
            teacher_model=args.teacher,
            student_model=args.student,
            artifact=artifact,
            policies=policies,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output_dir": args.output_dir, "unique_allocations": result["unique_allocations_evaluated"]}, indent=2))
    elif args.command == "analyze-trust":
        artifact = MultiBitCalibrationArtifact.load(args.artifact)
        result = analyze_trust_layers(
            artifact,
            output_path=args.output,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output": args.output, "gate": result["gate"]}, indent=2))
    elif args.command == "build-trust-policies":
        artifact = MultiBitCalibrationArtifact.load(args.artifact)
        analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
        result = build_trust_policies(
            artifact, analysis, output_path=args.output, budgets=args.budgets
        )
        print(json.dumps({"output": args.output, "budgets": result["budgets"]}, indent=2))
    elif args.command == "benchmark-trust":
        artifact = MultiBitCalibrationArtifact.load(args.artifact)
        policies = json.loads(Path(args.report).read_text(encoding="utf-8"))
        result = benchmark_trust_policies(
            data_path=args.data,
            student_model=args.student,
            artifact=artifact,
            policies=policies,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "unique_allocations": result["unique_allocations_evaluated"],
                },
                indent=2,
            )
        )
    elif args.command == "cascade-gate":
        artifact = MultiBitCalibrationArtifact.load(args.artifact)
        probabilities = (
            artifact.student_probabilities
            if args.role == "student"
            else artifact.teacher_probabilities
        )
        result = evaluate_router_gate(
            probabilities,
            artifact.labels,
            output_path=args.output,
            model=args.model,
            threshold=args.threshold,
        )
        print(json.dumps({"output": args.output, "selected_signal": result["selected_signal"], "gate": result["gate"]}, indent=2))
    elif args.command == "collect-cascade":
        artifact = collect_cascade_probabilities(
            calibration_data_path=args.calibration_data,
            evaluation_data_path=args.evaluation_data,
            slm_model=args.slm,
            fallback_model=args.fallback,
            output_path=args.output,
            bits=args.bits,
            group_size=args.group_size,
            batch_size=args.batch_size,
            fallback_batch_size=args.fallback_batch_size,
            max_batch_tokens=args.max_batch_tokens,
            fallback_max_batch_tokens=args.fallback_max_batch_tokens,
            fallback_checkpoint_size=args.fallback_checkpoint_size,
            reuse_calibration_artifact=args.reuse_calibration_artifact,
            reuse_evaluation_probabilities=args.reuse_evaluation_probabilities,
            reuse_key=args.reuse_key,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "bits": artifact.bits, "evaluation_examples": len(artifact.evaluation_labels)}, indent=2))
    elif args.command == "analyze-cascade":
        artifact = CascadeProbabilityArtifact.load(args.artifact)
        result = analyze_cascade(
            artifact,
            output_path=args.output,
            target_rates=args.target_rates,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output": args.output, "decision": result["decision"]}, indent=2))
    elif args.command == "replace-cascade-fallback":
        artifact = replace_cascade_fallback(
            artifact_path=args.artifact,
            calibration_data_path=args.calibration_data,
            evaluation_data_path=args.evaluation_data,
            fallback_model=args.fallback,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"artifact": args.artifact, "fallback": artifact.metadata["fallback_model"]}, indent=2))
    elif args.command == "score-cascade-gate":
        result = score_router_gate(
            data_path=args.data,
            model=args.model,
            output_path=args.output,
            probability_output=args.probability_output,
            threshold=args.threshold,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "selected_signal": result["selected_signal"], "gate": result["gate"]}, indent=2))
    elif args.command == "analyze-cascade-drift":
        result = analyze_cascade_drift_checkpoint(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            target_rates=args.target_rates,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output": args.output, "status": result["status"], "router": result["router"]}, indent=2))
    elif args.command == "collect-cascade-probes":
        artifact = collect_cascade_layer_probes(
            cascade_artifact_path=args.cascade_artifact,
            calibration_data_path=args.calibration_data,
            slm_model=args.slm,
            output_path=args.output,
            bits=args.bits,
            group_size=args.group_size,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "bits": artifact.probe_bits, "layers": len(artifact.layer_names)}, indent=2))
    elif args.command == "analyze-cascade-compression":
        artifact = MultiBitCalibrationArtifact.load(args.artifact)
        result = analyze_cascade_compression(
            artifact,
            output_path=args.output,
            router_signal=args.router_signal,
            target_rates=args.target_rates,
            cost_coefficients=args.cost_coefficients,
            budgets=args.budgets,
            random_seeds=args.random_seeds,
        )
        print(json.dumps({"output": args.output, "stage": result["stage"], "budgets": result["scope"]["budgets"]}, indent=2))
    elif args.command == "benchmark-cascade-compression":
        artifact = MultiBitCalibrationArtifact.load(args.probe_artifact)
        policies = json.loads(Path(args.policies).read_text(encoding="utf-8"))
        methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
        result = benchmark_cascade_compression(
            data_path=args.data,
            cascade_artifact_path=args.cascade_artifact,
            probe_artifact=artifact,
            policies=policies,
            slm_model=args.slm,
            output_dir=args.output_dir,
            methods=methods,
            batch_size=args.batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output_dir": args.output_dir, "gate": result["gate"], "configurations": len(result["configurations"])}, indent=2))
    elif args.command == "search-c3q":
        artifact = MultiBitCalibrationArtifact.load(args.probe_artifact)
        policies = json.loads(Path(args.policies).read_text(encoding="utf-8"))
        result = search_c3q(
            calibration_data_path=args.calibration_data,
            cascade_artifact_path=args.cascade_artifact,
            probe_artifact=artifact,
            policies=policies,
            slm_model=args.slm,
            output_dir=args.output_dir,
            budgets=args.budgets,
            candidates_per_budget=args.candidates_per_budget,
            random_candidates=args.random_candidates,
            generation_seeds=args.generation_seeds,
            router_signal=args.router_signal,
            target_rates=args.target_rates,
            cost_coefficients=args.cost_coefficients,
            kl_tolerance=args.kl_tolerance,
            accuracy_tolerance=args.accuracy_tolerance,
            headroom_threshold=args.headroom_threshold,
            screen_with_layer_probes=args.screen_with_layer_probes,
            exact_finalists=args.exact_finalists,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output_dir": args.output_dir, "gate": result["gate"]}, indent=2))
    elif args.command == "benchmark-c3q":
        artifact = MultiBitCalibrationArtifact.load(args.probe_artifact)
        search = json.loads(Path(args.search_results).read_text(encoding="utf-8"))
        result = benchmark_c3q(
            evaluation_data_path=args.evaluation_data,
            cascade_artifact_path=args.cascade_artifact,
            probe_artifact=artifact,
            search_results=search,
            slm_model=args.slm,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output_dir": args.output_dir, "gate": result["gate"]}, indent=2))
    elif args.command == "collect-sparse-hierarchy":
        result = collect_sparse_hierarchy(
            calibration_data_path=args.calibration_data,
            cascade_artifact_path=args.cascade_artifact,
            slm_model=args.slm,
            output_path=args.output,
            masks_path=args.masks_output,
            importance_path=args.importance_output,
            medium_sparsity=args.medium_sparsity,
            aggressive_sparsity=args.aggressive_sparsity,
            collapse_margin=args.collapse_margin,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result["metadata"]}, indent=2))
    elif args.command == "analyze-densification":
        result = analyze_densification(
            args.artifact,
            output_path=args.output,
            probe_path=args.probe_output,
            folds=args.folds,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            target_rates=args.target_rates,
        )
        print(json.dumps({"output": args.output, "gate": result["gate"]}, indent=2))
    elif args.command == "score-sparse-hierarchy":
        result = score_sparse_hierarchy(
            data_path=args.data,
            cascade_artifact_path=args.cascade_artifact,
            masks_path=args.masks,
            slm_model=args.slm,
            output_path=args.output,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result["metadata"]}, indent=2))
    elif args.command == "benchmark-densification":
        result = benchmark_densification(
            calibration_analysis_path=args.analysis,
            probe_path=args.probe,
            evaluation_artifact_path=args.evaluation_artifact,
            output_dir=args.output_dir,
            random_seeds=args.random_seeds,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output_dir": args.output_dir, "gate": result["gate"]}, indent=2))
    elif args.command == "collect-capacity-hierarchy":
        result = collect_capacity_hierarchy(
            iteration6_artifact_path=args.iteration6_artifact,
            importance_path=args.importance,
            data_path=args.data,
            slm_model=args.slm,
            output_path=args.output,
            masks_path=args.masks_output,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result["metadata"]}, indent=2))
    elif args.command == "analyze-capacity-headroom":
        result = analyze_oracle_headroom(
            args.hierarchy,
            output_path=args.output,
            target_rates=args.target_rates,
            minimum_gain_at_30=args.minimum_gain_at_30,
        )
        print(json.dumps({"output": args.output, "gate": result["gate_a"], "selection": result["selection"]}, indent=2))
    elif args.command == "collect-capacity-representations":
        result = collect_capacity_representations(
            hierarchy_path=args.hierarchy,
            oracle_analysis_path=args.oracle_analysis,
            masks_path=args.masks,
            data_path=args.data,
            slm_model=args.slm,
            output_path=args.output,
            layer_indices=args.layer_indices,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result["metadata"]}, indent=2))
    elif args.command == "analyze-counterfactual-routing":
        result = analyze_counterfactual_routing(
            hierarchy_path=args.hierarchy,
            oracle_analysis_path=args.oracle_analysis,
            representations_path=args.representations,
            output_path=args.output,
            probe_output_path=args.probe_output,
            folds=args.folds,
            ridge_alpha=args.ridge_alpha,
            target_rates=args.target_rates,
            random_seeds=args.random_seeds,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output": args.output, "gate": result["gate_b"]}, indent=2))
    elif args.command == "collect-capacity-contrast":
        result = collect_capacity_contrast(
            data_path=args.data,
            masks_path=args.masks,
            slm_model=args.slm,
            output_path=args.output,
            layer_indices=args.layer_indices,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result["metadata"]}, indent=2))
    elif args.command == "analyze-capacity-contrast":
        result = analyze_capacity_contrast(
            contrast_artifact_path=args.contrast_artifact,
            arc_hierarchy_path=args.arc_hierarchy,
            arc_representations_path=args.arc_representations,
            iteration7_probes_path=args.iteration7_probes,
            output_path=args.output,
            scores_output_path=args.scores_output,
            models_output_path=args.models_output,
            validation_fraction=args.validation_fraction,
            training_seed=args.training_seed,
            target_rates=args.target_rates,
            random_seeds=args.random_seeds,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            device=args.device,
        )
        print(json.dumps({"output": args.output, "gate_1": result["gate_1"], "gate_2": result["gate_2"], "gate_3": result["gate_3"]}, indent=2))
    elif args.command == "freeze-ccr":
        result = freeze_iteration8_linear_ccr(
            paired_artifact_path=args.paired_artifact,
            iteration8_models_path=args.iteration8_models,
            iteration8_scores_path=args.iteration8_scores,
            iteration8_analysis_path=args.iteration8_analysis,
            output_path=args.output,
        )
        print(json.dumps({"output": args.output, "metadata": result["metadata"]}, indent=2))
    elif args.command == "prepare-ccr-evaluation":
        result = prepare_deduplicated_evaluation(
            training_data_path=args.training_data,
            evaluation_data_path=args.evaluation_data,
            output_path=args.output,
            manifest_path=args.manifest,
        )
        print(json.dumps({"output": args.output, "manifest": args.manifest, "counts": {
            "checked": result["evaluation_questions_checked"],
            "removed": result["exact_duplicates_removed"] + result["near_duplicates_removed"],
            "final": result["final_evaluation_size"],
        }}, indent=2))
    elif args.command == "collect-ccr-evaluation":
        result = collect_ccr_evaluation(
            data_path=args.data, masks_path=args.masks, frozen_router_path=args.frozen_router,
            slm_model=args.slm, output_path=args.output, batch_size=args.batch_size,
            device_map=args.device_map, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
            include_s25=not args.skip_s25,
        )
        print(json.dumps({"output": args.output, "metadata": result["metadata"]}, indent=2))
    elif args.command == "analyze-ccr-system":
        result = analyze_ccr_system(
            evaluation_artifact_path=args.evaluation_artifact,
            paired_artifact_path=args.paired_artifact,
            iteration8_scores_path=args.iteration8_scores,
            output_path=args.output, scores_output_path=args.scores_output,
            target_rates=args.target_rates, random_seeds=args.random_seeds,
            bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output": args.output, "gate": result["gate"]}, indent=2))
    elif args.command == "collect-ursa-masks":
        result = collect_ursa_masks(
            data_path=args.data,
            slm_model=args.slm,
            output_path=args.output,
            baseline_masks_path=args.baseline_masks_output,
            ursa_masks_path=args.ursa_masks_output,
            index_artifact_path=args.index_artifact,
            index_key=args.index_key,
            strata=args.strata,
            temperature=args.temperature,
            refresh_interval=args.refresh_interval,
            medium_sparsity=args.medium_sparsity,
            aggressive_sparsity=args.aggressive_sparsity,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            quantization=args.quantization,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result}, indent=2))
    elif args.command == "score-ursa-comparison":
        result = score_ursa_comparison(
            data_path=args.data,
            slm_model=args.slm,
            baseline_masks_path=args.baseline_masks,
            ursa_masks_path=args.ursa_masks,
            output_path=args.output,
            index_artifact_path=args.index_artifact,
            index_key=args.index_key,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result}, indent=2))
    elif args.command == "analyze-ursa-comparison":
        result = analyze_ursa_comparison(
            args.artifact,
            output_path=args.output,
            strata=args.strata,
            conformal_alpha=args.conformal_alpha,
            conformal_fraction=args.conformal_fraction,
            split_seed=args.split_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output": args.output, "gate": result["gate"]}, indent=2))
    elif args.command == "train-upgd-masks":
        result = train_upgd_masks(
            data_path=args.data,
            calibration_artifact_path=args.calibration_artifact,
            slm_model=args.slm,
            output_path=args.output,
            masks_output_path=args.masks_output,
            sparsity=args.sparsity,
            steps=args.steps,
            per_stratum=args.per_stratum,
            learning_rate=args.learning_rate,
            start_temperature=args.start_temperature,
            end_temperature=args.end_temperature,
            start_noise=args.start_noise,
            end_noise=args.end_noise,
            entropy_weight=args.entropy_weight,
            margin_weight=args.margin_weight,
            robust_weight=args.robust_weight,
            robust_temperature=args.robust_temperature,
            objective_name=args.objective_name,
            seed=args.seed,
            device_map=args.device_map,
            dtype=args.dtype,
            quantization=args.quantization,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result}, indent=2))
    elif args.command == "score-pruned-mask":
        result = score_pruned_mask(
            data_path=args.data,
            slm_model=args.slm,
            masks_path=args.masks,
            mask_key=args.mask_key,
            output_path=args.output,
            index_artifact_path=args.index_artifact,
            index_key=args.index_key,
            batch_size=args.batch_size,
            device_map=args.device_map,
            dtype=args.dtype,
            quantization=args.quantization,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            mask_mode=args.mask_mode,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps({"output": args.output, "metadata": result}, indent=2))
    elif args.command == "analyze-upgd-comparison":
        result = analyze_upgd_comparison(
            reference_artifact_path=args.reference_artifact,
            baseline_artifact_path=args.baseline_artifact,
            upgd_artifact_path=args.upgd_artifact,
            output_path=args.output,
            strata=args.strata,
            conformal_alpha=args.conformal_alpha,
            conformal_fraction=args.conformal_fraction,
            split_seed=args.split_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps({"output": args.output, "gate": result["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
