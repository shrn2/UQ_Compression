"""Independent integrity checks for Iteration 7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from tuac.counterfactual_routing import (
    load_capacity_hierarchy,
    load_representation_artifact,
    pairwise_ranking_accuracy,
)
from tuac.data import load_jsonl
from tuac.metrics import (
    binary_auroc,
    binary_average_precision,
    classification_metrics,
    spearman_correlation,
)
from tuac.scoring import kl_divergence


ROOT = Path("runs/iter7_arc")


def close(left, right) -> bool:
    return bool(np.isclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=True))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def top_k(scores: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, min(len(scores), int(round(rate * len(scores)))))
    order = np.argsort(-np.asarray(scores), kind="stable")
    result = np.zeros(len(scores), dtype=bool)
    result[order[:count]] = True
    return result


def main() -> int:
    data = load_jsonl("runs/iter1_arc/calibration_pool.jsonl")
    hierarchy = load_capacity_hierarchy(ROOT / "hierarchy.npz")
    representations = load_representation_artifact(ROOT / "representations.npz")
    oracle = json.loads((ROOT / "oracle_analysis.json").read_text(encoding="utf-8"))
    development = json.loads((ROOT / "development_analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "report/artifact.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "masks.npz", allow_pickle=False) as saved:
        masks = {key: saved[key] for key in saved.files}
    with np.load(ROOT / "probes.npz", allow_pickle=False) as saved:
        probes = {key: saved[key] for key in saved.files}

    checks = {
        "arc_train_count_750": len(data) == 750,
        "hierarchy_ids_match": hierarchy["example_ids"] == tuple(item.id for item in data),
        "representation_ids_match": representations["example_ids"] == hierarchy["example_ids"],
        "labels_match": np.array_equal(hierarchy["labels"], representations["labels"]),
        "four_capacity_variants": all(hierarchy[name].shape == (750, 5) for name in ("dense", "s25", "s40", "s50")),
        "s25_width": masks["s25"].shape == (28, 6720),
        "s40_width": masks["s40"].shape == (28, 5376),
        "s50_width": masks["s50"].shape == (28, 4480),
        "representation_shape": representations["hidden_states"].shape == (750, 4, 1536),
        "representation_layers": representations["layer_indices"].tolist() == [6, 13, 20, 27],
        "prompt_only_sparse_representation": representations["metadata"]["sparse_only_forward"] is True,
        "representation_reference_accepted": representations["metadata"]["reference_check"]["mean_absolute_probability_difference"] <= 0.02 and representations["metadata"]["reference_check"]["prediction_agreement"] >= 0.97,
        "gate_a_passed": oracle["gate_a"]["passed"] is True,
        "s50_selected": oracle["selection"]["selected_sparse_variant"] == "s50",
        "gate_b_failed": development["gate_b"]["passed"] is False,
        "summary_no_go": summary["decision"]["label"] == "NO-GO",
        "arc_validation_not_reused": summary["scope"]["arc_validation_reused"] is False and not (ROOT / "evaluation.npz").exists(),
        "mmlu_not_run": summary["decision"]["mmlu_pro_run"] is False and not Path("runs/iter7_mmlu_pro").exists(),
        "runtime_not_run": summary["decision"]["runtime_run"] is False and not (ROOT / "runtime").exists(),
        "report_html_exists": (ROOT / "report/report.html").is_file(),
        "report_html_nontrivial": (ROOT / "report/report.html").stat().st_size > 50_000,
        "report_five_charts": len(report["manifest"]["charts"]) == 5,
        "report_two_tables": len(report["manifest"]["tables"]) == 2,
        "chart_map_exists": (ROOT / "report/chart_map.json").is_file(),
    }
    for layer, (s25, s40, s50) in enumerate(zip(masks["s25"], masks["s40"], masks["s50"])):
        checks[f"nested_s50_s40_{layer}"] = np.isin(s50, s40).all()
        checks[f"nested_s40_s25_{layer}"] = np.isin(s40, s25).all()
        checks[f"unique_s25_{layer}"] = len(np.unique(s25)) == len(s25)
        checks[f"unique_s40_{layer}"] = len(np.unique(s40)) == len(s40)
        checks[f"unique_s50_{layer}"] = len(np.unique(s50)) == len(s50)

    for variant in ("dense", "s25", "s40", "s50"):
        probabilities = hierarchy[variant]
        checks[f"{variant}_finite"] = np.isfinite(probabilities).all()
        checks[f"{variant}_nonnegative"] = (probabilities >= 0).all()
        checks[f"{variant}_rows_sum"] = np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    static_lookup = {row["variant"]: row for row in oracle["static_rows"]}
    for variant in ("dense", "s25", "s40", "s50"):
        observed = classification_metrics(hierarchy[variant], hierarchy["labels"])
        for metric, value in observed.items():
            checks[f"static_{variant}_{metric}"] = close(value, static_lookup[variant][metric])
        checks[f"static_{variant}_kl"] = close(
            np.mean(kl_divergence(hierarchy["dense"], hierarchy[variant])),
            static_lookup[variant]["output_kl_from_dense"],
        )

    labels = hierarchy["labels"]
    dense_correct_hierarchy = hierarchy["dense"].argmax(axis=1) == labels
    oracle_lookup = {(row["variant"], row["target_rate"]): row for row in oracle["oracle_rows"]}
    for variant in ("s25", "s40", "s50"):
        sparse_correct = hierarchy[variant].argmax(axis=1) == labels
        advantage = dense_correct_hierarchy.astype(np.int8) - sparse_correct.astype(np.int8)
        for rate in oracle["scope"]["target_rates"]:
            decisions = top_k(advantage, float(rate))
            gain = float(np.mean(advantage * decisions))
            row = oracle_lookup[(variant, float(rate))]
            checks[f"oracle_{variant}_{rate}_gain"] = close(gain, row["oracle_gain"])
            checks[f"oracle_{variant}_{rate}_accuracy"] = close(sparse_correct.mean() + gain, row["oracle_accuracy"])
            checks[f"oracle_{variant}_{rate}_efficiency"] = close(gain / decisions.mean(), row["oracle_efficiency"])

    sparse = representations["selected_probabilities"]
    dense = hierarchy["dense"]
    sparse_correct = sparse.argmax(axis=1) == labels
    dense_correct = dense.argmax(axis=1) == labels
    signed = dense_correct.astype(np.int8) - sparse_correct.astype(np.int8)
    loss_advantage = np.log(
        np.clip(dense[np.arange(len(labels)), labels], 1e-12, 1.0)
        / np.clip(sparse[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    checks["positive_target_count"] = int(np.sum(signed == 1)) == development["targets"]["positive_advantage_count"]
    checks["neutral_target_count"] = int(np.sum(signed == 0)) == development["targets"]["neutral_advantage_count"]
    checks["harmful_target_count"] = int(np.sum(signed == -1)) == development["targets"]["harmful_advantage_count"]
    checks["loss_target_mean"] = close(loss_advantage.mean(), development["targets"]["loss_advantage_mean"])

    score_names = [str(value) for value in probes["score_names"]]
    scores = {name: probes["scores"][index].astype(np.float64) for index, name in enumerate(score_names)}
    estimator_lookup = {row["estimator"]: row for row in development["estimator_rows"]}
    checks["score_names_complete"] = set(score_names) == set(estimator_lookup)
    for name, values in scores.items():
        row = estimator_lookup[name]
        checks[f"estimator_{name}_auroc"] = close(binary_auroc(values, signed == 1), row["auroc_positive_advantage"])
        checks[f"estimator_{name}_auprc"] = close(binary_average_precision(values, signed == 1), row["auprc_positive_advantage"])
        checks[f"estimator_{name}_spearman"] = np.isclose(spearman_correlation(values, loss_advantage), row["loss_advantage_spearman"], rtol=1e-5, atol=1e-6, equal_nan=True)
        checks[f"estimator_{name}_pairwise"] = np.isclose(pairwise_ranking_accuracy(values, signed), row["signed_pairwise_accuracy"], rtol=1e-5, atol=1e-6, equal_nan=True)

    policy_lookup = {(row["method"], row["target_rate"]): row for row in development["policy_rows"]}
    sparse_compute = development["sparse_compute"]
    for name, values in scores.items():
        for rate in development["scope"]["target_rates"]:
            decisions = top_k(values, float(rate))
            nac = float(np.mean(signed * decisions))
            oracle_nac = float(np.mean(signed * top_k(signed, float(rate))))
            final_correct = sparse_correct.astype(np.int8) + signed * decisions
            row = policy_lookup[(name, float(rate))]
            checks[f"policy_{name}_{rate}_accuracy"] = close(final_correct.mean(), row["accuracy"])
            checks[f"policy_{name}_{rate}_nac"] = close(nac, row["net_advantage_capture"])
            checks[f"policy_{name}_{rate}_recovery"] = close(nac / oracle_nac, row["oracle_recovery"])
            checks[f"policy_{name}_{rate}_compute"] = close(sparse_compute + decisions.mean(), row["expected_sequential_compute"])

    generic = ("margin", "entropy", "max_probability")
    best_generic_auc = max(estimator_lookup[name]["auroc_positive_advantage"] for name in generic)
    best_generic_nac = max(policy_lookup[(name, 0.3)]["net_advantage_capture"] for name in generic)
    selected = development["gate_b"]["selected_method"]
    selected_metrics = estimator_lookup[selected]
    selected_policy = policy_lookup[(selected, 0.3)]
    expected_gate = (
        selected_metrics["auroc_positive_advantage"] > best_generic_auc
        and selected_policy["net_advantage_capture"] > best_generic_nac
        and selected_metrics["positive_fold_spearman_count"] >= 4
        and (selected_metrics["auroc_positive_advantage"] >= 0.68 or selected_policy["oracle_recovery"] >= 0.40)
    )
    checks["gate_b_recomputed"] = development["gate_b"]["passed"] == expected_gate
    checks["actual_disagreement_reproduced"] = estimator_lookup["actual_capacity_disagreement"]["auroc_positive_advantage"] >= 0.70
    checks["predicted_instability_below_actual"] = max(
        row["auroc_positive_advantage"] for row in development["estimator_rows"]
        if row["estimator"].startswith("predicted_instability_hidden")
    ) < estimator_lookup["actual_capacity_disagreement"]["auroc_positive_advantage"]

    source_lookup = {source["id"]: source for source in report["manifest"]["sources"]}
    checks["all_charts_have_sql_sources"] = all(source_lookup[chart["sourceId"]].get("query", {}).get("sql") for chart in report["manifest"]["charts"])
    checks["all_tables_have_sql_sources"] = all(source_lookup[table["sourceId"]].get("query", {}).get("sql") for table in report["manifest"]["tables"])
    block_index = {block["id"]: index for index, block in enumerate(report["manifest"]["blocks"])}
    for chart in report["manifest"]["charts"]:
        chart_blocks = [block for block in report["manifest"]["blocks"] if block.get("chartId") == chart["id"]]
        checks[f"chart_{chart['id']}_adjacent_narrative"] = len(chart_blocks) == 1 and block_index[chart_blocks[0]["id"]] > 0 and report["manifest"]["blocks"][block_index[chart_blocks[0]["id"]] - 1]["type"] == "markdown"
    checks["required_report_sections"] = all(
        any(label in block.get("body", "") for block in report["manifest"]["blocks"])
        for label in ("Technical summary", "Scope, data, and metric definitions", "Model specification and experimental design", "Limitations, uncertainty, and robustness", "Recommended next steps", "Further questions")
    )

    checks = {key: bool(value) for key, value in checks.items()}
    key_files = [
        ROOT / "hierarchy.npz", ROOT / "masks.npz", ROOT / "oracle_analysis.json",
        ROOT / "representations.npz", ROOT / "probes.npz", ROOT / "development_analysis.json",
        ROOT / "summary.json", ROOT / "report/artifact.json", ROOT / "report/report.html",
    ]
    output = {
        "assessment": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "all_checks_passed": all(checks.values()), "check_count": len(checks), "checks": checks,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "report_package_verification": "structural_only",
        "sha256": {str(path): digest(path) for path in key_files},
        "material_caveats": [
            "ARC train is development data; ARC validation was intentionally not reused and MMLU-Pro was gated off.",
            "The S50 representation rerun has 98.13% prediction agreement with its hierarchy reference; probe targets use the fresh rerun.",
            "Actual S50-S25 disagreement is an expensive diagnostic, not a deployment-cost-equivalent router.",
            "Linear fixed-alpha probes do not rule out jointly trained, nonlinear, or pruning-recovered representations.",
            "Compute is an analytical transformer matrix-parameter proxy, not measured runtime or memory.",
            "Portable report packaging passed structurally; browser interaction and viewport QA were unavailable.",
        ],
    }
    (ROOT / "validation.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"assessment": output["assessment"], "passed": sum(checks.values()), "total": len(checks)}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
