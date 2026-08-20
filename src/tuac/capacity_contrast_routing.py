"""Frozen Capacity-Contrast Routing system evaluation (Iteration 9)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .cascade import uncertainty_scores
from .cascade_compression import _mcnemar, threshold_for_rate
from .capacity_contrast import (
    deterministic_stratified_split,
    load_contrast_artifact,
    scalar_output_features,
)
from .counterfactual_routing import (
    _clear_cuda,
    _extract_prompt_representations,
    _paired_policy_interval,
    _policy_row,
    _rank_decisions,
    _write_json,
    fit_ridge,
    predict_ridge,
)
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import binary_average_precision, binary_auroc
from .quantization import transformer_blocks
from .structured_sparsity import apply_dense_masks, js_divergence, precision_at


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(prompt: str) -> str:
    """Normalize only the question portion for outcome-blind duplicate checks."""

    text = str(prompt).strip()
    if text.lower().startswith("question:"):
        text = text[len("question:") :]
    for marker in ("\noptions:", "\nanswer:"):
        position = text.lower().find(marker)
        if position >= 0:
            text = text[:position]
            break
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def prepare_deduplicated_evaluation(
    *, training_data_path: str, evaluation_data_path: str, output_path: str, manifest_path: str
) -> dict:
    """Freeze evaluation IDs after outcome-blind exact/trivial-near duplicate removal."""

    training_rows = [json.loads(line) for line in Path(training_data_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluation_rows = [json.loads(line) for line in Path(evaluation_data_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    training_questions = {normalize_question(row["prompt"]) for row in training_rows}
    training_compact = {question.replace(" ", "") for question in training_questions}
    prefix_buckets: dict[str, list[str]] = {}
    for question in training_questions:
        prefix_buckets.setdefault(question[:64], []).append(question)
    kept, exact_removed, near_removed = [], [], []
    for row in evaluation_rows:
        question = normalize_question(row["prompt"])
        if question in training_questions or question.replace(" ", "") in training_compact:
            exact_removed.append(str(row.get("id")))
            continue
        near = any(
            (left.startswith(right) or right.startswith(left))
            and min(len(left), len(right)) / max(1, max(len(left), len(right))) >= 0.98
            for left in (question,)
            for right in prefix_buckets.get(question[:64], [])
        )
        if near:
            near_removed.append(str(row.get("id")))
        else:
            kept.append(row)
    if len({str(row.get("id")) for row in kept}) != len(kept):
        raise ValueError("evaluation IDs are not unique")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "outcome_blind_mmlu_pro_deduplication",
        "training_source": str(Path(training_data_path).as_posix()),
        "evaluation_source": str(Path(evaluation_data_path).as_posix()),
        "training_questions_checked": len(training_rows),
        "evaluation_questions_checked": len(evaluation_rows),
        "exact_duplicates_removed": len(exact_removed),
        "near_duplicates_removed": len(near_removed),
        "removed_ids": exact_removed + near_removed,
        "final_evaluation_size": len(kept),
        "final_ids": [str(row.get("id")) for row in kept],
        "normalization": "question portion; Unicode NFKC; lowercase; punctuation to spaces; repeated spaces collapsed",
        "near_duplicate_rule": "prefix containment with at least 98% normalized-length agreement",
        "correctness_not_used": True,
        "source_sha256": _sha256(evaluation_data_path),
        "output_sha256": _sha256(destination),
    }
    _write_json(Path(manifest_path), manifest)
    return manifest


def load_frozen_ccr(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as values:
        return {
            "hidden_mean": values["hidden_mean"], "hidden_scale": values["hidden_scale"],
            "scalar_mean": values["scalar_mean"], "scalar_scale": values["scalar_scale"],
            "weight": values["weight"], "bias": float(values["bias"]),
            "metadata": json.loads(str(values["metadata"])),
        }


def linear_ccr_scores(router: dict, hidden_l14: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    hidden = (np.asarray(hidden_l14, dtype=np.float64) - router["hidden_mean"]) / router["hidden_scale"]
    scalars = (np.asarray(scalar, dtype=np.float64) - router["scalar_mean"]) / router["scalar_scale"]
    features = np.column_stack([hidden, scalars])
    logits = np.clip(features @ router["weight"] + router["bias"], -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))


def freeze_iteration8_linear_ccr(
    *, paired_artifact_path: str, iteration8_models_path: str,
    iteration8_scores_path: str, iteration8_analysis_path: str, output_path: str,
) -> dict:
    """Export the matched Iteration 8 linear router and reproduce its held-out score."""

    import torch

    paired = load_contrast_artifact(paired_artifact_path)
    checkpoint = torch.load(iteration8_models_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dicts"]["linear"]
    position = 1
    router = {
        "hidden_mean": state["hidden_mean"][position].numpy().astype(np.float32),
        "hidden_scale": state["hidden_scale"][position].numpy().astype(np.float32),
        "scalar_mean": state["scalar_mean"].numpy().astype(np.float32),
        "scalar_scale": state["scalar_scale"].numpy().astype(np.float32),
        "weight": state["output.weight"][0].numpy().astype(np.float32),
        "bias": float(state["output.bias"][0]),
    }
    with np.load(iteration8_scores_path, allow_pickle=False) as scores:
        names = [str(value) for value in scores["validation_score_names"]]
        validation_indices = scores["validation_indices"]
        reference = scores["validation_scores"][names.index("linear")]
        hard = scores["hard_validation"].astype(bool)
    observed = linear_ccr_scores(
        router, paired["hidden_states"][validation_indices, position].astype(np.float32),
        paired["scalar_features"][validation_indices],
    )
    maximum_difference = float(np.max(np.abs(observed - reference)))
    if maximum_difference > 1e-5:
        raise ValueError(f"frozen CCR does not reproduce Iteration 8 scores: {maximum_difference}")
    analysis = json.loads(Path(iteration8_analysis_path).read_text(encoding="utf-8"))
    observed_auroc = binary_auroc(observed, hard)
    expected_auroc = next(
        row["auroc"] for row in analysis["contrast_metrics"] if row["method"] == "linear_instability"
    )
    if abs(observed_auroc - expected_auroc) > 1e-12:
        raise ValueError("frozen CCR AUROC does not reproduce Iteration 8")
    train_indices, expected_validation = deterministic_stratified_split(
        paired["hard"], paired["example_ids"], seed=int(checkpoint["training_seed"])
    )
    if not np.array_equal(validation_indices, expected_validation):
        raise ValueError("Iteration 8 split is not reproducible")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "pre_registered_ccr_freeze",
        "source_checkpoint_sha256": _sha256(iteration8_models_path),
        "paired_artifact_sha256": _sha256(paired_artifact_path),
        "training_examples": len(train_indices), "validation_examples": len(validation_indices),
        "training_seed": int(checkpoint["training_seed"]),
        "training_target": "hard S50-S25 prediction disagreement",
        "features": ["h14", "entropy", "margin_uncertainty", "max_probability_uncertainty", "logit_gap"],
        "classifier": "regularized linear logistic router trained in Iteration 8",
        "heldout_disagreement_auroc": observed_auroc,
        "heldout_score_max_absolute_difference": maximum_difference,
        "outcome_labels_read": False,
        "weights_frozen_before_evaluation": True,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, **router,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return load_frozen_ccr(destination)


def load_ccr_evaluation(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as values:
        return {
            "s50": values["s50"], "dense": values["dense"], "s25": values["s25"],
            "hidden_l14": values["hidden_l14"], "scalar_features": values["scalar_features"],
            "ccr_scores": values["ccr_scores"], "labels": values["labels"],
            "example_ids": tuple(str(value) for value in values["example_ids"]),
            "metadata": json.loads(str(values["metadata"])),
        }


def _load_probability_pass(path: Path, ids: tuple[str, ...], *, hidden: bool = False) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        saved_ids = tuple(str(value) for value in values["example_ids"])
        if saved_ids != ids:
            raise ValueError(f"cache IDs do not match: {path}")
        result = {"probabilities": values["probabilities"]}
        if hidden:
            result["hidden_l14"] = values["hidden_l14"]
        return result


def collect_ccr_evaluation(
    *, data_path: str, masks_path: str, frozen_router_path: str, slm_model: str,
    output_path: str, batch_size: int = 64, device_map: str = "auto", dtype: str = "auto",
    trust_remote_code: bool = False, include_s25: bool = True,
) -> dict:
    """Sequentially cache S50, dense, and optional S25 results on a frozen cohort."""

    destination = Path(output_path)
    if destination.exists():
        return load_ccr_evaluation(destination)
    examples = load_jsonl(data_path)
    ids = tuple(item.id for item in examples)
    if any(item.answer is None for item in examples):
        raise ValueError("labeled evaluation examples are required")
    with np.load(masks_path, allow_pickle=False) as masks:
        s50_mask, s25_mask = masks["s50"], masks["s25"]
    caches = {name: destination.with_name(destination.stem + f"_{name}.npz") for name in ("s50", "dense", "s25")}
    elapsed = {"s50_seconds": 0.0, "dense_seconds": 0.0, "s25_seconds": 0.0}

    s50 = _load_probability_pass(caches["s50"], ids, hidden=True)
    if s50 is None:
        scorer = HFCausalLMScorer(
            slm_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code,
            batch_size=batch_size, max_batch_tokens=4096,
        )
        apply_dense_masks(transformer_blocks(scorer.model), s50_mask)
        started = time.perf_counter()
        probabilities = scorer.score(examples)
        hidden_l14 = _extract_prompt_representations(scorer, examples, (13,))[:, 0]
        elapsed["s50_seconds"] = time.perf_counter() - started
        np.savez_compressed(
            caches["s50"], probabilities=probabilities.astype(np.float32),
            hidden_l14=hidden_l14.astype(np.float16), example_ids=np.asarray(ids, dtype=np.str_),
        )
        s50 = {"probabilities": probabilities, "hidden_l14": hidden_l14}
        del scorer
        _clear_cuda()
        print(f"MMLU-Pro S50 pass completed in {elapsed['s50_seconds']:.1f}s", flush=True)

    dense = _load_probability_pass(caches["dense"], ids)
    if dense is None:
        scorer = HFCausalLMScorer(
            slm_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code,
            batch_size=batch_size, max_batch_tokens=4096,
        )
        started = time.perf_counter()
        probabilities = scorer.score(examples)
        elapsed["dense_seconds"] = time.perf_counter() - started
        np.savez_compressed(caches["dense"], probabilities=probabilities.astype(np.float32), example_ids=np.asarray(ids, dtype=np.str_))
        dense = {"probabilities": probabilities}
        del scorer
        _clear_cuda()
        print(f"MMLU-Pro dense pass completed in {elapsed['dense_seconds']:.1f}s", flush=True)

    s25 = _load_probability_pass(caches["s25"], ids)
    if include_s25 and s25 is None:
        scorer = HFCausalLMScorer(
            slm_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code,
            batch_size=batch_size, max_batch_tokens=4096,
        )
        apply_dense_masks(transformer_blocks(scorer.model), s25_mask)
        started = time.perf_counter()
        probabilities = scorer.score(examples)
        elapsed["s25_seconds"] = time.perf_counter() - started
        np.savez_compressed(caches["s25"], probabilities=probabilities.astype(np.float32), example_ids=np.asarray(ids, dtype=np.str_))
        s25 = {"probabilities": probabilities}
        del scorer
        _clear_cuda()
        print(f"MMLU-Pro S25 diagnostic pass completed in {elapsed['s25_seconds']:.1f}s", flush=True)
    if s25 is None:
        s25 = {"probabilities": np.empty((0, s50["probabilities"].shape[1]), dtype=np.float32)}

    scalar = scalar_output_features(s50["probabilities"])
    router = load_frozen_ccr(frozen_router_path)
    ccr = linear_ccr_scores(router, s50["hidden_l14"], scalar)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": "frozen_ccr_mmlu_pro_collection",
        "model": slm_model, "dataset": "MMLU-Pro test", "examples": len(examples),
        "data_sha256": _sha256(data_path), "frozen_router_sha256": _sha256(frozen_router_path),
        "physical_s50_and_s25": True, "s25_diagnostic_included": include_s25,
        "sequential_model_loading": True, "passes": elapsed,
        "router_features": router["metadata"]["features"], "router_frozen_before_outcomes": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, s50=np.asarray(s50["probabilities"], dtype=np.float32),
        dense=np.asarray(dense["probabilities"], dtype=np.float32),
        s25=np.asarray(s25["probabilities"], dtype=np.float32),
        hidden_l14=np.asarray(s50["hidden_l14"], dtype=np.float16), scalar_features=scalar,
        ccr_scores=ccr.astype(np.float32), labels=np.asarray([item.answer for item in examples], dtype=np.int64),
        example_ids=np.asarray(ids, dtype=np.str_), metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return load_ccr_evaluation(destination)


def _metric_row(name: str, values: np.ndarray, target: np.ndarray, deployability: str) -> dict:
    return {
        "method": name, "deployability": deployability,
        "auroc": binary_auroc(values, target), "auprc": binary_average_precision(values, target),
        "precision_at_10": precision_at(values, target, 0.10),
        "precision_at_20": precision_at(values, target, 0.20),
        "precision_at_30": precision_at(values, target, 0.30),
    }


def _threshold_policy(
    scores: np.ndarray, threshold: float, signed_advantage: np.ndarray,
    sparse_correct: np.ndarray, sparse_compute: float,
) -> dict:
    decisions = np.asarray(scores) > threshold
    final_correct = sparse_correct.astype(np.int8) + signed_advantage * decisions
    rate = float(decisions.mean())
    oracle = _rank_decisions(signed_advantage, rate)
    oracle_nac = float(np.mean(signed_advantage * oracle))
    selected = max(1, int(decisions.sum()))
    nac = float(np.mean(signed_advantage * decisions))
    return {
        "threshold": float(threshold), "actual_dense_rate": rate,
        "accuracy": float(final_correct.mean()), "expected_sequential_compute": sparse_compute + rate,
        "net_advantage_capture": nac, "oracle_recovery": nac / oracle_nac if oracle_nac > 0 else None,
        "beneficial_selected_rate": float(np.sum(decisions & (signed_advantage == 1)) / selected),
        "harmful_selected_rate": float(np.sum(decisions & (signed_advantage == -1)) / selected),
    }


def analyze_ccr_system(
    *, evaluation_artifact_path: str, paired_artifact_path: str, iteration8_scores_path: str,
    output_path: str, scores_output_path: str, target_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
    random_seeds: Sequence[int] = tuple(range(20)), bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260826,
) -> dict:
    """Evaluate positive advantage and exact matched-compute routing for frozen CCR."""

    evaluation = load_ccr_evaluation(evaluation_artifact_path)
    labels, s50, dense = evaluation["labels"], evaluation["s50"], evaluation["dense"]
    sparse_correct = s50.argmax(axis=1) == labels
    dense_correct = dense.argmax(axis=1) == labels
    signed_advantage = dense_correct.astype(np.int8) - sparse_correct.astype(np.int8)
    positive = signed_advantage == 1
    harmful = signed_advantage == -1
    uncertainty = uncertainty_scores(s50)
    scores: dict[str, np.ndarray] = {
        "margin": uncertainty["margin"], "entropy": uncertainty["entropy"],
        "max_probability": uncertainty["max_probability"], "ccr": evaluation["ccr_scores"].astype(np.float64),
    }

    paired = load_contrast_artifact(paired_artifact_path)
    _, calibration = deterministic_stratified_split(paired["hard"], paired["example_ids"])
    calibration_features = np.column_stack([
        paired["hidden_states"][calibration, 1].astype(np.float64), paired["scalar_features"][calibration]
    ])
    calibration_error = paired["s50"][calibration].argmax(axis=1) != paired["labels"][calibration]
    error_probe = fit_ridge(calibration_features, calibration_error.astype(float), alpha=1.0)
    evaluation_features = np.column_stack([evaluation["hidden_l14"].astype(np.float64), evaluation["scalar_features"]])
    scores["error_probe"] = predict_ridge(error_probe, evaluation_features)[:, 0]
    if len(evaluation["s25"]):
        scores["actual_capacity_disagreement"] = (
            s50.argmax(axis=1) != evaluation["s25"].argmax(axis=1)
        ).astype(float)
        scores["actual_capacity_js"] = js_divergence(s50, evaluation["s25"])
    scores["positive_uplift_oracle"] = positive.astype(float)
    scores["signed_advantage_oracle"] = signed_advantage.astype(float)

    deployability = {
        name: "oracle" if name.endswith("oracle") else "expensive_diagnostic" if name.startswith("actual_") else "sparse_only"
        for name in scores
    }
    metric_rows = [_metric_row(name, values, positive, deployability[name]) for name, values in scores.items()]
    sparse_compute = 0.5588235294117647
    policy_rows, decisions, correctness = [], {}, {}
    for name, values in scores.items():
        for rate in target_rates:
            row, decision, correct = _policy_row(
                name, values, float(rate), signed_advantage, sparse_correct, sparse_compute
            )
            policy_rows.append(row)
            decisions[(name, float(rate))] = decision
            correctness[(name, float(rate))] = correct
    random_rows = []
    for seed in random_seeds:
        values = np.random.default_rng(int(seed)).random(len(labels))
        for rate in target_rates:
            row, _, _ = _policy_row(
                f"random_seed_{seed}", values, float(rate), signed_advantage, sparse_correct, sparse_compute
            )
            random_rows.append({"seed": int(seed), **row})
    random_summary = []
    for rate in target_rates:
        selected = [row for row in random_rows if row["target_rate"] == float(rate)]
        random_summary.append({
            "method": "random_20_seed_mean", "target_rate": float(rate),
            "accuracy": float(np.mean([row["accuracy"] for row in selected])),
            "accuracy_sd": float(np.std([row["accuracy"] for row in selected], ddof=1)),
            "net_advantage_capture": float(np.mean([row["net_advantage_capture"] for row in selected])),
            "oracle_recovery": float(np.mean([row["oracle_recovery"] for row in selected])),
            "expected_sequential_compute": sparse_compute + float(rate),
        })

    comparisons = []
    for rate in (0.2, 0.3):
        for baseline in ("entropy", "max_probability"):
            values = _paired_policy_interval(
                decisions[("ccr", rate)], decisions[(baseline, rate)], signed_advantage,
                sparse_correct, rate, seed=bootstrap_seed + int(rate * 1000) + len(baseline),
                replicates=bootstrap_replicates,
            )
            comparisons.append({
                "target_rate": rate, "comparison": f"ccr_minus_{baseline}", **values,
                "mcnemar": _mcnemar(correctness[("ccr", rate)], correctness[(baseline, rate)]),
            })

    with np.load(iteration8_scores_path, allow_pickle=False) as saved:
        names = [str(value) for value in saved["validation_score_names"]]
        calibration_ccr_scores = saved["validation_scores"][names.index("linear")]
    threshold_rows = []
    for rate in (0.2, 0.3):
        threshold = threshold_for_rate(calibration_ccr_scores, rate)
        threshold_rows.append({"target_calibration_rate": rate, **_threshold_policy(
            scores["ccr"], threshold, signed_advantage, sparse_correct, sparse_compute
        )})

    outcomes = []
    for name in ("ccr", "entropy", "max_probability"):
        decision = decisions[(name, 0.3)]
        selected = max(1, int(decision.sum()))
        categories = {
            "beneficial": decision & positive,
            "harmful": decision & harmful,
            "neutral_both_correct": decision & sparse_correct & dense_correct,
            "neutral_both_wrong": decision & ~sparse_correct & ~dense_correct,
        }
        for outcome, mask in categories.items():
            outcomes.append({"method": name, "outcome": outcome, "selected_rate": float(mask.sum() / selected), "selected_count": int(mask.sum())})

    metric_lookup = {row["method"]: row for row in metric_rows}
    policy_lookup = {(row["method"], row["target_rate"]): row for row in policy_rows}
    alignment_passed = metric_lookup["ccr"]["auroc"] > max(
        metric_lookup["entropy"]["auroc"], metric_lookup["max_probability"]["auroc"]
    )
    meaningful_points = []
    for rate in (0.2, 0.3):
        ccr = policy_lookup[("ccr", rate)]
        entropy = policy_lookup[("entropy", rate)]
        maxp = policy_lookup[("max_probability", rate)]
        meaningful_points.append({
            "target_rate": rate, "ccr_accuracy": ccr["accuracy"],
            "best_uncertainty_accuracy": max(entropy["accuracy"], maxp["accuracy"]),
            "ccr_nac": ccr["net_advantage_capture"],
            "best_uncertainty_nac": max(entropy["net_advantage_capture"], maxp["net_advantage_capture"]),
            "ccr_oracle_recovery": ccr["oracle_recovery"],
            "best_uncertainty_oracle_recovery": max(entropy["oracle_recovery"], maxp["oracle_recovery"]),
            "passed": bool(
                ccr["accuracy"] > max(entropy["accuracy"], maxp["accuracy"])
                and ccr["net_advantage_capture"] > max(entropy["net_advantage_capture"], maxp["net_advantage_capture"])
                and ccr["oracle_recovery"] > max(entropy["oracle_recovery"], maxp["oracle_recovery"])
            ),
        })
    system_passed = any(row["passed"] for row in meaningful_points)
    gate_passed = bool(alignment_passed and system_passed)
    best_effect = max(row["ccr_accuracy"] - row["best_uncertainty_accuracy"] for row in meaningful_points)
    strong_support = any(
        row["accuracy_difference"] >= 0.015 and row["accuracy_difference_ci"][0] > 0
        for row in comparisons
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": "frozen_ccr_mmlu_pro_system_evaluation",
        "scope": {
            "model": evaluation["metadata"]["model"], "dataset": evaluation["metadata"]["dataset"],
            "examples": len(labels), "target_rates": list(target_rates), "random_seeds": list(random_seeds),
            "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
            "router_outcome_tuning": False, "compute_proxy_not_latency": True,
        },
        "static": {
            "s50_accuracy": float(sparse_correct.mean()), "dense_accuracy": float(dense_correct.mean()),
            "positive_advantage_count": int(positive.sum()), "harmful_advantage_count": int(harmful.sum()),
            "neutral_count": int(np.sum(signed_advantage == 0)),
            "s50_compute_proxy": sparse_compute, "dense_compute_proxy": 1.0,
        },
        "advantage_metrics": metric_rows, "policy_rows": policy_rows,
        "random_rows": random_rows, "random_summary": random_summary,
        "comparisons": comparisons, "threshold_transfer": threshold_rows,
        "routing_outcomes_30": outcomes,
        "gate": {
            "passed": gate_passed, "alignment_passed": alignment_passed,
            "system_passed": system_passed, "meaningful_operating_points": meaningful_points,
            "best_accuracy_gain_over_uncertainty": best_effect,
            "practical_target_met": best_effect >= 0.015,
            "strong_bootstrap_support": strong_support,
            "criterion": "CCR must beat entropy and max-probability AUROC for positive advantage and improve accuracy, NAC, and oracle recovery over both at 20% or 30% exact dense execution",
            "decision": "GO" if gate_passed else "NO-GO",
            "next_stage": "conditional_ablations_and_runtime" if gate_passed else "STOP",
        },
        "conditional_followups": {
            "training_data_ablation": "authorized" if gate_passed else "not_run_gate_failed",
            "feature_ablation": "authorized" if gate_passed else "not_run_gate_failed",
            "s40_evaluation": "authorized" if gate_passed else "not_run_gate_failed",
            "runtime": "authorized" if gate_passed else "not_run_gate_failed",
        },
    }
    score_destination = Path(scores_output_path)
    score_destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(scores)
    np.savez_compressed(
        score_destination, score_names=np.asarray(ordered, dtype=np.str_),
        scores=np.stack([scores[name] for name in ordered]).astype(np.float64),
        signed_advantage=signed_advantage, sparse_correct=sparse_correct, dense_correct=dense_correct,
        metadata=np.asarray(json.dumps({"frozen_ccr": True, "evaluation_outcome_tuning": False}, sort_keys=True), dtype=np.str_),
    )
    _write_json(Path(output_path), result)
    return result
