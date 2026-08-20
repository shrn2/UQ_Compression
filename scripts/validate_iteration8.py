"""Recompute and validate Iteration 8 gates and deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.capacity_contrast import capacity_contrast_targets, deterministic_stratified_split
from tuac.metrics import binary_average_precision, binary_auroc


ROOT = Path("runs/iter8_arc")


def main() -> int:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    with np.load(ROOT / "paired_capacity.npz", allow_pickle=False) as paired:
        ids = tuple(str(value) for value in paired["example_ids"])
        metadata = json.loads(str(paired["metadata"]))
        targets = capacity_contrast_targets(paired["s50"], paired["s25"])
        check("paired example count", len(ids) == 3000, str(len(ids)))
        check("paired IDs unique", len(set(ids)) == len(ids))
        check("MMLU-Pro excluded", metadata["mmlu_pro_excluded"] is True)
        check("source is MMLU auxiliary train", metadata["dataset_counts"] == {"MMLU-auxiliary-train": 3000})
        check("physical channel pruning", metadata["physical_channel_pruning"] is True)
        check("capacity pair", metadata["capacity_pair"] == ["s50", "s25"])
        check("four residual layers", metadata["layer_labels"] == ["L7", "L14", "L21", "L28"])
        check("hard targets recompute", np.array_equal(targets["hard"], paired["hard"]))
        check("JS targets recompute", np.allclose(targets["js"], paired["js"], atol=1e-7))
        check("KL targets recompute", np.allclose(targets["kl"], paired["kl"], atol=1e-7))
        check("confidence-shift targets recompute", np.allclose(targets["confidence_shift"], paired["confidence_shift"], atol=1e-7))
        check("hidden tensor shape", paired["hidden_states"].shape == (3000, 4, 1536), str(paired["hidden_states"].shape))
        check("scalar tensor shape", paired["scalar_features"].shape == (3000, 4), str(paired["scalar_features"].shape))
        check("probabilities finite", np.isfinite(paired["s50"]).all() and np.isfinite(paired["s25"]).all())
        check("probabilities normalized", np.allclose(paired["s50"].sum(axis=1), 1, atol=1e-5) and np.allclose(paired["s25"].sum(axis=1), 1, atol=1e-5))
        train_expected, validation_expected = deterministic_stratified_split(paired["hard"], ids)

    analysis = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "scores.npz", allow_pickle=False) as scores:
        names = [str(value) for value in scores["validation_score_names"]]
        matrix = scores["validation_scores"]
        hard = scores["hard_validation"].astype(bool)
        check("four validation score vectors", names == ["linear", "hard", "soft", "multitask"])
        check("held-out score shape", matrix.shape == (4, 600), str(matrix.shape))
        check("training split reproducible", np.array_equal(scores["train_indices"], train_expected))
        check("validation split reproducible", np.array_equal(scores["validation_indices"], validation_expected))
        check("split overlap absent", len(set(scores["train_indices"]) & set(scores["validation_indices"])) == 0)
        recomputed = {}
        for index, name in enumerate(names):
            recomputed[name] = {
                "auroc": binary_auroc(matrix[index], hard),
                "auprc": binary_average_precision(matrix[index], hard),
            }
        rows = {row["method"]: row for row in analysis["contrast_metrics"]}
        check("linear AUROC recomputes", abs(recomputed["linear"]["auroc"] - rows["linear_instability"]["auroc"]) < 1e-12)
        check("CCD-Hard AUROC recomputes", abs(recomputed["hard"]["auroc"] - rows["ccd_hard"]["auroc"]) < 1e-12)
        check("CCD-Soft AUROC recomputes", abs(recomputed["soft"]["auroc"] - rows["ccd_soft"]["auroc"]) < 1e-12)
        check("CCD-MultiTask AUROC recomputes", abs(recomputed["multitask"]["auroc"] - rows["ccd_multitask"]["auroc"]) < 1e-12)
        check("all score values finite", np.isfinite(matrix).all())
        check("ARC scores absent after stop", scores["arc_scores"].shape == (0, 0))

    gate = analysis["gate_1"]
    nonlinear = [rows[name] for name in ("ccd_hard", "ccd_soft", "ccd_multitask")]
    selected = max(nonlinear, key=lambda row: row["auroc"])
    criterion = selected["auroc"] >= 0.65 and selected["auroc"] >= max(
        rows["linear_instability"]["auroc"], gate["iteration7_linear_auroc"]
    ) + 0.01
    check("Gate 1 selected method recomputes", gate["selected_regime"] == selected["method"])
    check("Gate 1 outcome recomputes", gate["passed"] == criterion)
    check("Gate 1 STOP enforced", gate["passed"] is False and gate["next_stage"] == "STOP")
    check("Gate 2 not run", analysis["gate_2"]["status"] == "not_run_gate_1_failed")
    check("Gate 3 not run", analysis["gate_3"]["status"] == "not_run_gate_1_failed")
    check("fresh benchmark not run", analysis["fresh_benchmark"]["status"] == "not_run")
    check("runtime not run", analysis["runtime"]["status"] == "not_run")
    check("router models saved", (ROOT / "routers.pt").stat().st_size > 0)
    artifact = json.loads((ROOT / "report" / "artifact.json").read_text(encoding="utf-8"))
    html = (ROOT / "report" / "report.html").read_text(encoding="utf-8")
    check("report surface", artifact["surface"] == "report")
    check("report ready snapshot", artifact["snapshot"]["status"] == "ready")
    check("one chart declared", len(artifact["manifest"]["charts"]) == 1)
    check("chart dataset has four rows", len(artifact["snapshot"]["datasets"]["contrast"]) == 4)
    check("technical report sections present", all(text in html for text in (
        "Technical summary", "Scope, data, and metric definitions", "Model specification and experimental design",
        "Limitations, uncertainty, and robustness checks", "Recommended next steps", "Further questions",
    )))
    check("portable HTML embeds semantic fallback", "portable-fallback" in html)
    result = {
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "passed": sum(row["passed"] for row in checks), "total": len(checks), "checks": checks,
    }
    (ROOT / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("status", "passed", "total")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
