import numpy as np

from tuac.cascade import (
    CascadeProbabilityArtifact,
    analyze_cascade,
    fit_isotonic,
    fit_temperature,
    predict_isotonic,
    temperature_scale,
    threshold_for_fraction,
    top_fraction_decisions,
)


def _rows(count: int) -> tuple[np.ndarray, np.ndarray]:
    correct = np.tile([0.90, 0.05, 0.05], (count // 2, 1))
    errors = np.tile([0.40, 0.35, 0.25], (count - count // 2, 1))
    probabilities = np.vstack([correct, errors])
    labels = np.concatenate(
        [np.zeros(count // 2, dtype=int), np.ones(count - count // 2, dtype=int)]
    )
    return probabilities, labels


def _artifact() -> CascadeProbabilityArtifact:
    calibration, calibration_labels = _rows(40)
    evaluation, evaluation_labels = _rows(30)
    fallback_calibration = np.tile([0.05, 0.90, 0.05], (40, 1))
    fallback_calibration[:20] = [0.90, 0.05, 0.05]
    fallback_evaluation = np.tile([0.05, 0.90, 0.05], (30, 1))
    fallback_evaluation[:15] = [0.90, 0.05, 0.05]
    quantized_calibration = []
    quantized_evaluation = []
    for amount in (0.01, 0.05, 0.12):
        q_cal = calibration.copy()
        q_eval = evaluation.copy()
        q_cal[:, 0] -= amount
        q_eval[:, 0] -= amount
        q_cal[:, 2] += amount
        q_eval[:, 2] += amount
        quantized_calibration.append(q_cal)
        quantized_evaluation.append(q_eval)
    return CascadeProbabilityArtifact(
        calibration_labels=calibration_labels,
        evaluation_labels=evaluation_labels,
        calibration_ids=tuple(f"c{index}" for index in range(40)),
        evaluation_ids=tuple(f"e{index}" for index in range(30)),
        slm_fp_calibration=calibration,
        slm_fp_evaluation=evaluation,
        fallback_calibration=fallback_calibration,
        fallback_evaluation=fallback_evaluation,
        quantized_calibration=np.asarray(quantized_calibration),
        quantized_evaluation=np.asarray(quantized_evaluation),
        bits=(8, 4, 2),
        metadata={},
    )


def test_threshold_and_top_fraction_are_deterministic():
    scores = np.arange(10, dtype=float)
    decisions = top_fraction_decisions(scores, 0.3)
    assert decisions.sum() == 3
    assert decisions[-3:].all()
    threshold = threshold_for_fraction(scores, 0.3)
    assert threshold["actual_rate"] == 0.3


def test_temperature_and_isotonic_calibrators():
    probabilities, labels = _rows(40)
    temperature = fit_temperature(probabilities, labels)
    scaled = temperature_scale(probabilities, temperature)
    np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
    errors = probabilities.argmax(axis=1) != labels
    scores = np.linspace(0, 1, len(labels))
    model = fit_isotonic(scores, errors)
    predicted = predict_isotonic(model, scores)
    assert np.all(np.diff(predicted) >= 0)


def test_cascade_artifact_roundtrip_and_analysis(tmp_path):
    artifact = _artifact()
    path = tmp_path / "probabilities.npz"
    artifact.save(path)
    loaded = CascadeProbabilityArtifact.load(path)
    result = analyze_cascade(
        loaded,
        output_path=tmp_path / "results.json",
        bootstrap_replicates=20,
        bootstrap_seed=3,
    )
    assert result["router"]["gate_passed"] is True
    assert result["router"]["selected_signal"] in {"entropy", "max_probability", "margin"}
    assert len(result["drift_rows"]) == 4
    assert len(result["flip_rows"]) == 12
    assert len(result["matched_rate_comparisons"]) == 12
    assert len(result["routing_only_comparisons"]) == 12
