import numpy as np

from dualkd.evaluation import (
    cascade_curve,
    error_detection_metrics,
    paired_bootstrap_binary_ranking_difference,
    threshold_compatibility,
)


def test_error_detection_uses_errors_as_positive_class():
    result = error_detection_metrics(
        np.array([True, False, True, False]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([0.9, 0.2, 0.8, 0.3]),
        np.array([0.8, 0.1, 0.7, 0.2]),
    )
    assert result["error_rate"] == 0.5
    assert result["entropy_auroc"] == 1.0
    assert result["entropy_auprc"] == 1.0


def test_threshold_compatibility_uses_one_teacher_defined_threshold():
    rows = threshold_compatibility(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.1, 0.7, 0.6, 0.95]),
        [0.5],
    )
    assert rows == [
        {
            "target_teacher_rate": 0.5,
            "threshold": 0.5,
            "teacher_rate": 0.5,
            "student_rate": 0.75,
            "disagreement": 0.25,
        }
    ]


def test_cascade_defers_highest_uncertainty():
    rows = cascade_curve(
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0.1, 0.8, 0.9, 0.2]),
        [0.0, 0.5, 1.0],
    )
    assert rows[0]["accuracy"] == 0.25
    assert rows[1]["deferred_examples"] == 2
    assert rows[1]["accuracy"] == 0.5


def test_ranking_bootstrap_is_paired():
    correct = np.array([True, False, True, False, True, False])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    result = paired_bootstrap_binary_ranking_difference(
        correct, score, correct, score, reps=100, seed=7, metric="error_auroc"
    )
    assert np.nanmax(np.abs(result)) == 0.0
