import numpy as np
import pytest

from tuac.metrics import (
    binary_average_precision,
    binary_auroc,
    classification_metrics,
    margin_uncertainty,
    pairwise_ordering_flip_rate,
    uncertainty_preservation,
)


def test_metrics_for_correct_confident_predictions():
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = classification_metrics(probabilities, np.array([0, 1]), bins=2)
    assert result["accuracy"] == 1.0
    assert result["ece"] == pytest.approx(0.1)
    assert np.isnan(result["error_auroc"])


def test_binary_auroc_with_ties_uses_average_ranks():
    assert binary_auroc(np.array([0.0, 0.5, 0.5, 1.0]), np.array([0, 0, 1, 1])) == 0.875


def test_average_precision_groups_tied_thresholds():
    value = binary_average_precision(
        np.array([1.0, 0.5, 0.5, 0.0]), np.array([1, 0, 1, 0])
    )
    assert value == pytest.approx(5 / 6)


def test_margin_uncertainty_and_pairwise_ordering_flip():
    probabilities = np.array([[0.8, 0.2], [0.55, 0.45]])
    np.testing.assert_allclose(margin_uncertainty(probabilities), [0.4, 0.9])
    assert pairwise_ordering_flip_rate(np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.2, 0.1])) == 1.0


def test_identical_probabilities_preserve_uncertainty():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.5, 0.5]])
    result = uncertainty_preservation(probabilities, probabilities)
    assert result["mean_absolute_uncertainty_distortion"] == 0.0
    assert result["student_uncertainty_spearman"] == pytest.approx(1.0)
