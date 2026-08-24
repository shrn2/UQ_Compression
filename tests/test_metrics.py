import numpy as np

from dualkd.metrics import (
    area_under_risk_coverage,
    binary_average_precision,
    binary_auroc,
    expected_calibration_error,
)
from dualkd.scoring import entropy


def test_binary_metrics_rank_errors():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    targets = np.array([False, True, False, True])
    assert binary_auroc(scores, targets) == 1.0
    assert binary_average_precision(scores, targets) == 1.0


def test_entropy_ignores_zero_padding_in_normalization():
    values = entropy(np.array([[0.5, 0.5, 0.0, 0.0]]), normalized=True)
    np.testing.assert_allclose(values, [1.0])


def test_calibration_and_aurc_are_finite():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    labels = np.array([0, 1, 1])
    assert np.isfinite(expected_calibration_error(probabilities, labels))
    assert np.isfinite(area_under_risk_coverage(probabilities, labels))
