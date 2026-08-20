import numpy as np

from tuac.uncertainty_robust_pruning import (
    greedy_minimax_order,
    lac_metrics,
    lac_threshold,
    masks_from_order,
    uncertainty_strata,
)


def test_uncertainty_strata_are_balanced_and_monotonic():
    probabilities = np.asarray(
        [
            [0.99, 0.01],
            [0.90, 0.10],
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.50, 0.50],
        ]
    )
    uncertainty, labels = uncertainty_strata(probabilities, strata=3)
    assert np.bincount(labels).tolist() == [2, 2, 2]
    assert uncertainty[labels == 0].max() <= uncertainty[labels == 1].min()
    assert uncertainty[labels == 1].max() <= uncertainty[labels == 2].min()


def test_greedy_minimax_protects_stratum_specific_channels():
    # Channels 0 and 1 are uniquely important to different strata.  The robust
    # order must select both before the shared low-value channels.
    importance = np.asarray(
        [
            [[9.0, 0.1, 1.0, 1.0]],
            [[0.1, 9.0, 1.0, 1.0]],
        ]
    )
    order, history = greedy_minimax_order(importance, temperature=0.05)
    assert set(order[0, :2]) == {0, 1}
    assert np.all(np.diff(history[0, :, :], axis=0) <= 1e-12)


def test_masks_from_order_are_nested():
    order = np.asarray([[3, 1, 5, 0, 2, 4], [0, 2, 4, 1, 3, 5]])
    masks = masks_from_order(order, medium_sparsity=1 / 3, aggressive_sparsity=0.5)
    assert masks["medium"].shape == (2, 4)
    assert masks["aggressive"].shape == (2, 3)
    for medium, aggressive in zip(masks["medium"], masks["aggressive"]):
        assert set(aggressive).issubset(set(medium))


def test_lac_threshold_and_metrics_cover_calibration_examples():
    probabilities = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7]]
    )
    labels = np.asarray([0, 0, 0, 1, 1])
    threshold = lac_threshold(probabilities, labels, alpha=0.2)
    metrics = lac_metrics(probabilities, labels, threshold)
    assert metrics["coverage"] >= 0.8
    assert 1.0 <= metrics["mean_set_size"] <= 2.0
