import numpy as np

from tuac.counterfactual_routing import (
    _mask_for_sparsity,
    _policy_row,
    cross_validated_ridge,
    pairwise_ranking_accuracy,
    predict_ridge,
)
from tuac.metrics import spearman_correlation


def test_capacity_masks_are_nested_for_shared_importance():
    importance = np.asarray([
        [4, 1, 8, 3, 7, 2, 6, 5, 0, 9],
        [0, 9, 1, 8, 2, 7, 3, 6, 4, 5],
    ], dtype=float)
    s25 = _mask_for_sparsity(importance, 0.25)
    s40 = _mask_for_sparsity(importance, 0.40)
    s50 = _mask_for_sparsity(importance, 0.50)
    for left, middle, right in zip(s25, s40, s50):
        assert set(right).issubset(set(middle))
        assert set(middle).issubset(set(left))


def test_cross_validated_multitarget_ridge_recovers_signal():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(180, 24))
    targets = np.column_stack([
        2 * features[:, 0] - features[:, 2],
        -features[:, 1] + 0.5 * features[:, 3],
    ])
    stratification = targets[:, 0] > np.median(targets[:, 0])
    predictions, fold_ids, model = cross_validated_ridge(
        features, targets, stratification, folds=5, seed=4, alpha=0.1
    )
    assert set(fold_ids) == set(range(5))
    assert spearman_correlation(predictions[:, 0], targets[:, 0]) > 0.9
    full = predict_ridge(model, features)
    assert spearman_correlation(full[:, 1], targets[:, 1]) > 0.95


def test_policy_metrics_capture_signed_benefit_and_harm():
    advantage = np.asarray([1, 1, 0, -1, 0], dtype=np.int8)
    sparse_correct = np.asarray([False, False, True, True, False])
    scores = np.asarray([5, 4, 3, 1, 2], dtype=float)
    row, decisions, final_correct = _policy_row(
        "probe", scores, 0.4, advantage, sparse_correct, 0.6
    )
    assert decisions.tolist() == [True, True, False, False, False]
    assert row["net_advantage_capture"] == 0.4
    assert row["oracle_recovery"] == 1.0
    assert final_correct.tolist() == [True, True, True, True, False]
    assert pairwise_ranking_accuracy(scores, advantage) == 1.0
