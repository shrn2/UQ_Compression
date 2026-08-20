import numpy as np

from tuac.capacity_contrast import (
    capacity_contrast_targets,
    deterministic_stratified_split,
    scalar_output_features,
)


def test_capacity_contrast_targets_match_definitions():
    s50 = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.4, 0.35, 0.25]])
    s25 = np.array([[0.6, 0.3, 0.1], [0.6, 0.2, 0.2], [0.3, 0.45, 0.25]])
    targets = capacity_contrast_targets(s50, s25)
    np.testing.assert_array_equal(targets["hard"], [0, 1, 1])
    assert np.all(targets["js"] >= 0)
    assert np.all(targets["kl"] >= 0)
    np.testing.assert_allclose(targets["confidence_shift"], [0.2, 0.2, 0.1], atol=1e-7)


def test_scalar_features_are_choice_count_agnostic_and_finite():
    probabilities = np.array([[0.7, 0.2, 0.1, 0.0], [0.25, 0.25, 0.25, 0.25]])
    features = scalar_output_features(probabilities)
    assert features.shape == (2, 4)
    assert np.isfinite(features).all()
    assert features[0, 3] > features[1, 3]


def test_deterministic_split_is_disjoint_and_stratified():
    labels = np.array([0] * 80 + [1] * 20)
    ids = [f"example-{index}" for index in range(100)]
    train_a, validation_a = deterministic_stratified_split(labels, ids, validation_fraction=0.2)
    train_b, validation_b = deterministic_stratified_split(labels, ids, validation_fraction=0.2)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert not set(train_a) & set(validation_a)
    assert len(validation_a) == 20
    assert labels[validation_a].sum() == 4


def test_mismatched_probability_shapes_are_rejected():
    with np.testing.assert_raises(ValueError):
        capacity_contrast_targets(np.ones((2, 3)) / 3, np.ones((2, 4)) / 4)
