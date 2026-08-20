import numpy as np

from tuac.capability_search import evaluate_configuration, generate_exact_neighborhood


def test_exact_neighborhood_preserves_budget_and_is_reproducible():
    base = [4] * 8
    counts = np.ones(8, dtype=np.int64)
    first = generate_exact_neighborhood(
        base, counts, local_count=6, random_count=2, seeds=(11, 23, 37)
    )
    second = generate_exact_neighborhood(
        base, counts, local_count=6, random_count=2, seeds=(11, 23, 37)
    )
    assert first == second
    assert len(first) == 9
    assert len({tuple(row["bits"]) for row in first}) == len(first)
    assert all(sum(row["bits"]) == sum(base) for row in first)
    assert first[0]["family"] == "kl"
    assert {row["family"] for row in first} == {"kl", "local", "random"}


def test_configuration_metrics_use_matched_rates_and_expected_utility():
    fp = np.asarray(
        [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.45, 0.55]], dtype=float
    )
    candidate = np.asarray(
        [[0.7, 0.3], [0.55, 0.45], [0.4, 0.6], [0.49, 0.51]], dtype=float
    )
    fallback = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]], dtype=float
    )
    labels = np.asarray([0, 1, 1, 0])
    metrics, arrays = evaluate_configuration(
        candidate,
        fp_probabilities=fp,
        fallback_probabilities=fallback,
        labels=labels,
        router_signal="margin",
        target_rates=(0.25, 0.5),
        cost_coefficients=(0.0, 0.1),
    )
    assert metrics["standalone_accuracy"] == 0.5
    assert arrays["utility"].shape == (4, 4)
    assert metrics["routing"]["0.25"]["delegation_rate"] == 0.25
    assert metrics["routing"]["0.5"]["delegation_rate"] == 0.5
    assert np.isclose(metrics["mean_utility"], arrays["utility"].mean())
