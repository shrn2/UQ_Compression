import numpy as np
import pytest

from tuac.allocation import allocate_bits, average_bits, random_allocation


def test_allocator_protects_more_important_layer():
    result = allocate_bits([10.0, 1.0], [100, 100], 3.5, bit_choices=(3, 4))
    assert result.bits.tolist() == [4, 3]
    assert result.average_bits == 3.5
    assert result.total_bits <= result.budget_bits


def test_allocator_accounts_for_unequal_layer_sizes():
    result = allocate_bits([1.0, 10.0], [200, 100], 10 / 3, bit_choices=(3, 4))
    assert result.bits.tolist() == [3, 4]
    assert average_bits(result.bits, [200, 100]) == pytest.approx(10 / 3)


def test_allocator_rejects_impossible_budget():
    with pytest.raises(ValueError, match="target_average_bits"):
        allocate_bits([1.0], [100], 2.0, bit_choices=(3, 4, 8))


def test_random_allocator_is_reproducible_and_matched():
    left = random_allocation([100, 100, 100], 3.5, bit_choices=(3, 4), seed=7)
    right = random_allocation([100, 100, 100], 3.5, bit_choices=(3, 4), seed=7)
    np.testing.assert_array_equal(left.bits, right.bits)
    assert left.total_bits == right.total_bits
    assert left.total_bits <= left.budget_bits


def test_allocator_can_spend_maximum_attainable_budget():
    counts = np.array([1, 1, 1, 1])
    errors = np.array([[-1.0, 0.0, 1.0]] * 4)
    result = allocate_bits(
        np.ones(4), counts, 3.5, bit_choices=(2, 4, 8), errors=errors,
        spend_full_budget=True,
    )
    random = random_allocation(counts, 3.5, bit_choices=(2, 4, 8), seed=0)
    assert result.total_bits == random.total_bits
    assert result.average_bits == 3.5
