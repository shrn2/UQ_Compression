"""Budget-constrained mixed-precision bit allocation."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AllocationResult:
    bits: np.ndarray
    average_bits: float
    total_bits: int
    budget_bits: int
    objective: float


def average_bits(bits: Sequence[int], parameter_counts: Sequence[int]) -> float:
    widths = np.asarray(bits, dtype=np.int64)
    counts = np.asarray(parameter_counts, dtype=np.int64)
    if widths.shape != counts.shape or widths.ndim != 1:
        raise ValueError("bits and parameter_counts must be same-length vectors")
    if np.any(counts <= 0):
        raise ValueError("parameter counts must be positive")
    return float(np.dot(widths, counts) / counts.sum())


def _error_matrix(
    importance: np.ndarray,
    bit_choices: tuple[int, ...],
    errors: Mapping[int, float] | np.ndarray | None,
) -> np.ndarray:
    layers = importance.size
    if errors is None:
        # High-resolution scalar quantization noise is approximately O(2^-2b).
        base = np.asarray([2.0 ** (-2 * bit) for bit in bit_choices])
        return importance[:, None] * base[None, :]
    if isinstance(errors, Mapping):
        missing = set(bit_choices) - set(errors)
        if missing:
            raise ValueError(f"missing error estimates for bits: {sorted(missing)}")
        base = np.asarray([errors[bit] for bit in bit_choices], dtype=np.float64)
        return importance[:, None] * base[None, :]
    matrix = np.asarray(errors, dtype=np.float64)
    if matrix.shape != (layers, len(bit_choices)):
        raise ValueError("layer errors must have shape [layers, bit_choices]")
    return importance[:, None] * matrix


def allocate_bits(
    importance: Sequence[float],
    parameter_counts: Sequence[int],
    target_average_bits: float,
    *,
    bit_choices: Sequence[int] = (3, 4, 8),
    errors: Mapping[int, float] | np.ndarray | None = None,
    max_states: int = 200_000,
    spend_full_budget: bool = False,
) -> AllocationResult:
    """Minimize weighted quantization error under a total bit budget.

    Uses multiple-choice knapsack dynamic programming. Costs are divided by the
    greatest common divisor of layer parameter counts, which keeps normal
    transformer-block problems compact while retaining the exact budget.
    """

    scores = np.asarray(importance, dtype=np.float64)
    counts = np.asarray(parameter_counts, dtype=np.int64)
    choices = tuple(sorted(set(int(bit) for bit in bit_choices)))
    if scores.ndim != 1 or counts.shape != scores.shape or scores.size == 0:
        raise ValueError("importance and parameter_counts must be non-empty, same-length vectors")
    if np.any(~np.isfinite(scores)) or np.any(scores < 0) or np.any(counts <= 0):
        raise ValueError("importance must be finite/non-negative and parameter counts positive")
    if not choices or choices[0] <= 0:
        raise ValueError("bit choices must be positive integers")
    minimum = average_bits(np.full(scores.size, choices[0]), counts)
    maximum = average_bits(np.full(scores.size, choices[-1]), counts)
    if not minimum <= target_average_bits <= maximum:
        raise ValueError(f"target_average_bits must be in [{minimum}, {maximum}]")
    if max_states < 1:
        raise ValueError("max_states must be positive")

    budget = int(np.floor(float(target_average_bits) * int(counts.sum()) + 1e-9))
    unit = int(counts[0])
    for count in counts[1:]:
        unit = gcd(unit, int(count))
    scaled_counts = counts // unit
    scaled_budget = budget // unit
    losses = _error_matrix(scores, choices, errors)

    # cost -> (loss, selected widths). Only the lowest-loss path per cost survives.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for layer, count in enumerate(scaled_counts):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for old_cost, (old_loss, selected) in states.items():
            for column, bit in enumerate(choices):
                cost = old_cost + int(count) * bit
                if cost > scaled_budget:
                    continue
                loss = old_loss + float(losses[layer, column])
                incumbent = next_states.get(cost)
                if incumbent is None or loss < incumbent[0]:
                    next_states[cost] = (loss, selected + (bit,))
        if not next_states:
            raise ValueError("budget is too small for the minimum precision allocation")
        if len(next_states) > max_states:
            raise RuntimeError(
                "allocation state space exceeded max_states; group equal-sized layers "
                "or increase max_states"
            )
        states = next_states

    if spend_full_budget:
        cost = max(states)
        objective, selected = states[cost]
    else:
        cost, (objective, selected) = min(states.items(), key=lambda item: (item[1][0], -item[0]))
    widths = np.asarray(selected, dtype=np.int64)
    actual_total = int(np.dot(widths, counts))
    return AllocationResult(
        bits=widths,
        average_bits=average_bits(widths, counts),
        total_bits=actual_total,
        budget_bits=budget,
        objective=float(objective),
    )


def random_allocation(
    parameter_counts: Sequence[int],
    target_average_bits: float,
    *,
    bit_choices: Sequence[int] = (3, 4, 8),
    seed: int = 0,
    max_states: int = 200_000,
) -> AllocationResult:
    """Sample a feasible policy that spends the largest attainable matched budget."""

    counts = np.asarray(parameter_counts, dtype=np.int64)
    choices = tuple(sorted(set(int(bit) for bit in bit_choices)))
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        raise ValueError("parameter_counts must be a non-empty positive vector")
    if not choices or choices[0] <= 0:
        raise ValueError("bit choices must be positive integers")
    minimum = choices[0]
    maximum = choices[-1]
    if not minimum <= target_average_bits <= maximum:
        raise ValueError(f"target_average_bits must be in [{minimum}, {maximum}]")
    budget = int(np.floor(float(target_average_bits) * int(counts.sum()) + 1e-9))
    unit = int(counts[0])
    for count in counts[1:]:
        unit = gcd(unit, int(count))
    scaled = counts // unit
    cap = budget // unit

    # Store every feasible predecessor so backward sampling is random while the
    # final cost exactly matches the maximum attainable cost under the cap.
    levels: list[dict[int, list[tuple[int, int]]]] = [{0: []}]
    for count in scaled:
        current: dict[int, list[tuple[int, int]]] = {}
        for old_cost in levels[-1]:
            for bit in choices:
                cost = old_cost + int(count) * bit
                if cost <= cap:
                    current.setdefault(cost, []).append((old_cost, bit))
        if not current:
            raise ValueError("budget is too small for the minimum precision allocation")
        if len(current) > max_states:
            raise RuntimeError("random allocation state space exceeded max_states")
        levels.append(current)
    rng = np.random.default_rng(seed)
    cost = max(levels[-1])
    selected = []
    for layer in range(counts.size, 0, -1):
        candidates = levels[layer][cost]
        old_cost, bit = candidates[int(rng.integers(len(candidates)))]
        selected.append(bit)
        cost = old_cost
    widths = np.asarray(selected[::-1], dtype=np.int64)
    total = int(np.dot(widths, counts))
    return AllocationResult(widths, average_bits(widths, counts), total, budget, float("nan"))
