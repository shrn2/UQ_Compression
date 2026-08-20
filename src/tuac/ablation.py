"""Objective configurations for the UPGD ablation matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    """Weights for the shared stochastic gate-training loop."""

    kl_weight: float = 1.0
    entropy_weight: float = 0.0
    margin_weight: float = 0.0
    robust_weight: float = 0.0


ABLATION_CONFIGS: dict[str, LossConfig] = {
    "kl_only": LossConfig(kl_weight=1.0),
    "kl_entropy": LossConfig(kl_weight=1.0, entropy_weight=1.0),
    "kl_margin": LossConfig(kl_weight=1.0, margin_weight=0.5),
    "kl_entropy_margin": LossConfig(kl_weight=1.0, entropy_weight=1.0, margin_weight=0.5),
    "full_upgd": LossConfig(kl_weight=1.0, entropy_weight=1.0, margin_weight=0.5, robust_weight=0.5),
}
