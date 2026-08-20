"""Teacher-uncertainty-aware compression (TUAC)."""

from .allocation import AllocationResult, allocate_bits, average_bits, random_allocation
from .metrics import classification_metrics, uncertainty_preservation
from .scoring import ImportanceResult, compute_importance, kl_divergence

__all__ = [
    "AllocationResult",
    "ImportanceResult",
    "allocate_bits",
    "average_bits",
    "classification_metrics",
    "compute_importance",
    "kl_divergence",
    "random_allocation",
    "uncertainty_preservation",
]

__version__ = "0.1.0"
