"""Structured FFN compression with vanilla KL or Dual-constrained KD."""

from .config import SUPPORTED_DATASETS, SUPPORTED_METHODS, load_config
from .objectives import (
    dual_constrained_distillation_loss,
    forward_kl_distillation_loss,
    update_dual_variables,
)

__all__ = [
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
    "dual_constrained_distillation_loss",
    "forward_kl_distillation_loss",
    "load_config",
    "update_dual_variables",
]

__version__ = "1.0.0"
