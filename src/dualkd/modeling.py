"""Transformer and FFN helpers for logical structured channel masks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def transformer_blocks(model) -> list[tuple[str, object]]:
    common_paths = (
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
    )
    for path in common_paths:
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue
        if len(current) > 0:
            return [(f"{path}.{index}", block) for index, block in enumerate(current)]
    raise ValueError("could not locate transformer blocks in the causal LM")


def ffn_modules(block) -> tuple[object, object, object]:
    mlp = getattr(block, "mlp", None)
    if mlp is None or not all(
        hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")
    ):
        raise ValueError("expected a gated FFN with gate_proj, up_proj, and down_proj")
    return mlp.gate_proj, mlp.up_proj, mlp.down_proj


def register_channel_mask_hooks(
    blocks: Sequence[tuple[str, object]], retained_indices: np.ndarray
) -> list[object]:
    """Apply logical masks before every FFN down projection."""

    import torch

    indices = np.asarray(retained_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != len(blocks):
        raise ValueError("one retained-index row is required per transformer block")
    handles: list[object] = []
    for (_, block), retained in zip(blocks, indices):
        _, _, down = ffn_modules(block)
        if retained.ndim != 1 or retained.size < 1 or len(np.unique(retained)) != retained.size:
            raise ValueError("retained channel rows must be non-empty and unique")
        if int(retained.min()) < 0 or int(retained.max()) >= down.in_features:
            raise ValueError("retained channel index is outside the FFN width")
        mask = torch.zeros(down.in_features, device=down.weight.device, dtype=torch.float32)
        mask[torch.as_tensor(retained, dtype=torch.long, device=mask.device)] = 1.0

        def hook(_module, inputs, channel_mask=mask):
            values = inputs[0]
            shape = (1,) * (values.ndim - 1) + (-1,)
            return (
                values
                * channel_mask.to(device=values.device, dtype=values.dtype).view(shape),
            )

        handles.append(down.register_forward_pre_hook(hook))
    return handles
