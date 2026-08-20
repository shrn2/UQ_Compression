"""PyTorch fake weight quantization used to estimate layer sensitivity.

This module intentionally does not claim packed-kernel speed or storage gains.
It simulates the numerical perturbation used to choose a precision policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PyTorch is required; install tuac[hf]") from exc
    return torch


def transformer_blocks(model) -> list[tuple[str, object]]:
    """Find the ordered transformer block list in common causal LM families."""

    common_paths = (
        "model.layers",       # Llama, Mistral, Qwen2
        "transformer.h",      # GPT-2, Falcon-style wrappers
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
    raise ValueError("could not locate transformer blocks; pass a supported causal LM architecture")


def parameter_count(module) -> int:
    """Count matrix-weight values affected by :func:`temporarily_quantized`."""

    seen = set()
    total = 0
    for child in module.modules():
        weight = getattr(child, "weight", None)
        if weight is None or weight.ndim < 2 or not weight.is_floating_point():
            continue
        identity = id(weight)
        if identity not in seen:
            seen.add(identity)
            total += weight.numel()
    return total


def fake_quantize_tensor(tensor, bits: int, *, group_size: int = 128):
    """Symmetric per-row/per-group fake quantization, returned in source dtype."""

    return fake_quantize_tensor_(tensor.detach().clone(), bits, group_size=group_size)


def fake_quantize_tensor_(tensor, bits: int, *, group_size: int = 128):
    """In-place form of :func:`fake_quantize_tensor` with bounded scratch memory."""

    torch = _torch()
    if bits < 2:
        raise ValueError("bits must be at least 2")
    original_shape = tensor.shape
    if tensor.ndim < 2:
        return tensor
    matrix = tensor.detach().reshape(tensor.shape[0], -1)
    width = matrix.shape[1]
    qmax = (1 << (bits - 1)) - 1
    # Vectorize groups to avoid thousands of tiny accelerator kernels for large
    # transformer matrices. Zero padding cannot increase an absolute group max;
    # the padded tail is removed before restoring the original shape.
    group_count = (width + group_size - 1) // group_size
    padded_width = group_count * group_size
    # Bound float32 scratch space while retaining vectorization across all groups.
    rows_per_chunk = max(1, 4_000_000 // max(1, width))
    for start in range(0, matrix.shape[0], rows_per_chunk):
        chunk = matrix[start : start + rows_per_chunk].float()
        if padded_width != width:
            chunk = torch.nn.functional.pad(chunk, (0, padded_width - width))
        grouped = chunk.reshape(chunk.shape[0], group_count, group_size)
        scale = grouped.abs().amax(dim=2, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = torch.clamp(torch.round(grouped / scale), -qmax, qmax) * scale
        matrix[start : start + chunk.shape[0]].copy_(
            quantized.reshape(chunk.shape[0], padded_width)[:, :width]
        )
    return tensor.reshape(original_shape)


@contextmanager
def temporarily_quantized(
    module, bits: int, *, group_size: int = 128, backup_to_cpu: bool = False
) -> Iterator[None]:
    """Temporarily fake-quantize all matrix weights below a module and restore them."""

    backups = []
    try:
        for child in module.modules():
            weight = getattr(child, "weight", None)
            if weight is None or weight.ndim < 2 or not weight.is_floating_point():
                continue
            original = (
                weight.detach().to(device="cpu", copy=True)
                if backup_to_cpu
                else weight.detach().clone()
            )
            backups.append((weight, original))
            fake_quantize_tensor_(weight.data, bits, group_size=group_size)
        yield
    finally:
        for weight, original in backups:
            weight.data.copy_(original)


@contextmanager
def temporarily_mixed_quantized(
    blocks: Sequence[tuple[str, object]], bits: Sequence[int], *, group_size: int = 128
) -> Iterator[None]:
    if len(blocks) != len(bits):
        raise ValueError("one bit width is required per block")
    managers = []
    try:
        for (_, block), width in zip(blocks, bits):
            # Whole-model mixed policies otherwise retain a second complete model
            # on the accelerator, leaving too little activation memory on 8GB GPUs.
            manager = temporarily_quantized(
                block, int(width), group_size=group_size, backup_to_cpu=True
            )
            manager.__enter__()
            managers.append(manager)
        yield
    finally:
        for manager in reversed(managers):
            manager.__exit__(None, None, None)


@contextmanager
def persistent_mixed_quantizer(
    blocks: Sequence[tuple[str, object]], *, group_size: int = 128
) -> Iterator[object]:
    """Yield an updater that rewrites only blocks whose requested width changes.

    One CPU copy of every floating matrix weight is retained for the lifetime of
    the context. Each update is computed from that FP source, so moving Q4->Q8
    never requantizes an already quantized tensor. This is intended for bounded
    whole-model search where repeatedly entering ``temporarily_mixed_quantized``
    would copy and quantize every block for every nearby candidate.
    """

    backups: list[list[tuple[object, object]]] = []
    current: list[int | None] = [None] * len(blocks)
    for _, block in blocks:
        layer = []
        for child in block.modules():
            weight = getattr(child, "weight", None)
            if weight is None or weight.ndim < 2 or not weight.is_floating_point():
                continue
            layer.append((weight, weight.detach().to(device="cpu", copy=True)))
        backups.append(layer)

    def apply(bits: Sequence[int]) -> None:
        if len(bits) != len(blocks):
            raise ValueError("one bit width is required per block")
        for layer_index, width in enumerate(bits):
            width = int(width)
            if current[layer_index] == width:
                continue
            for weight, original in backups[layer_index]:
                if current[layer_index] is not None:
                    weight.data.copy_(original)
                fake_quantize_tensor_(weight.data, width, group_size=group_size)
            current[layer_index] = width

    try:
        yield apply
    finally:
        for layer in backups:
            for weight, original in layer:
                weight.data.copy_(original)
