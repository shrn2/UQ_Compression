import json

import numpy as np

from tuac.artifacts import MultiBitCalibrationArtifact
from tuac.cascade_compression import (
    analyze_cascade_compression,
    routing_outcomes,
    threshold_for_rate,
)
from tuac.quantization import fake_quantize_tensor, persistent_mixed_quantizer, temporarily_mixed_quantized


def test_threshold_and_routing_outcomes():
    scores = np.arange(10, dtype=float)
    threshold = threshold_for_rate(scores, 0.2)
    assert np.sum(scores > threshold) == 2
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    fallback = np.array([[0.1, 0.9], [0.2, 0.8]])
    metrics, correct, decisions = routing_outcomes(
        probabilities, fallback, np.array([1, 1]), np.array([0.9, 0.1]), 0.5
    )
    assert decisions.tolist() == [True, False]
    assert correct.tolist() == [True, True]
    assert metrics["cascade_accuracy"] == 1.0
    assert metrics["fallback_correction_rate"] == 1.0


def test_stage1_analysis_builds_matched_policies(tmp_path):
    rng = np.random.default_rng(4)
    examples, layers = 20, 4
    labels = np.arange(examples) % 2
    fp = np.stack([np.where(labels == 0, 0.7, 0.3), np.where(labels == 1, 0.7, 0.3)], axis=1)
    fallback = np.stack([np.where(labels == 0, 0.8, 0.2), np.where(labels == 1, 0.8, 0.2)], axis=1)
    probes = []
    for scale in (0.25, 0.08, 0.01):
        rows = []
        for _ in range(layers):
            values = np.clip(fp + rng.normal(0, scale, fp.shape), 1e-4, None)
            rows.append(values / values.sum(axis=1, keepdims=True))
        probes.append(np.stack(rows))
    artifact = MultiBitCalibrationArtifact(
        teacher_probabilities=fallback,
        student_probabilities=fp,
        quantized_probabilities=np.stack(probes),
        probe_bits=(2, 4, 8),
        layer_names=tuple(f"L{i}" for i in range(layers)),
        parameter_counts=np.ones(layers, dtype=np.int64),
        labels=labels,
        example_ids=tuple(str(i) for i in range(examples)),
        metadata={"group_size": 128},
    )
    output = tmp_path / "policies.json"
    result = analyze_cascade_compression(
        artifact,
        output_path=str(output),
        budgets=(3.5, 4.0),
        random_seeds=range(10),
    )
    assert output.exists()
    assert len(result["layer_rows"]) == 12
    for budget in ("3.5", "4.0"):
        totals = {
            tuple(row["bits"])
            for name, row in result["policies"][budget].items()
            if name != "random"
        }
        assert totals
        expected_total = sum(next(iter(totals)))
        assert all(sum(bits) == expected_total for bits in totals)
        assert all(sum(row["bits"]) == expected_total for row in result["policies"][budget]["random"].values())
    loaded = json.loads(output.read_text())
    assert loaded["stage"] == "stage1_local_cuaq"


def test_mixed_quantization_restores_weights_with_cpu_backups():
    import torch

    layer = torch.nn.Linear(8, 4, bias=False)
    original = layer.weight.detach().clone()
    with temporarily_mixed_quantized((("layer", layer),), (2,), group_size=4):
        assert not torch.equal(layer.weight, original)
    torch.testing.assert_close(layer.weight, original)


def test_persistent_quantizer_rewrites_from_fp_and_restores():
    import torch

    layer = torch.nn.Linear(8, 4, bias=False)
    original = layer.weight.detach().clone()
    with persistent_mixed_quantizer((("layer", layer),), group_size=4) as apply:
        apply((2,))
        q2 = layer.weight.detach().clone()
        apply((8,))
        q8 = layer.weight.detach().clone()
        assert not torch.equal(q2, q8)
        # Reapplying Q2 must reproduce direct FP->Q2, not quantize Q8.
        apply((2,))
        torch.testing.assert_close(layer.weight, q2)
    torch.testing.assert_close(layer.weight, original)


def test_vectorized_fake_quantization_matches_group_reference():
    import torch

    tensor = torch.linspace(-2.0, 3.0, 30, dtype=torch.float32).reshape(3, 2, 5)
    matrix = tensor.reshape(3, -1)
    pieces = []
    qmax = 7
    for start in range(0, matrix.shape[1], 4):
        group = matrix[:, start : start + 4]
        scale = group.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        pieces.append(torch.clamp(torch.round(group / scale), -qmax, qmax) * scale)
    expected = torch.cat(pieces, dim=1).reshape(tensor.shape)
    torch.testing.assert_close(fake_quantize_tensor(tensor, 4, group_size=4), expected)
