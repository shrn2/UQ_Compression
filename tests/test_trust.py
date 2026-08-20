import numpy as np

from tuac.artifacts import MultiBitCalibrationArtifact
from tuac.trust import analyze_trust_layers, build_trust_policies


def trust_artifact() -> MultiBitCalibrationArtifact:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(36, 3))
    student = np.exp(logits - logits.max(axis=1, keepdims=True))
    student /= student.sum(axis=1, keepdims=True)
    labels = student.argmax(axis=1)
    labels[::3] = (labels[::3] + 1) % 3
    teacher = 0.85 * student + 0.15 / 3
    probes = np.empty((3, 6, 36, 3))
    for bit_index, scale in enumerate((0.32, 0.15, 0.04)):
        for layer in range(6):
            noise = rng.normal(scale=scale * (layer + 1) / 6, size=student.shape)
            values = np.clip(student + noise, 1e-5, None)
            probes[bit_index, layer] = values / values.sum(axis=1, keepdims=True)
    return MultiBitCalibrationArtifact(
        teacher_probabilities=teacher,
        student_probabilities=student,
        quantized_probabilities=probes,
        probe_bits=(2, 4, 8),
        layer_names=tuple(f"model.layers.{index}" for index in range(6)),
        parameter_counts=np.full(6, 100),
        labels=labels,
        example_ids=tuple(f"e{index}" for index in range(36)),
        metadata={"group_size": 128},
    )


def test_trust_analysis_and_policy_construction(tmp_path):
    artifact = trust_artifact()
    analysis = analyze_trust_layers(
        artifact,
        output_path=tmp_path / "analysis.json",
        bootstrap_replicates=12,
        bootstrap_seed=4,
    )
    assert len(analysis["layer_rows"]) == 18
    assert len(analysis["region_rows"]) == 9
    assert np.asarray(analysis["score_matrices"]["trust_entropy"]).shape == (6, 3)
    analysis["gate"]["passed"] = True
    policies = build_trust_policies(
        artifact, analysis, output_path=tmp_path / "policies.json", budgets=(4.0, 3.0)
    )
    assert set(policies["policies"]) == {
        "capability_kl",
        "capability_flip",
        "capability_accuracy_drop",
        "trust_entropy",
        "joint_0.25",
        "joint_0.50",
        "joint_0.75",
    }
    for policy in policies["policies"].values():
        assert len(policy["allocations"]["3.0"]["bits"]) == 6
