import numpy as np

from tuac.artifacts import MultiBitCalibrationArtifact
from tuac.iteration import analyze_iteration, perturbations, stratified_subset_indices


def synthetic_artifact() -> MultiBitCalibrationArtifact:
    teacher = np.array([[0.9, 0.1], [0.2, 0.8]] * 10)
    student = np.array([[0.55, 0.45], [0.6, 0.4]] * 10)
    labels = np.array([0, 1] * 10)
    probes = []
    for amount in (0.30, 0.15, 0.02):
        layers = []
        for scale in (1.0, 0.5):
            values = student.copy()
            values[:, 0] = np.clip(values[:, 0] - amount * scale, 0.01, 0.99)
            values[:, 1] = 1.0 - values[:, 0]
            layers.append(values)
        probes.append(layers)
    return MultiBitCalibrationArtifact(
        teacher_probabilities=teacher,
        student_probabilities=student,
        quantized_probabilities=np.asarray(probes),
        probe_bits=(2, 4, 8),
        layer_names=("model.layers.0", "model.layers.1"),
        parameter_counts=np.array([100, 100]),
        labels=labels,
        example_ids=tuple(f"example-{index}" for index in range(20)),
        metadata={},
    )


def test_stratified_subsets_are_reproducible_and_sized():
    artifact = synthetic_artifact()
    left = stratified_subset_indices(
        artifact.teacher_probabilities,
        artifact.student_probabilities,
        artifact.labels,
        size=12,
        seed=11,
    )
    right = stratified_subset_indices(
        artifact.teacher_probabilities,
        artifact.student_probabilities,
        artifact.labels,
        size=12,
        seed=11,
    )
    np.testing.assert_array_equal(left, right)
    assert len(left) == len(set(left.tolist())) == 12


def test_iteration_uses_every_empirical_bit_and_builds_policies(tmp_path):
    artifact = synthetic_artifact()
    assert perturbations(artifact).shape == (3, 2, 20)
    report = analyze_iteration(
        artifact,
        output_path=tmp_path / "policies.json",
        subset_size=12,
        calibration_seeds=(11, 23),
        budgets=(4.0, 3.0),
    )
    assert report["probe_bits"] == [2, 4, 8]
    for seed in (11, 23):
        row = report["seeds"][str(seed)]
        assert len(row["subset_indices"]) == 12
        assert set(row["policies"]) == {
            "student_only",
            "teacher_confidence",
            "teacher_uncertainty",
            "disagreement",
            "oracle",
        }
        for policy in row["policies"].values():
            assert np.asarray(policy["empirical_error_matrix"]).shape == (2, 3)
