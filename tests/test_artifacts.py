import numpy as np

from tuac.artifacts import CalibrationArtifact, MultiBitCalibrationArtifact


def test_artifact_round_trip(tmp_path):
    artifact = CalibrationArtifact(
        teacher_probabilities=np.array([[0.9, 0.1]]),
        student_probabilities=np.array([[0.8, 0.2]]),
        quantized_probabilities=np.array([[[0.7, 0.3]]]),
        layer_names=("model.layers.0",),
        parameter_counts=np.array([100]),
        labels=np.array([0]),
        example_ids=("one",),
        probe_bits=4,
        metadata={"student_model": "test"},
    )
    path = tmp_path / "artifact.npz"
    artifact.save(path)
    loaded = CalibrationArtifact.load(path)
    assert loaded.layer_names == artifact.layer_names
    assert loaded.metadata == artifact.metadata
    np.testing.assert_allclose(loaded.teacher_probabilities, artifact.teacher_probabilities)


def test_multibit_artifact_round_trip(tmp_path):
    artifact = MultiBitCalibrationArtifact(
        teacher_probabilities=np.array([[0.9, 0.1], [0.4, 0.6]]),
        student_probabilities=np.array([[0.8, 0.2], [0.7, 0.3]]),
        quantized_probabilities=np.array(
            [
                [[[0.6, 0.4], [0.5, 0.5]]],
                [[[0.75, 0.25], [0.65, 0.35]]],
            ]
        ),
        probe_bits=(2, 4),
        layer_names=("model.layers.0",),
        parameter_counts=np.array([100]),
        labels=np.array([0, 1]),
        example_ids=("one", "two"),
        metadata={"iteration": 1},
    )
    path = tmp_path / "multibit.npz"
    artifact.save(path)
    loaded = MultiBitCalibrationArtifact.load(path)
    assert loaded.probe_bits == (2, 4)
    assert loaded.metadata == {"iteration": 1}
    np.testing.assert_allclose(loaded.quantized_probabilities, artifact.quantized_probabilities)
