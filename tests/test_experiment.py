import numpy as np

from tuac.artifacts import CalibrationArtifact
from tuac.experiment import SCORE_MODES, analyze_artifact


def test_analysis_builds_every_ablation_and_baseline():
    artifact = CalibrationArtifact(
        teacher_probabilities=np.array([[0.95, 0.05], [0.55, 0.45]]),
        student_probabilities=np.array([[0.40, 0.60], [0.60, 0.40]]),
        quantized_probabilities=np.array(
            [
                [[0.30, 0.70], [0.55, 0.45]],
                [[0.50, 0.50], [0.80, 0.20]],
            ]
        ),
        layer_names=("model.layers.0", "model.layers.1"),
        parameter_counts=np.array([100, 100]),
        labels=np.array([0, 1]),
        example_ids=("one", "two"),
        probe_bits=4,
        metadata={},
    )
    report = analyze_artifact(artifact, budgets=(3.5,), bit_choices=(3, 4))
    assert set(report["policies"]) == set(SCORE_MODES)
    assert len(report["baselines"]["random_mixed_precision"]["3.5"]) == 5
    for mode in SCORE_MODES:
        assert report["policies"][mode]["budgets"]["3.5"]["average_bits"] == 3.5
