import numpy as np

from tuac.scoring import compute_importance, entropy, example_importance, kl_divergence


def test_combined_score_prioritizes_confident_teacher_student_disagreement():
    teacher = np.array([[0.99, 0.01], [0.50, 0.50]])
    student = np.array([[0.10, 0.90], [0.10, 0.90]])
    weights, uncertainty, confidence, disagreement = example_importance(
        teacher, student, mode="combined", normalize_weights=False
    )
    assert weights[0] > weights[1]
    assert confidence[0] > confidence[1]
    assert uncertainty[0] < uncertainty[1]
    assert disagreement[0] > 0


def test_student_only_importance_is_mean_perturbation():
    teacher = np.array([[0.9, 0.1], [0.6, 0.4]])
    student = np.array([[0.8, 0.2], [0.7, 0.3]])
    quantized = np.array(
        [
            [[0.7, 0.3], [0.6, 0.4]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )
    result = compute_importance(teacher, student, quantized, mode="student_only")
    expected = np.array([kl_divergence(student, layer).mean() for layer in quantized])
    np.testing.assert_allclose(result.layer_importance, expected)


def test_normalized_entropy_handles_zero_padding():
    values = np.array([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    np.testing.assert_allclose(entropy(values, normalized=True), [1.0, 1.0])

