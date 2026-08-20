import json

import numpy as np

from tuac.capacity_contrast_routing import (
    linear_ccr_scores,
    normalize_question,
    prepare_deduplicated_evaluation,
)


def test_question_normalization_is_format_and_punctuation_invariant():
    left = "Question: Which planet is red?\nOptions:\nA. Earth\nB. Mars\nAnswer:"
    right = "  WHICH planet is red!!!\nAnswer:"
    assert normalize_question(left) == "which planet is red"
    assert normalize_question(left) == normalize_question(right)


def test_deduplication_uses_questions_not_answers(tmp_path):
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.json"
    training.write_text(
        json.dumps({"id": "train", "prompt": "Question: Same text?\nOptions:\nA. x", "choices": [" A", " B"], "answer": 0}) + "\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        "\n".join([
            json.dumps({"id": "duplicate", "prompt": "same TEXT!\nAnswer:", "choices": [" A", " B"], "answer": 1}),
            json.dumps({"id": "fresh", "prompt": "A new question", "choices": [" A", " B"], "answer": 0}),
        ]) + "\n",
        encoding="utf-8",
    )
    result = prepare_deduplicated_evaluation(
        training_data_path=str(training), evaluation_data_path=str(evaluation),
        output_path=str(output), manifest_path=str(manifest),
    )
    assert result["correctness_not_used"] is True
    assert result["exact_duplicates_removed"] == 1
    assert result["final_evaluation_size"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "fresh"


def test_linear_ccr_scores_apply_frozen_normalization_and_logistic_link():
    router = {
        "hidden_mean": np.array([1.0, 2.0]), "hidden_scale": np.array([2.0, 4.0]),
        "scalar_mean": np.array([0.5]), "scalar_scale": np.array([0.5]),
        "weight": np.array([1.0, -2.0, 0.5]), "bias": -0.25,
    }
    hidden = np.array([[3.0, 6.0], [1.0, 2.0]])
    scalar = np.array([[1.0], [0.5]])
    scores = linear_ccr_scores(router, hidden, scalar)
    expected_logits = np.array([-0.75, -0.25])
    np.testing.assert_allclose(scores, 1 / (1 + np.exp(-expected_logits)))
    assert np.all((scores > 0) & (scores < 1))
