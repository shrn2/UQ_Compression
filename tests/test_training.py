from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dualkd.data import MultipleChoiceExample
from dualkd.training import (
    _choice_probabilities,
    hard_topk_straight_through,
    stratified_epoch_batches,
)


def test_hard_topk_gate_has_exact_budget_and_gradient():
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3]], requires_grad=True)
    gate, hard = hard_topk_straight_through(logits, 2, temperature=0.5)
    assert hard.sum().item() == 2
    assert gate.detach().tolist() == hard.tolist()
    (gate * torch.arange(4.0)).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_exact_epochs_present_every_example_once_per_epoch():
    labels = np.repeat(np.arange(3), 5)
    batches = list(stratified_epoch_batches(labels, epochs=2, per_stratum=2, seed=7))
    counts = np.bincount(np.concatenate(batches), minlength=len(labels))
    assert [len(batch) for batch in batches] == [6, 6, 3, 6, 6, 3]
    assert np.array_equal(counts, np.full(len(labels), 2))


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {
            "Prompt": [0],
            " shared good": [1, 2],
            " shared bad": [1, 3],
        }[text]

    def pad(self, payload, padding=True, return_tensors="pt"):
        values = payload["input_ids"]
        return {
            "input_ids": torch.tensor(values, dtype=torch.long),
            "attention_mask": torch.ones((len(values), len(values[0])), dtype=torch.long),
        }


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=16)

    def __init__(self):
        self.gate = torch.tensor(1.0, requires_grad=True)

    def __call__(self, input_ids, **_kwargs):
        logits = torch.zeros((*input_ids.shape, 5), dtype=torch.float32)
        logits[:, 0, 1] = 2.0
        logits[0, 1, 2] = self.gate
        logits[1, 1, 3] = -self.gate
        return SimpleNamespace(logits=logits)


def test_full_choice_probabilities_use_tokens_after_shared_prefix():
    model = FakeModel()
    scorer = SimpleNamespace(
        torch=torch,
        tokenizer=FakeTokenizer(),
        model=model,
        _logits_keep_parameter=None,
        _input_device=lambda: torch.device("cpu"),
        length_normalize=True,
    )
    example = MultipleChoiceExample(
        id="shared", prompt="Prompt", choices=(" shared good", " shared bad"), answer=0
    )
    probabilities = _choice_probabilities(scorer, [example], choice_count=2)
    assert probabilities[0, 0] > probabilities[0, 1]
    (-probabilities[0, 0].log()).backward()
    assert model.gate.grad is not None and model.gate.grad.abs() > 0


def test_nonfinite_gradient_is_detectable_before_it_poisons_the_gates():
    """The training loop skips a step when clip_grad_norm_ reports a non-finite
    norm. Pin the two torch behaviours that guard depends on: the reported norm
    goes non-finite, and neither clipping nor clamping clears the NaN."""

    logits = torch.nn.Parameter(torch.zeros(1, 4))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    logits.grad = torch.tensor([[float("nan"), 1.0, 1.0, 1.0]])

    grad_norm = torch.nn.utils.clip_grad_norm_([logits], 5.0)
    assert not torch.isfinite(grad_norm)

    optimizer.step()
    with torch.no_grad():
        logits.sub_(logits.mean(dim=1, keepdim=True)).clamp_(-8.0, 8.0)
    assert not torch.isfinite(logits).all()
