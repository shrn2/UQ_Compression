import numpy as np

from tuac.stochastic_gate_pruning import (
    hard_topk_straight_through,
    uncertainty_distillation_loss,
)


def test_hard_topk_gate_has_exact_budget_and_gradient():
    import torch

    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3]], requires_grad=True)
    gate, hard = hard_topk_straight_through(logits, 2, temperature=0.5)
    assert hard.sum().item() == 2
    assert gate.detach().tolist() == hard.tolist()
    (gate * torch.arange(4.0)).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_uncertainty_distillation_loss_is_zero_for_matching_predictions():
    import torch

    probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6], [0.55, 0.45]])
    labels = torch.tensor([0, 1, 2])
    loss, components = uncertainty_distillation_loss(probabilities, probabilities, labels)
    assert abs(float(loss)) < 1e-6
    assert abs(float(components["kl"])) < 1e-6


def test_uncertainty_distillation_penalizes_entropy_drift():
    import torch

    targets = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    predictions = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss, components = uncertainty_distillation_loss(predictions, targets, labels)
    assert float(loss) > 0
    assert float(components["entropy_error"]) > 0
    loss.backward()
    assert torch.isfinite(predictions.grad).all()
