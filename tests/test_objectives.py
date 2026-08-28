import pytest

torch = pytest.importorskip("torch")

from dualkd.objectives import (
    dual_constrained_distillation_loss,
    forward_kl_distillation_loss,
    update_dual_variables,
)


def test_forward_kl_is_zero_for_matching_distributions():
    probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    loss, components = forward_kl_distillation_loss(probabilities, probabilities)
    assert abs(float(loss)) < 1e-6
    assert abs(float(components["entropy_bias"])) < 1e-6


def test_dual_loss_uses_forward_kl_and_detaches_teacher():
    predictions = torch.tensor([[0.7, 0.3], [0.2, 0.8]], requires_grad=True)
    targets = torch.tensor([[0.6, 0.4], [0.4, 0.6]], requires_grad=True)
    strata = torch.tensor([0, 1])
    dual = torch.tensor(0.25)
    loss, components, entropy_bias = dual_constrained_distillation_loss(
        predictions, targets, strata, dual
    )
    expected_kl = (
        targets.detach() * (targets.detach().log() - predictions.log())
    ).sum(dim=1).mean()
    assert torch.isclose(components["kl"], expected_kl)
    assert torch.isclose(loss, expected_kl + dual * entropy_bias.mean())
    loss.backward()
    assert predictions.grad is not None
    assert targets.grad is None


def test_dual_increases_when_student_entropy_is_too_high():
    dual = torch.tensor(0.0)
    update_dual_variables(dual, torch.tensor([0.2, 0.4]), learning_rate=0.5)
    assert torch.isclose(dual, torch.tensor(0.15))


def test_dual_decreases_when_student_entropy_is_too_low():
    dual = torch.tensor(0.0)
    update_dual_variables(dual, torch.tensor([-0.2, -0.4]), learning_rate=0.5)
    assert torch.isclose(dual, torch.tensor(-0.15))


def test_dual_is_unchanged_when_constraint_is_satisfied():
    dual = torch.tensor(0.75)
    update_dual_variables(dual, torch.tensor([-0.2, 0.2]), learning_rate=0.5)
    assert torch.isclose(dual, torch.tensor(0.75))


def test_per_stratum_duals_do_not_cancel():
    duals = torch.zeros(3)
    bias = torch.tensor([0.3, -0.2, 0.0, 0.1, -0.4, 0.0])
    strata = torch.tensor([0, 1, 2, 0, 1, 2])
    update_dual_variables(
        duals, bias, learning_rate=0.5, stratum_labels=strata
    )
    assert torch.allclose(duals, torch.tensor([0.1, -0.15, 0.0]))


def test_microbatched_backward_matches_full_batch_gradient():
    """Per-microbatch backward must accumulate the exact full-batch gradient.

    The training loop relies on both objectives being means of per-example
    terms, so it weights each microbatch by 1/batch and backwards it alone.
    """

    from dualkd.objectives import per_example_terms

    def gradient(microbatch):
        torch.manual_seed(0)
        parameter = torch.nn.Parameter(torch.randn(6, 4))
        targets = torch.softmax(torch.randn(6, 4), dim=1)
        dual = torch.tensor(0.7)
        for start in range(0, 6, microbatch):
            stop = start + microbatch
            predictions = torch.softmax(parameter[start:stop], dim=1)
            kl, bias = per_example_terms(predictions, targets[start:stop])
            weighted = kl.sum() / 6 + dual * (bias.sum() / 6)
            weighted.backward(retain_graph=True)
        return parameter.grad.clone()

    full = gradient(6)
    for microbatch in (1, 2, 3):
        assert torch.allclose(gradient(microbatch), full, atol=1e-6), microbatch
