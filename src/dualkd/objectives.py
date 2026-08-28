"""The two supported distillation objectives."""

from __future__ import annotations


def _distributions(predictions, targets):
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("predictions and targets must share [batch, classes] shape")
    tiny = predictions.new_tensor(1e-12)
    student = predictions.clamp_min(tiny)
    student = student / student.sum(dim=1, keepdim=True)
    teacher = targets.detach().clamp_min(tiny)
    teacher = teacher / teacher.sum(dim=1, keepdim=True)
    return student, teacher


def per_example_terms(predictions, targets):
    """Return per-example forward KL and entropy bias, without reducing.

    Both supported objectives are means of per-example terms, so a caller that
    processes a batch in microbatches can weight and backward each piece
    separately and still accumulate the exact full-batch gradient.
    """

    student, teacher = _distributions(predictions, targets)
    kl = (teacher * (teacher.log() - student.log())).sum(dim=1)
    teacher_entropy = -(teacher * teacher.log()).sum(dim=1).detach()
    student_entropy = -(student * student.log()).sum(dim=1)
    return kl, student_entropy - teacher_entropy


def forward_kl_distillation_loss(predictions, targets):
    """Vanilla teacher-to-student KL with diagnostic entropy statistics."""

    kl, entropy_bias = per_example_terms(predictions, targets)
    total = kl.mean()
    return total, {
        "loss": total.detach(),
        "kl": total.detach(),
        "entropy_bias": entropy_bias.mean().detach(),
        "entropy_mae": entropy_bias.abs().mean().detach(),
    }


def dual_constrained_distillation_loss(
    predictions,
    targets,
    stratum_labels,
    dual_variables,
):
    """Forward KL plus a global or entropy-stratum equality Lagrangian."""

    import torch

    if stratum_labels.shape != (predictions.shape[0],):
        raise ValueError("stratum labels must have one entry per example")
    if dual_variables.ndim not in {0, 1}:
        raise ValueError("dual variables must be scalar or one-dimensional")
    kl, entropy_bias = per_example_terms(predictions, targets)
    if dual_variables.ndim == 0:
        dual_term = dual_variables.detach() * entropy_bias.mean()
    else:
        labels = torch.unique(stratum_labels)
        if torch.any(labels < 0) or torch.any(labels >= len(dual_variables)):
            raise ValueError("stratum label is outside the dual-variable range")
        dual_term = torch.stack(
            [
                dual_variables[label].detach()
                * entropy_bias[stratum_labels == label].mean()
                for label in labels
            ]
        ).mean()
    total = kl.mean() + dual_term
    return total, {
        "loss": total.detach(),
        "kl": kl.mean().detach(),
        "entropy_bias": entropy_bias.mean().detach(),
        "entropy_mae": entropy_bias.abs().mean().detach(),
        "dual_term": dual_term.detach(),
    }, entropy_bias


def update_dual_variables(
    dual_variables,
    entropy_bias,
    *,
    learning_rate: float,
    stratum_labels=None,
):
    """Apply detached dual ascent for the entropy equality constraint."""

    import torch

    if learning_rate <= 0:
        raise ValueError("dual learning rate must be positive")
    if dual_variables.ndim not in {0, 1}:
        raise ValueError("dual variables must be scalar or one-dimensional")
    constraint = entropy_bias.detach()
    with torch.no_grad():
        if dual_variables.ndim == 0:
            dual_variables.add_(learning_rate * constraint.mean())
        else:
            if stratum_labels is None or stratum_labels.shape != constraint.shape:
                raise ValueError("per-stratum dual ascent needs matching labels")
            for label in torch.unique(stratum_labels):
                if label < 0 or label >= len(dual_variables):
                    raise ValueError("stratum label is outside the dual-variable range")
                dual_variables[label].add_(
                    learning_rate * constraint[stratum_labels == label].mean()
                )
    return dual_variables
