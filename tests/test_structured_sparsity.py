from types import SimpleNamespace

import numpy as np

from tuac.metrics import binary_auroc
from tuac.structured_sparsity import (
    cross_validated_logistic,
    nested_masks,
    predict_logistic,
    prune_ffn_channels,
)


def test_nested_masks_are_ordered_subsets():
    importance = np.asarray([[1, 7, 2, 6, 3, 5, 4, 0], [8, 1, 7, 2, 6, 3, 5, 4]], dtype=float)
    masks = nested_masks(importance, 0.25, 0.50)
    assert masks["medium"].shape == (2, 6)
    assert masks["aggressive"].shape == (2, 4)
    for medium, aggressive in zip(masks["medium"], masks["aggressive"]):
        assert set(aggressive).issubset(set(medium))
        assert np.all(np.diff(medium) > 0)
        assert np.all(np.diff(aggressive) > 0)


def test_physical_ffn_pruning_matches_selected_channel_computation():
    import torch

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 6, bias=False)
            self.up_proj = torch.nn.Linear(4, 6, bias=False)
            self.down_proj = torch.nn.Linear(6, 4, bias=False)
            self.intermediate_size = 6

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    torch.manual_seed(3)
    block = SimpleNamespace(mlp=MLP())
    x = torch.randn(2, 4)
    retained = torch.tensor([0, 2, 5])
    with torch.no_grad():
        hidden = torch.nn.functional.silu(block.mlp.gate_proj(x)) * block.mlp.up_proj(x)
        expected = torch.nn.functional.linear(hidden.index_select(1, retained), block.mlp.down_proj.weight.index_select(1, retained))
    prune_ffn_channels(block, retained.tolist())
    assert block.mlp.gate_proj.out_features == 3
    assert block.mlp.down_proj.in_features == 3
    assert block.mlp.intermediate_size == 3
    torch.testing.assert_close(block.mlp(x), expected)


def test_cross_validated_logistic_learns_rank_signal():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(200, 4))
    labels = features[:, 0] + 0.5 * features[:, 1] > 0
    predictions, folds, model = cross_validated_logistic(features, labels, folds=5, seed=9)
    assert len(folds) == 5
    assert binary_auroc(predictions, labels) > 0.9
    full_predictions = predict_logistic(model, features)
    assert binary_auroc(full_predictions, labels) > 0.9
