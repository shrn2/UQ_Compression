from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dualkd.modeling import register_channel_mask_hooks


def test_logical_mask_keeps_only_selected_ffn_channels():
    down = torch.nn.Linear(4, 2, bias=False)
    block = SimpleNamespace(
        mlp=SimpleNamespace(
            gate_proj=SimpleNamespace(), up_proj=SimpleNamespace(), down_proj=down
        )
    )
    values = torch.ones(1, 1, 4)
    handle = register_channel_mask_hooks(
        [("model.layers.0", block)], np.asarray([[0, 2]], dtype=np.int64)
    )[0]
    observed = {}

    def capture(_module, inputs):
        observed["values"] = inputs[0].detach().clone()

    capture_handle = down.register_forward_pre_hook(capture)
    try:
        down(values)
    finally:
        handle.remove()
        capture_handle.remove()
    assert observed["values"].tolist() == [[[1.0, 0.0, 1.0, 0.0]]]
