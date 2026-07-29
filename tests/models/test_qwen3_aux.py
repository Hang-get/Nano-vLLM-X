import pytest
import torch
from torch import nn

from nanovllm.models.model_output import TargetModelOutput
from nanovllm.models.qwen3 import Qwen3ForCausalLM, Qwen3Model


class FakeEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, self.hidden_size)


class FakeLayer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, positions, hidden_states, residual):
        del positions
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        return torch.full_like(hidden_states, self.value), residual


class FakeNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


def make_model(hidden_size=2):
    model = Qwen3Model.__new__(Qwen3Model)
    nn.Module.__init__(model)
    model.embed_tokens = FakeEmbedding(hidden_size)
    model.layers = nn.ModuleList([FakeLayer(1), FakeLayer(2), FakeLayer(3)])
    model.norm = FakeNorm()
    return model


def test_qwen3_auxiliary_capture_preserves_final_hidden_states():
    model = make_model()
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])

    normal = model(input_ids, positions)
    captured = model(input_ids, positions, auxiliary_layer_ids=(0, 2))

    assert isinstance(normal, torch.Tensor)
    assert isinstance(captured, TargetModelOutput)
    torch.testing.assert_close(captured.hidden_states, normal)
    assert captured.auxiliary_hidden_states.shape == (2, 4)
    torch.testing.assert_close(
        captured.auxiliary_hidden_states,
        torch.tensor([[2.0, 2.0, 7.0, 7.0], [3.0, 3.0, 8.0, 8.0]]),
    )


@pytest.mark.parametrize("layer_ids", [(1, 1), (2, 0), (-1,), (3,)])
def test_qwen3_auxiliary_capture_rejects_invalid_layer_ids(layer_ids):
    model = make_model()

    with pytest.raises(ValueError):
        model(torch.tensor([1]), torch.tensor([0]), auxiliary_layer_ids=layer_ids)


def test_qwen3_causal_lm_threads_auxiliary_layer_ids():
    wrapper = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = make_model()

    output = wrapper(
        torch.tensor([1]),
        torch.tensor([0]),
        auxiliary_layer_ids=(0, 2),
    )

    assert isinstance(output, TargetModelOutput)
    assert output.auxiliary_hidden_states.shape == (1, 4)
