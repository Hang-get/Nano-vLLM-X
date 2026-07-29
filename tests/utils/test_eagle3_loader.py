from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.models.qwen3_eagle3 import Qwen3Eagle3ForCausalLM
from nanovllm.utils.eagle3_loader import load_eagle3_weights


def make_model():
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        hidden_act="silu",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=1000000,
        rope_scaling=None,
        vocab_size=13,
        draft_vocab_size=4,
    )
    target_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
    return Qwen3Eagle3ForCausalLM(config, target_embedding), target_embedding


def checkpoint_tensors(model):
    tensors = {}
    for name, parameter in model.named_parameters():
        tensor = parameter.detach().cpu().clone()
        if name == "model.embed_tokens.weight":
            continue
        if name == "draft_id_to_target_id":
            tensors["d2t"] = tensor
        elif name == "model.layers.0.self_attn.qkv_proj.weight":
            tensors["midlayer.self_attn.q_proj.weight"] = tensor[:8]
            tensors["midlayer.self_attn.k_proj.weight"] = tensor[8:12]
            tensors["midlayer.self_attn.v_proj.weight"] = tensor[12:16]
        elif name == "model.layers.0.mlp.gate_up_proj.weight":
            tensors["midlayer.mlp.gate_proj.weight"] = tensor[:16]
            tensors["midlayer.mlp.up_proj.weight"] = tensor[16:32]
        elif name.startswith("model.layers.0."):
            tensors[name.replace("model.layers.0.", "midlayer.")] = tensor
        elif name.startswith("model."):
            tensors[name.removeprefix("model.")] = tensor
        else:
            tensors[name] = tensor
    tensors["t2d"] = torch.arange(model.config.vocab_size, dtype=torch.long)
    return tensors


def write_checkpoint(path: Path, tensors):
    path.mkdir()
    save_file(tensors, path / "model.safetensors")


def test_strict_loader_maps_packed_weights_and_injects_embedding(
    tmp_path, single_rank_dist
):
    source, _ = make_model()
    for index, parameter in enumerate(source.parameters()):
        if parameter.is_floating_point():
            parameter.data.fill_(index + 1)
    checkpoint = checkpoint_tensors(source)
    checkpoint["d2t"] = torch.tensor([0, 2, 4, 6])
    draft_dir = tmp_path / "draft"
    write_checkpoint(draft_dir, checkpoint)
    loaded, target_embedding = make_model()

    report = load_eagle3_weights(loaded, str(draft_dir), target_embedding)

    assert "t2d" in report.skipped
    assert report.injected == ("model.embed_tokens.weight",)
    assert report.missing == ()
    assert report.unexpected == ()
    assert loaded.model.embed_tokens is target_embedding
    torch.testing.assert_close(
        loaded.model.layers[0].self_attn.qkv_proj.weight,
        source.model.layers[0].self_attn.qkv_proj.weight,
    )
    torch.testing.assert_close(
        loaded.model.layers[0].mlp.gate_up_proj.weight,
        source.model.layers[0].mlp.gate_up_proj.weight,
    )


@pytest.mark.parametrize("mutation", ["missing_d2t", "unknown", "wrong_shape"])
def test_strict_loader_rejects_incomplete_or_unknown_weights(
    tmp_path, single_rank_dist, mutation
):
    source, _ = make_model()
    tensors = checkpoint_tensors(source)
    if mutation == "missing_d2t":
        tensors.pop("d2t")
    elif mutation == "unknown":
        tensors["unknown.weight"] = torch.zeros(1)
    else:
        tensors["fc.weight"] = tensors["fc.weight"][:-1]
    draft_dir = tmp_path / "draft"
    write_checkpoint(draft_dir, tensors)
    loaded, target_embedding = make_model()

    with pytest.raises(ValueError):
        load_eagle3_weights(loaded, str(draft_dir), target_embedding)


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        (torch.tensor([0, -1, 0, 0]), "must be unique"),
        (torch.tensor([13, 0, 0, 0]), "within target vocabulary"),
    ],
)
def test_strict_loader_rejects_invalid_d2t(
    tmp_path, single_rank_dist, offsets, message
):
    source, _ = make_model()
    tensors = checkpoint_tensors(source)
    tensors["d2t"] = offsets
    draft_dir = tmp_path / "draft"
    write_checkpoint(draft_dir, tensors)
    loaded, target_embedding = make_model()

    with pytest.raises(ValueError, match=message):
        load_eagle3_weights(loaded, str(draft_dir), target_embedding)
