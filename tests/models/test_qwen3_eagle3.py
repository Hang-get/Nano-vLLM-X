from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.models.qwen3_eagle3 import Qwen3Eagle3ForCausalLM


def make_config():
    return SimpleNamespace(
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
        architectures=["LlamaForCausalLM"],
    )


def make_model():
    config = make_config()
    target_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
    return Qwen3Eagle3ForCausalLM(config, target_embedding), target_embedding


def test_draft_model_shares_target_embedding_and_uses_double_width_qkv(
    single_rank_dist,
):
    model, target_embedding = make_model()

    assert model.model.embed_tokens is target_embedding
    assert model.model.embed_tokens.weight is target_embedding.weight
    assert model.model.layers[0].self_attn.qkv_proj.weight.shape[1] == 16


def test_combine_hidden_states_requires_three_target_features(single_rank_dist):
    model, _ = make_model()
    features = torch.arange(2 * 24, dtype=torch.float32).reshape(2, 24)

    output = model.combine_hidden_states(features)

    assert output.shape == (2, 8)
    with pytest.raises(
        ValueError, match="expected three concatenated target hidden states"
    ):
        model.combine_hidden_states(torch.zeros(2, 16))


def test_compute_logits_scatters_draft_vocabulary_with_d2t(single_rank_dist):
    model, _ = make_model()
    model.draft_id_to_target_id.copy_(torch.tensor([0, 2, 4, 6]))
    known_weight = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    model.lm_head.weight.data.copy_(known_weight)
    hidden_states = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)

    logits = model.compute_logits(hidden_states)

    assert logits.shape == (2, 13)
    targets = torch.tensor([0, 3, 6, 9])
    torch.testing.assert_close(logits[:, targets], F.linear(hidden_states, known_weight))
    uncovered = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11, 12])
    assert torch.isneginf(logits[:, uncovered]).all()


def test_target_embedding_shape_is_validated_before_model_construction(
    single_rank_dist,
):
    config = make_config()
    wrong_embedding = VocabParallelEmbedding(12, config.hidden_size)

    with pytest.raises(ValueError, match="target embedding shape expected"):
        Qwen3Eagle3ForCausalLM(config, wrong_embedding)


def test_thoughtworks_checkpoint_uses_int32_d2t_mapping(single_rank_dist):
    config = make_config()
    model = Qwen3Eagle3ForCausalLM(
        config,
        VocabParallelEmbedding(config.vocab_size, config.hidden_size),
    )

    assert model.draft_id_to_target_id.dtype == torch.int32
    model.draft_id_to_target_id.copy_(torch.tensor([0, 2, 4, 6], dtype=torch.int32))
    logits = model.compute_logits(torch.zeros((1, config.hidden_size)))

    assert logits.shape == (1, config.vocab_size)
