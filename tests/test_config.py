from types import SimpleNamespace

import pytest

from nanovllm.config import Config, SpeculativeConfig
from nanovllm.v1.spec_decode.eagle3_config import validate_eagle3_checkpoint_pair


TARGET_PATH = "target-checkpoint"
DRAFT_PATH = "draft-checkpoint"


def make_target_config(**overrides):
    values = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "attention_bias": False,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "rope_scaling": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_draft_config(**overrides):
    values = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 1,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "draft_vocab_size": 32000,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def assert_contract_error(role, field, actual, expected, target_config, draft_config):
    path = TARGET_PATH if role == "target" else DRAFT_PATH
    with pytest.raises(ValueError) as exc_info:
        validate_eagle3_checkpoint_pair(
            target_config,
            draft_config,
            target_path=TARGET_PATH,
            draft_path=DRAFT_PATH,
        )

    message = str(exc_info.value)
    assert f"{role} checkpoint {path}" in message
    assert f"{field} expected {expected!r}" in message
    assert f"got {actual!r}" in message


def create_checkpoint_dirs(tmp_path):
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    return target_dir, draft_dir


def install_config_loader(monkeypatch, target_dir, draft_dir, target_config, draft_config):
    configs = {
        str(target_dir): target_config,
        str(draft_dir): draft_config,
    }
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda path: configs[str(path)],
    )


def eagle3_options(draft_dir, **overrides):
    options = {
        "method": "eagle3",
        "draft_model": str(draft_dir),
        "num_speculative_tokens": 5,
    }
    options.update(overrides)
    return options


def test_eagle3_config_accepts_thoughtworks_qwen3_14b(
    tmp_path, monkeypatch, single_rank_dist
):
    target_dir, draft_dir = create_checkpoint_dirs(tmp_path)
    target_config = make_target_config()
    draft_config = make_draft_config()
    install_config_loader(
        monkeypatch, target_dir, draft_dir, target_config, draft_config
    )

    config = Config(
        str(target_dir),
        enforce_eager=True,
        speculative_config=eagle3_options(draft_dir),
    )

    assert config.max_model_len == 4096
    assert config.enable_prefix_cache is False
    assert config.speculative_config.auxiliary_layer_ids == (2, 20, 37)
    assert config.speculative_config.draft_hf_config is draft_config
    assert config.speculative_config.draft_model == str(draft_dir)
    assert config.hf_config.rope_scaling is None


@pytest.mark.parametrize(
    ("role", "field", "actual", "expected"),
    [
        ("target", "hidden_size", 1, 5120),
        ("target", "intermediate_size", 1, 17408),
        ("target", "num_attention_heads", 1, 40),
        ("target", "num_key_value_heads", 1, 8),
        ("target", "head_dim", 1, 128),
        ("target", "vocab_size", 1, 151936),
        ("target", "dtype", "float16", "bfloat16"),
        ("draft", "hidden_size", 1, 5120),
        ("draft", "intermediate_size", 1, 17408),
        ("draft", "num_attention_heads", 1, 40),
        ("draft", "num_key_value_heads", 1, 8),
        ("draft", "head_dim", 1, 128),
        ("draft", "vocab_size", 1, 151936),
        ("draft", "draft_vocab_size", 1, 32000),
        ("draft", "dtype", "float16", "bfloat16"),
    ],
)
def test_eagle3_contract_rejects_incompatible_dimensions_and_dtype(
    role, field, actual, expected
):
    target_config = make_target_config()
    draft_config = make_draft_config()
    if role == "target":
        setattr(target_config, field, actual)
    else:
        setattr(draft_config, field, actual)

    assert_contract_error(
        role, field, actual, expected, target_config, draft_config
    )


@pytest.mark.parametrize(
    ("role", "field", "actual", "expected", "target_config", "draft_config"),
    [
        (
            "target",
            "architecture",
            "LlamaForCausalLM",
            "Qwen3ForCausalLM",
            make_target_config(architectures=["LlamaForCausalLM"]),
            make_draft_config(),
        ),
        (
            "draft",
            "architecture",
            "Qwen3ForCausalLM",
            "LlamaForCausalLM",
            make_target_config(),
            make_draft_config(architectures=["Qwen3ForCausalLM"]),
        ),
        (
            "target",
            "num_hidden_layers",
            39,
            40,
            make_target_config(num_hidden_layers=39),
            make_draft_config(),
        ),
    ],
)
def test_eagle3_contract_rejects_architecture_and_length_mismatches(
    role, field, actual, expected, target_config, draft_config
):
    assert_contract_error(
        role, field, actual, expected, target_config, draft_config
    )


def test_eagle3_config_rejects_missing_draft_directory(tmp_path, monkeypatch):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    missing_draft = tmp_path / "missing-draft"
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda path: make_target_config(),
    )

    with pytest.raises(ValueError) as exc_info:
        Config(
            str(target_dir),
            enforce_eager=True,
            speculative_config=eagle3_options(missing_draft),
        )

    message = str(exc_info.value)
    assert f"draft checkpoint {missing_draft}" in message
    assert "directory expected 'existing directory'" in message
    assert "got 'missing'" in message


def test_eagle3_config_rejects_missing_target_directory(tmp_path):
    missing_target = tmp_path / "missing-target"
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        Config(
            str(missing_target),
            enforce_eager=True,
            speculative_config=eagle3_options(draft_dir),
        )

    message = str(exc_info.value)
    assert f"target checkpoint {missing_target}" in message
    assert "directory expected 'existing directory'" in message
    assert "got 'missing'" in message


def test_eagle3_config_rejects_missing_target_config_file(tmp_path):
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        Config(
            str(target_dir),
            enforce_eager=True,
            speculative_config=eagle3_options(draft_dir),
        )

    message = str(exc_info.value)
    assert f"target checkpoint {target_dir}" in message
    assert "config.json expected 'existing file'" in message
    assert "got 'missing'" in message


def test_eagle3_config_rejects_missing_draft_config_file(tmp_path, monkeypatch):
    target_dir, draft_dir = create_checkpoint_dirs(tmp_path)
    (draft_dir / "config.json").unlink()
    install_config_loader(
        monkeypatch,
        target_dir,
        draft_dir,
        make_target_config(),
        make_draft_config(),
    )

    with pytest.raises(ValueError) as exc_info:
        Config(
            str(target_dir),
            enforce_eager=True,
            speculative_config=eagle3_options(draft_dir),
        )

    message = str(exc_info.value)
    assert f"draft checkpoint {draft_dir}" in message
    assert "config.json expected 'existing file'" in message
    assert "got 'missing'" in message


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"enforce_eager": False}, "enforce_eager expected True, got False"),
        (
            {"enforce_eager": True, "tensor_parallel_size": 2},
            "tensor_parallel_size expected 1, got 2",
        ),
    ],
)
def test_eagle3_config_requires_eager_single_rank(
    tmp_path, monkeypatch, kwargs, expected
):
    target_dir, draft_dir = create_checkpoint_dirs(tmp_path)
    install_config_loader(
        monkeypatch,
        target_dir,
        draft_dir,
        make_target_config(),
        make_draft_config(),
    )

    with pytest.raises(ValueError, match=expected):
        Config(
            str(target_dir),
            speculative_config=eagle3_options(draft_dir),
            **kwargs,
        )


def test_eagle3_config_requires_positive_speculative_token_count():
    with pytest.raises(
        ValueError,
        match="num_speculative_tokens expected at least 1, got 0",
    ):
        SpeculativeConfig(
            method="eagle3",
            draft_model="draft-checkpoint",
            num_speculative_tokens=0,
        )


def test_ngram_config_construction_is_unchanged(tmp_path, monkeypatch):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_config = make_target_config(max_position_embeddings=3072)
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda path: target_config,
    )

    config = Config(
        str(target_dir),
        speculative_config={
            "method": "ngram",
            "prompt_lookup_min": 1,
            "prompt_lookup_max": 2,
            "num_speculative_tokens": 3,
        },
    )

    assert config.max_model_len == 3072
    assert config.enable_prefix_cache is True
    assert config.speculative_config.method == "ngram"
    assert config.speculative_config.draft_model is None
    assert config.speculative_config.draft_hf_config is None
    assert config.speculative_config.auxiliary_layer_ids == ()
