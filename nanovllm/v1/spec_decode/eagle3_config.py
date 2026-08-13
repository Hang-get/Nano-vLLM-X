TARGET_ARCH = "Qwen3ForCausalLM"
DRAFT_ARCH = "LlamaForCausalLM"
AUXILIARY_LAYER_IDS = (2, 20, 37)

_COMMON_TARGET = {
    "architecture": TARGET_ARCH,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 40,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "attention_bias": False,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
    "dtype": "bfloat16",
}

TARGET_MAX_POSITION_EMBEDDINGS = 40960


def _architecture(config, role: str, path: str) -> str:
    architectures = getattr(config, "architectures", None)
    if not architectures:
        raise ValueError(
            f"{role} checkpoint {path}: architectures expected one value, got "
            f"{architectures!r}"
        )
    return architectures[0]


def _dtype_name(config) -> str:
    value = getattr(config, "dtype", None)
    if value is None:
        value = getattr(config, "torch_dtype", None)
    return str(value).removeprefix("torch.")


def _value(config, name: str, default=None):
    return getattr(config, name, default)


def _require(role: str, path: str, name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(
            f"{role} checkpoint {path}: {name} expected {expected!r}, "
            f"got {actual!r}"
        )


def validate_eagle3_checkpoint_pair(
    target_config,
    draft_config,
    *,
    target_path: str,
    draft_path: str,
) -> tuple[int, int, int]:
    target_values = {
        "architecture": _architecture(target_config, "target", target_path),
        "hidden_size": target_config.hidden_size,
        "intermediate_size": target_config.intermediate_size,
        "num_hidden_layers": target_config.num_hidden_layers,
        "num_attention_heads": target_config.num_attention_heads,
        "num_key_value_heads": target_config.num_key_value_heads,
        "head_dim": target_config.head_dim,
        "vocab_size": target_config.vocab_size,
        "max_position_embeddings": _value(target_config, "max_position_embeddings"),
        "bos_token_id": target_config.bos_token_id,
        "eos_token_id": target_config.eos_token_id,
        "attention_bias": _value(target_config, "attention_bias"),
        "hidden_act": _value(target_config, "hidden_act"),
        "rms_norm_eps": _value(target_config, "rms_norm_eps"),
        "tie_word_embeddings": _value(target_config, "tie_word_embeddings"),
        "rope_theta": _value(target_config, "rope_theta"),
        "dtype": _dtype_name(target_config),
    }
    draft_values = {
        "architecture": _architecture(draft_config, "draft", draft_path),
        "hidden_size": draft_config.hidden_size,
        "intermediate_size": draft_config.intermediate_size,
        "num_hidden_layers": draft_config.num_hidden_layers,
        "num_attention_heads": draft_config.num_attention_heads,
        "num_key_value_heads": draft_config.num_key_value_heads,
        "head_dim": draft_config.head_dim,
        "vocab_size": draft_config.vocab_size,
        "draft_vocab_size": draft_config.draft_vocab_size,
        "max_position_embeddings": _value(draft_config, "max_position_embeddings"),
        "bos_token_id": _value(draft_config, "bos_token_id"),
        "eos_token_id": _value(draft_config, "eos_token_id"),
        "attention_bias": _value(draft_config, "attention_bias"),
        "hidden_act": _value(draft_config, "hidden_act"),
        "rms_norm_eps": _value(draft_config, "rms_norm_eps"),
        "rope_theta": _value(draft_config, "rope_theta"),
        "tie_word_embeddings": _value(draft_config, "tie_word_embeddings"),
        "dtype": _dtype_name(draft_config),
    }
    target_expected = {
        **_COMMON_TARGET,
        "max_position_embeddings": TARGET_MAX_POSITION_EMBEDDINGS,
    }
    draft_expected = {
        "architecture": DRAFT_ARCH,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 1,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "draft_vocab_size": 32000,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
    }
    for name, expected in target_expected.items():
        if expected is not None:
            _require("target", target_path, name, target_values[name], expected)
    for name, expected in draft_expected.items():
        if expected is not None:
            _require("draft", draft_path, name, draft_values[name], expected)
    return AUXILIARY_LAYER_IDS
