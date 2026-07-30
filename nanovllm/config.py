import os
from dataclasses import dataclass, field
from transformers import AutoConfig
from typing import Any

from nanovllm.v1.spec_decode.eagle3_config import (
    validate_eagle3_checkpoint_pair,
)

@dataclass
class SpeculativeConfig:
    method: str = "ngram"
    num_speculative_tokens: int = 3
    prompt_lookup_max: int = 2
    prompt_lookup_min: int = 1
    draft_model: str | None = None
    draft_hf_config: AutoConfig | None = field(init=False, default=None)
    auxiliary_layer_ids: tuple[int, ...] = field(init=False, default=())

    def __post_init__(self):
        supported_methods: list[str] = ["ngram", "eagle3"]
        if self.method not in supported_methods:
            raise ValueError(
                f"Unsupported speculative decoding method: {self.method}. "
                f"Supported methods: {supported_methods}"
            )
        if self.prompt_lookup_min > self.prompt_lookup_max:
            raise ValueError(
                "prompt_lookup_min must be less than or equal to prompt_lookup_max, "
                f"got {self.prompt_lookup_min} > {self.prompt_lookup_max}"
            )
        if self.method == "eagle3" and self.num_speculative_tokens < 1:
            raise ValueError(
                "num_speculative_tokens expected at least 1, "
                f"got {self.num_speculative_tokens}"
            )

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    speculative_config: dict[str, Any] | SpeculativeConfig | None = None
    enable_prefix_cache: bool = True

    def __post_init__(self):
        if not os.path.isdir(self.model):
            if self._is_eagle3_config():
                raise ValueError(
                    f"target checkpoint {self.model}: directory expected "
                    "'existing directory', got 'missing'"
                )
            raise AssertionError(f"model directory does not exist: {self.model}")
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self._is_eagle3_config():
            self._require_checkpoint_file("target", self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)
        if (
            isinstance(self.speculative_config, SpeculativeConfig)
            and self.speculative_config.method == "eagle3"
        ):
            self._configure_eagle3(self.speculative_config)
        self.hf_config.rope_scaling = None          # disable auto rope scaling
        assert self.max_num_batched_tokens >= self.max_model_len

    def _is_eagle3_config(self) -> bool:
        if isinstance(self.speculative_config, SpeculativeConfig):
            return self.speculative_config.method == "eagle3"
        return (
            isinstance(self.speculative_config, dict)
            and self.speculative_config.get("method") == "eagle3"
        )

    def _configure_eagle3(self, speculative_config: SpeculativeConfig) -> None:
        self._require_checkpoint_file("target", self.model)
        draft_model = speculative_config.draft_model
        if not draft_model or not os.path.isdir(draft_model):
            raise ValueError(
                f"draft checkpoint {draft_model}: directory expected "
                "'existing directory', got 'missing'"
            )
        self._require_checkpoint_file("draft", draft_model)
        if not self.enforce_eager:
            raise ValueError("enforce_eager expected True, got False")
        if self.tensor_parallel_size != 1:
            raise ValueError(
                "tensor_parallel_size expected 1, "
                f"got {self.tensor_parallel_size}"
            )

        draft_hf_config = AutoConfig.from_pretrained(draft_model)
        speculative_config.auxiliary_layer_ids = validate_eagle3_checkpoint_pair(
            self.hf_config,
            draft_hf_config,
            target_path=self.model,
            draft_path=draft_model,
        )
        draft_hf_config.rope_scaling = None
        self.hf_config.rope_scaling = None          # disable auto rope scaling
        speculative_config.draft_hf_config = draft_hf_config
        self.enable_prefix_cache = False
        self.max_model_len = min(
            self.max_model_len,
            draft_hf_config.max_position_embeddings,
        )

    @staticmethod
    def _require_checkpoint_file(role: str, path: str) -> None:
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            raise ValueError(
                f"{role} checkpoint {path}: config.json expected "
                f"'existing file', got 'missing'"
            )
