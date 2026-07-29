import os
from dataclasses import dataclass
from typing import Any
from transformers import AutoConfig


@dataclass(slots=True)
class SpeculativeConfig:
    method: str = "ngram"
    num_speculative_tokens: int = 3
    prompt_lookup_max: int = 2
    prompt_lookup_min: int = 1

    def __post_init__(self):
        if self.method != "ngram":
            raise ValueError(
                "Unsupported speculative decoding method: "
                f"{self.method}. Supported methods: ['ngram']"
            )
        if self.num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        if self.prompt_lookup_min <= 0:
            raise ValueError("prompt_lookup_min must be positive")
        if self.prompt_lookup_min > self.prompt_lookup_max:
            raise ValueError(
                "prompt_lookup_min must be less than or equal to "
                "prompt_lookup_max"
            )


@dataclass(slots=True)
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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)
