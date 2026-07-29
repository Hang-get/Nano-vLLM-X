import torch
from torch import nn

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3DecoderLayer


class Qwen3Eagle3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        qkv_input_size = 2 * config.hidden_size if layer_idx == 0 else config.hidden_size
        self.self_attn.qkv_proj = QKVParallelLinear(
            qkv_input_size,
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=getattr(config, "attention_bias", False),
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            embeds = self.input_layernorm(embeds)
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Eagle3Model(nn.Module):
    def __init__(self, config, target_embedding: nn.Module):
        super().__init__()
        self.config = config
        self.embed_tokens = target_embedding
        self.fc = ReplicatedLinear(3 * config.hidden_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3Eagle3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embed_tokens(input_ids)
        if hidden_states.shape != embeds.shape:
            raise ValueError(
                "fused hidden states and token embeddings must have identical shape"
            )

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, embeds, hidden_states, residual
            )
        hidden_states, auxiliary_hidden_states = self.norm(hidden_states, residual)
        return hidden_states, auxiliary_hidden_states


class Qwen3Eagle3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, target_embedding: nn.Module):
        super().__init__()
        self._validate_target_embedding(config, target_embedding)
        self.config = config
        self.model = Qwen3Eagle3Model(config, target_embedding)
        self.lm_head = ParallelLMHead(config.draft_vocab_size, config.hidden_size)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )
        target_weight = target_embedding.weight
        self.to(device=target_weight.device, dtype=target_weight.dtype)

    @staticmethod
    def _validate_target_embedding(config, target_embedding: nn.Module) -> None:
        if not hasattr(target_embedding, "weight"):
            raise ValueError("target embedding must expose a weight parameter")
        expected_shape = (config.vocab_size, config.hidden_size)
        if tuple(target_embedding.weight.shape) != expected_shape:
            raise ValueError(
                f"target embedding shape expected {expected_shape}, "
                f"got {tuple(target_embedding.weight.shape)}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        fused_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, fused_hidden_states)

    def combine_hidden_states(
        self, auxiliary_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if auxiliary_hidden_states.size(-1) != 3 * self.config.hidden_size:
            raise ValueError("expected three concatenated target hidden states")
        return self.model.fc(auxiliary_hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        draft_logits = self.lm_head(hidden_states, return_all_logits=True)
        draft_ids = torch.arange(
            self.config.draft_vocab_size,
            device=draft_logits.device,
            dtype=torch.long,
        )
        target_ids = draft_ids + self.draft_id_to_target_id
        logits = draft_logits.new_full(
            (draft_logits.size(0), self.config.vocab_size),
            float("-inf"),
        )
        logits[:, target_ids] = draft_logits
        return logits
