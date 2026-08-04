import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_id = -1

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if context.is_tree_verify:
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            return self._tree_attention(q, k, v)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

    def _tree_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        batch_size = context.tree_batch_size
        query_width = context.tree_query_width
        if batch_size < 1 or query_width < 1:
            raise ValueError("tree attention requires batch shape metadata")
        if q.size(0) != batch_size * query_width:
            raise ValueError("tree attention query shape mismatch")
        if context.tree_attn_mask is None or context.tree_cached_prompt_lens is None:
            raise ValueError("tree attention requires mask and prompt lengths")

        q_tree = q.view(batch_size, query_width, self.num_heads, self.head_dim)
        k_tree = k.view(batch_size, query_width, self.num_kv_heads, self.head_dim)
        v_tree = v.view(batch_size, query_width, self.num_kv_heads, self.head_dim)
        prompt_width = int(context.tree_cached_prompt_lens.max().item())
        k_prompt = k_tree.new_zeros(
            (batch_size, prompt_width, self.num_kv_heads, self.head_dim)
        )
        v_prompt = v_tree.new_zeros(k_prompt.shape)
        if self.k_cache.numel():
            for request_idx, prompt_len in enumerate(context.tree_cached_prompt_lens.tolist()):
                if prompt_len == 0:
                    continue
                table = context.block_tables[request_idx]
                block_count = (prompt_len + self.k_cache.size(1) - 1) // self.k_cache.size(1)
                blocks = table[:block_count]
                k_prompt[request_idx, :prompt_len] = self.k_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:prompt_len]
                v_prompt[request_idx, :prompt_len] = self.v_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:prompt_len]
        k_full = torch.cat((k_prompt, k_tree), dim=1).transpose(1, 2)
        v_full = torch.cat((v_prompt, v_tree), dim=1).transpose(1, 2)
        q_sdpa = q_tree.transpose(1, 2)
        mask = context.tree_attn_mask.unsqueeze(1)
        if self.num_heads == self.num_kv_heads:
            output = F.scaled_dot_product_attention(
                q_sdpa,
                k_full,
                v_full,
                attn_mask=mask,
                scale=self.scale,
            )
        else:
            if self.num_heads % self.num_kv_heads:
                raise ValueError("query head count must be divisible by KV head count")
            try:
                output = F.scaled_dot_product_attention(
                    q_sdpa,
                    k_full,
                    v_full,
                    attn_mask=mask,
                    scale=self.scale,
                    enable_gqa=True,
                )
            except (TypeError, RuntimeError):
                repeat = self.num_heads // self.num_kv_heads
                output = F.scaled_dot_product_attention(
                    q_sdpa,
                    k_full.repeat_interleave(repeat, dim=1),
                    v_full.repeat_interleave(repeat, dim=1),
                    attn_mask=mask,
                    scale=self.scale,
                )
        if context.tree_kv_stager is not None:
            context.tree_kv_stager.register_cache(self.layer_id, self.k_cache, self.v_cache)
            context.tree_kv_stager.stage(
                self.layer_id,
                k_tree[:, 1:].reshape(-1, self.num_kv_heads, self.head_dim),
                v_tree[:, 1:].reshape(-1, self.num_kv_heads, self.head_dim),
            )
        return output.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
