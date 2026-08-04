from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_tree_verify: bool = False
    tree_attn_mask: torch.Tensor | None = None
    tree_cached_prompt_lens: torch.Tensor | None = None
    tree_batch_size: int = 0
    tree_query_width: int = 0
    tree_root_row_indices: torch.Tensor | None = None
    tree_kv_stager: object | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, *, is_tree_verify=False, tree_attn_mask=None, tree_cached_prompt_lens=None, tree_batch_size=0, tree_query_width=0, tree_root_row_indices=None, tree_kv_stager=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, is_tree_verify, tree_attn_mask, tree_cached_prompt_lens, tree_batch_size, tree_query_width, tree_root_row_indices, tree_kv_stager)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
