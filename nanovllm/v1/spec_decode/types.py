from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpecReservation:
    draft_len: int
    new_block_ids: list[int]
    max_path_draft_len: int | None = None
    effective_tree_max_depth: int | None = None
    draft_block_ids: list[int] | None = None
    target_block_ids: list[int] | None = None

    def __post_init__(self):
        if self.draft_len < 0:
            raise ValueError("draft_len must be non-negative")
        if self.max_path_draft_len is not None and self.max_path_draft_len < 0:
            raise ValueError("max_path_draft_len must be non-negative")
        if (
            self.effective_tree_max_depth is not None
            and self.effective_tree_max_depth < 0
        ):
            raise ValueError("effective_tree_max_depth must be non-negative")


@dataclass
class TreeTopology:
    total_nodes: int
    draft_nodes: int
    parent: list[int]
    children: list[list[int]]
    depth: list[int]
    rope_positions: list[int]
    bfs_to_node: list[int]

    def __post_init__(self):
        if self.total_nodes != len(self.parent) or self.total_nodes != len(self.children):
            raise ValueError("tree parent/children must match total_nodes")
        if self.total_nodes != len(self.depth) or self.parent[0] != -1:
            raise ValueError("tree root must have parent -1")
        if self.draft_nodes != self.total_nodes - 1:
            raise ValueError("draft_nodes must exclude exactly one root")
        if len(self.rope_positions) != self.draft_nodes:
            raise ValueError("rope positions must contain one entry per draft node")
        if len(self.bfs_to_node) != self.draft_nodes:
            raise ValueError("BFS order must contain one entry per draft node")
        for node, parent in enumerate(self.parent[1:], start=1):
            if not 0 <= parent < node:
                raise ValueError("tree parents must precede children in BFS order")
            if node not in self.children[parent]:
                raise ValueError("parent/children relation is inconsistent")

    def build_tree_internal_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.draft_nodes, self.draft_nodes), dtype=torch.bool)
        for row, node in enumerate(self.bfs_to_node):
            ancestor = node
            while ancestor != -1:
                if ancestor != 0:
                    try:
                        column = self.bfs_to_node.index(ancestor)
                    except ValueError as error:
                        raise ValueError("BFS order is missing a tree node") from error
                    mask[row, column] = True
                ancestor = self.parent[ancestor]
        return mask


@dataclass
class DraftProposal:
    token_ids: list[list[int]]
    probabilities: torch.Tensor
    lengths: list[int]
    tree_topologies: list[TreeTopology | None] | None = None
    unused_draft_block_ids: list[list[int]] | None = None

    def __post_init__(self):
        if self.lengths != [len(row) for row in self.token_ids]:
            raise ValueError("lengths must match token_ids")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must be rank 2")
        if self.probabilities.size(0) != sum(self.lengths):
            raise ValueError("probability rows must equal sum(lengths)")
        if self.tree_topologies is not None:
            if len(self.tree_topologies) != len(self.lengths):
                raise ValueError("tree topology batch size mismatch")
            for length, topology in zip(self.lengths, self.tree_topologies):
                if topology is not None and topology.draft_nodes != length:
                    raise ValueError("tree topology draft node count mismatch")
        if self.unused_draft_block_ids is not None and len(
            self.unused_draft_block_ids
        ) != len(self.lengths):
            raise ValueError("unused Draft block batch size mismatch")

    def truncate(self, new_lengths: list[int]) -> "DraftProposal":
        if len(new_lengths) != len(self.lengths):
            raise ValueError("new_lengths batch size mismatch")
        rows = []
        token_ids = []
        offset = 0
        for old_len, new_len, request_tokens in zip(
            self.lengths, new_lengths, self.token_ids
        ):
            if not 0 <= new_len <= old_len:
                raise ValueError("new length exceeds proposed length")
            rows.append(self.probabilities[offset : offset + new_len])
            token_ids.append(request_tokens[:new_len])
            offset += old_len
        probabilities = torch.cat(rows, dim=0) if rows else self.probabilities[:0]
        if self.tree_topologies is not None and any(
            topology is not None for topology in self.tree_topologies
        ):
            raise ValueError("tree proposals cannot be linearly truncated")
        return DraftProposal(
            token_ids,
            probabilities,
            new_lengths,
            self.tree_topologies,
            self.unused_draft_block_ids,
        )


@dataclass
class SpecDecodeResult:
    output_token_ids: list[list[int]]
    accepted_draft_counts: list[int]
    accepted_paths: list[list[int] | None] | None = None

    def __post_init__(self):
        if len(self.output_token_ids) != len(self.accepted_draft_counts):
            raise ValueError("accepted-count batch size mismatch")
        if any(count < 0 for count in self.accepted_draft_counts):
            raise ValueError("accepted counts must be non-negative")
        if self.accepted_paths is not None and len(self.accepted_paths) != len(
            self.accepted_draft_counts
        ):
            raise ValueError("accepted-path batch size mismatch")


@dataclass(frozen=True)
class SpecDecodeMetrics:
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    mean_effective_draft_length: float
    fallback_decode_count: int
    draft_time_ms: float
    verify_time_ms: float
    sampling_time_ms: float


@dataclass
class Eagle3RequestState:
    anchor_hidden_states: torch.Tensor
    draft_num_computed_tokens: int
    valid: bool = True
