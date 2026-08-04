from dataclasses import dataclass, field

import torch

from nanovllm.v1.spec_decode.types import TreeTopology


@dataclass
class TreeTargetKVStager:
    """Owns transient Target K/V until rank acceptance selects a tree path."""

    batch_size: int
    query_width: int
    k_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    v_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def stage(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k_by_layer[layer_id] = k.detach().clone()
        self.v_by_layer[layer_id] = v.detach().clone()

    def register_cache(
        self, layer_id: int, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> None:
        self.caches[layer_id] = (k_cache, v_cache)

    def commit_path(
        self,
        topologies: list[TreeTopology],
        accepted_paths: list[list[int]],
        first_positions: list[int],
        block_tables: list[list[int]],
        block_size: int,
    ) -> None:
        if not (
            len(topologies)
            == len(accepted_paths)
            == len(first_positions)
            == len(block_tables)
            == self.batch_size
        ):
            raise ValueError("tree KV commit batch size mismatch")
        for layer_id, staged_k in self.k_by_layer.items():
            staged_v = self.v_by_layer[layer_id]
            k_cache, v_cache = self.caches[layer_id]
            for request_idx, (topology, path, first_position, table) in enumerate(
                zip(topologies, accepted_paths, first_positions, block_tables)
            ):
                for path_offset, node in enumerate(path):
                    bfs_index = topology.bfs_to_node.index(node)
                    # Staging contains only Draft rows; root K/V was already
                    # written into the primary cache through its valid slot.
                    source = request_idx * (self.query_width - 1) + bfs_index
                    position = first_position + path_offset
                    block_id = table[position // block_size]
                    slot = block_id * block_size + position % block_size
                    k_cache.view(-1, *k_cache.shape[2:])[slot].copy_(staged_k[source])
                    v_cache.view(-1, *v_cache.shape[2:])[slot].copy_(staged_v[source])

    def release(self) -> None:
        self.k_by_layer.clear()
        self.v_by_layer.clear()
        self.caches.clear()
