from dataclasses import dataclass

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context, set_context
from nanovllm.v1.spec_decode.types import (
    DraftProposal,
    Eagle3RequestState,
    SpecReservation,
    TreeTopology,
)


def draft_position_for_target_position(target_position: int) -> int:
    if target_position < 1:
        raise ValueError("target position must have a predecessor")
    return target_position - 1


def committed_draft_length(target_num_computed_tokens: int) -> int:
    return max(target_num_computed_tokens - 1, 0)


def build_draft_prefill_inputs(
    token_ids: list[int], target_auxiliary_hidden_states: torch.Tensor
) -> tuple[list[int], torch.Tensor, list[int]]:
    if target_auxiliary_hidden_states.ndim != 2:
        raise ValueError("target auxiliary hidden states must be rank 2")
    if target_auxiliary_hidden_states.size(0) != len(token_ids):
        raise ValueError("target auxiliary rows must match token count")
    return (
        token_ids[1:],
        target_auxiliary_hidden_states[:-1],
        list(range(max(len(token_ids) - 1, 0))),
    )


def select_next_anchor(
    verification_auxiliary_hidden_states: list[torch.Tensor],
    accepted_draft_counts: list[int],
) -> list[torch.Tensor]:
    if len(verification_auxiliary_hidden_states) != len(accepted_draft_counts):
        raise ValueError("anchor selection batch size mismatch")
    anchors = []
    for rows, accepted_count in zip(
        verification_auxiliary_hidden_states, accepted_draft_counts
    ):
        if accepted_count < 0 or accepted_count >= rows.size(0):
            raise ValueError("accepted count has no verification auxiliary row")
        anchors.append(rows[accepted_count])
    return anchors


class TreeDraftKVManager:
    """Copy-on-write Draft KV tables for one request's proposed tree.

    A COW block represents the block receiving a node's input token. Earlier
    blocks remain shared with the request's shifted Draft prefill cache.
    """

    def __init__(self, model, base_block_table: list[int], block_size: int, pool: list[int]):
        self.model = model
        self.base_block_table = list(base_block_table)
        self.block_size = block_size
        self.pool = list(pool)

    def _attention_modules(self):
        return [
            module
            for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]

    def fork(self, parent_table: list[int], write_position: int) -> list[int]:
        if not self.pool:
            raise RuntimeError("Draft COW reservation exhausted")
        block_index = write_position // self.block_size
        child_table = list(parent_table)
        while len(child_table) <= block_index:
            child_table.append(-1)
        source_block = child_table[block_index]
        child_block = self.pool.pop(0)
        for module in self._attention_modules():
            if source_block >= 0:
                module.k_cache[child_block].copy_(module.k_cache[source_block])
                module.v_cache[child_block].copy_(module.v_cache[source_block])
            else:
                module.k_cache[child_block].zero_()
                module.v_cache[child_block].zero_()
        child_table[block_index] = child_block
        return child_table

    def commit_table(
        self,
        source_table: list[int],
        destination_table: list[int],
    ) -> None:
        """Promote an accepted branch's Draft KV into persistent cache slots."""
        if len(source_table) > len(destination_table):
            raise ValueError("Draft COW table exceeds persistent block table")
        for block_index, source_block in enumerate(source_table):
            destination_block = destination_table[block_index]
            if source_block < 0 or source_block == destination_block:
                continue
            if destination_block < 0:
                raise ValueError("persistent Draft block table has an empty slot")
            for module in self._attention_modules():
                module.k_cache[destination_block].copy_(module.k_cache[source_block])
                module.v_cache[destination_block].copy_(module.v_cache[source_block])


@dataclass
class _TreeDraftNode:
    request_idx: int
    node_id: int
    parent_id: int
    depth: int
    token_id: int
    hidden: torch.Tensor
    table: list[int]


class Eagle3Proposer:
    def __init__(
        self,
        model,
        block_size: int,
        tree_top_k: int = 1,
        tree_max_depth: int = 0,
        tree_prune_ratio: float = 0.0,
    ):
        self.model = model
        self.block_size = block_size
        self.target_vocab_size = model.config.vocab_size
        self.tree_top_k = tree_top_k
        self.tree_max_depth = tree_max_depth
        self.tree_prune_ratio = tree_prune_ratio
        self.states: dict[int, Eagle3RequestState] = {}
        self._tree_states: list[Eagle3RequestState] = []
        self._tree_kv_managers: dict[int, TreeDraftKVManager] = {}
        self._tree_node_tables: dict[int, dict[int, list[int]]] = {}

    @staticmethod
    def _validate_batch_size(name: str, expected: int, *values) -> None:
        for value in values:
            if len(value) != expected:
                raise ValueError(f"{name} batch size mismatch")

    @staticmethod
    def _assert_stable_state(seq: Sequence, state: Eagle3RequestState) -> None:
        expected = committed_draft_length(seq.num_computed_tokens)
        if not state.valid or state.draft_num_computed_tokens != expected:
            raise ValueError(
                "draft/target computed-token invariant violated: "
                f"expected {expected}, got {state.draft_num_computed_tokens}"
            )

    def prefill(
        self,
        seqs: list[Sequence],
        target_auxiliary_hidden_states: list[torch.Tensor],
        target_num_computed_tokens: list[int],
    ) -> None:
        self._validate_batch_size(
            "prefill",
            len(seqs),
            target_auxiliary_hidden_states,
            target_num_computed_tokens,
        )
        pending_states = {}
        for seq, auxiliary, target_length in zip(
            seqs, target_auxiliary_hidden_states, target_num_computed_tokens
        ):
            if auxiliary.ndim != 2 or auxiliary.size(0) != len(seq):
                raise ValueError("prefill auxiliary rows must match sequence length")
            if target_length != len(seq):
                raise ValueError("prefill target computed length must equal prompt length")
            pending_states[seq.seq_id] = Eagle3RequestState(
                anchor_hidden_states=auxiliary[-1].detach(),
                draft_num_computed_tokens=committed_draft_length(target_length),
            )
        self._run_shifted_prefill(seqs, target_auxiliary_hidden_states)
        self.states.update(pending_states)

    def _run_shifted_prefill(
        self,
        seqs: list[Sequence],
        target_auxiliary_hidden_states: list[torch.Tensor],
    ) -> None:
        input_ids = []
        features = []
        positions = []
        slot_mapping = []
        cu_seqlens = [0]
        max_seqlen = 0
        device = target_auxiliary_hidden_states[0].device if seqs else torch.device("cpu")

        for seq, auxiliary in zip(seqs, target_auxiliary_hidden_states):
            tokens, request_features, request_positions = build_draft_prefill_inputs(
                seq.token_ids, auxiliary
            )
            if not tokens:
                continue
            input_ids.extend(tokens)
            features.append(request_features)
            positions.extend(request_positions)
            cu_seqlens.append(cu_seqlens[-1] + len(tokens))
            max_seqlen = max(max_seqlen, len(tokens))
            for position in request_positions:
                if seq.block_table:
                    block_id = seq.block_table[position // self.block_size]
                    slot_mapping.append(
                        block_id * self.block_size + position % self.block_size
                    )
                else:
                    slot_mapping.append(-1)
        if not input_ids:
            return

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        positions_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        feature_tensor = torch.cat(features, dim=0)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.int32, device=device
        )
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_tensor,
            cu_seqlens_k=cu_seqlens_tensor,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=slot_mapping_tensor,
        )
        try:
            fused_hidden = self.model.combine_hidden_states(feature_tensor)
            self.model(input_ids_tensor, positions_tensor, fused_hidden)
        finally:
            reset_context()

    def propose(
        self,
        seqs: list[Sequence],
        reservations: list[SpecReservation],
        temperatures: torch.Tensor,
    ) -> DraftProposal:
        self._validate_batch_size(
            "proposal", len(seqs), reservations, temperatures
        )
        if temperatures.ndim != 1:
            raise ValueError("proposal temperatures must be rank 1")
        for seq in seqs:
            state = self.states.get(seq.seq_id)
            if state is None:
                raise ValueError(f"missing EAGLE3 state for sequence {seq.seq_id}")
            self._assert_stable_state(seq, state)

        if self.tree_top_k > 1:
            return self._propose_tree(seqs, reservations, temperatures)

        request_tokens = [[] for _ in seqs]
        request_probabilities = [[] for _ in seqs]
        current_tokens = [seq.last_token for seq in seqs]
        current_hidden = [
            self.states[seq.seq_id].anchor_hidden_states for seq in seqs
        ]
        max_steps = max((item.draft_len + 1 for item in reservations), default=0)
        for step in range(max_steps):
            active = [
                index
                for index, reservation in enumerate(reservations)
                if step <= reservation.draft_len
            ]
            logits, next_hidden = self._run_step(
                seqs,
                reservations,
                active,
                step,
                current_tokens,
                current_hidden,
            )
            sampling_rows = [
                row
                for row, request_idx in enumerate(active)
                if step < reservations[request_idx].draft_len
            ]
            if not sampling_rows:
                continue
            sampling_requests = [active[row] for row in sampling_rows]
            active_temperatures = temperatures[sampling_requests].to(torch.float32)
            probabilities = torch.softmax(
                logits[sampling_rows].to(torch.float32)
                / active_temperatures.unsqueeze(-1),
                dim=-1,
            )
            sampled = probabilities.div(
                torch.empty_like(probabilities).exponential_().clamp_min_(1e-10)
            ).argmax(dim=-1)
            for row, request_idx in enumerate(sampling_requests):
                token_id = int(sampled[row].item())
                request_tokens[request_idx].append(token_id)
                request_probabilities[request_idx].append(probabilities[row])
                current_tokens[request_idx] = token_id
                current_hidden[request_idx] = next_hidden[sampling_rows[row]]

        probability_rows = [row for request in request_probabilities for row in request]
        probabilities = (
            torch.stack(probability_rows)
            if probability_rows
            else temperatures.new_empty((0, self.target_vocab_size))
        )
        return DraftProposal(
            token_ids=request_tokens,
            probabilities=probabilities,
            lengths=[len(row) for row in request_tokens],
        )

    def _propose_tree(
        self,
        seqs: list[Sequence],
        reservations: list[SpecReservation],
        temperatures: torch.Tensor,
    ) -> DraftProposal:
        self._tree_states = [self.states[seq.seq_id] for seq in seqs]
        managers: list[TreeDraftKVManager] = []
        for seq, reservation in zip(seqs, reservations):
            if reservation.draft_block_ids is None:
                raise ValueError("tree proposal requires Draft COW block reservations")
            managers.append(
                TreeDraftKVManager(
                    self.model,
                    seq.block_table,
                    self.block_size,
                    reservation.draft_block_ids,
                )
            )

        # The Draft root is positioned one row behind the Target root, matching
        # EAGLE3's shifted feature/cache convention.
        roots = []
        for request_idx, seq in enumerate(seqs):
            state = self.states[seq.seq_id]
            roots.append(
                _TreeDraftNode(
                    request_idx=request_idx,
                    node_id=0,
                    parent_id=-1,
                    depth=0,
                    token_id=seq.last_token,
                    hidden=state.anchor_hidden_states,
                    table=list(seq.block_table),
                )
            )
        root_logits, root_hidden = self._run_tree_step(roots, root=True)

        node_tokens: list[list[int]] = [[] for _ in seqs]
        node_probabilities: list[list[torch.Tensor]] = [[] for _ in seqs]
        parents: list[list[int]] = [[-1] for _ in seqs]
        children: list[list[list[int]]] = [[[]] for _ in seqs]
        depths: list[list[int]] = [[0] for _ in seqs]
        current: list[_TreeDraftNode] = []
        node_tables: list[dict[int, list[int]]] = [
            {0: root.table} for root in roots
        ]

        for root_row, root in enumerate(roots):
            reservation = reservations[root.request_idx]
            max_depth = reservation.effective_tree_max_depth or 0
            if max_depth <= 1:
                continue
            probabilities = torch.softmax(
                root_logits[root_row].to(torch.float32)
                / temperatures[root.request_idx].to(torch.float32),
                dim=-1,
            )
            token_ids = self._select_tree_children(probabilities)
            for token_id in token_ids:
                node_id = len(parents[root.request_idx])
                table = managers[root.request_idx].fork(
                    root.table,
                    self.states[seqs[root.request_idx].seq_id].draft_num_computed_tokens + 1,
                )
                parents[root.request_idx].append(0)
                children[root.request_idx].append([])
                children[root.request_idx][0].append(node_id)
                depths[root.request_idx].append(1)
                node_tokens[root.request_idx].append(token_id)
                node_probabilities[root.request_idx].append(probabilities)
                node_tables[root.request_idx][node_id] = table
                current.append(
                    _TreeDraftNode(
                        root.request_idx,
                        node_id,
                        0,
                        1,
                        token_id,
                        root_hidden[root_row],
                        table,
                    )
                )

        for depth in range(1, self.tree_max_depth):
            active = [
                node
                for node in current
                if (
                    node.depth
                    <= (reservations[node.request_idx].effective_tree_max_depth or 0)
                    - 1
                )
            ]
            if not active:
                break
            logits, next_hidden = self._run_tree_step(active, root=False)
            next_nodes: list[_TreeDraftNode] = []
            for row, node in enumerate(active):
                max_depth = reservations[node.request_idx].effective_tree_max_depth or 0
                if node.depth >= max_depth - 1:
                    continue
                probabilities = torch.softmax(
                    logits[row].to(torch.float32)
                    / temperatures[node.request_idx].to(torch.float32),
                    dim=-1,
                )
                token_ids = self._select_tree_children(probabilities)
                write_position = (
                    self.states[seqs[node.request_idx].seq_id].draft_num_computed_tokens
                    + node.depth
                    + 1
                )
                for token_id in token_ids:
                    node_id = len(parents[node.request_idx])
                    table = managers[node.request_idx].fork(node.table, write_position)
                    parents[node.request_idx].append(node.node_id)
                    children[node.request_idx].append([])
                    children[node.request_idx][node.node_id].append(node_id)
                    depths[node.request_idx].append(node.depth + 1)
                    node_tokens[node.request_idx].append(token_id)
                    node_probabilities[node.request_idx].append(probabilities)
                    node_tables[node.request_idx][node_id] = table
                    next_nodes.append(
                        _TreeDraftNode(
                            node.request_idx,
                            node_id,
                            node.node_id,
                            node.depth + 1,
                            token_id,
                            next_hidden[row],
                            table,
                        )
                    )
            current = next_nodes

        topologies = []
        for request_idx, tokens in enumerate(node_tokens):
            root_position = len(seqs[request_idx]) - 1
            topologies.append(
                TreeTopology(
                    total_nodes=len(parents[request_idx]),
                    draft_nodes=len(tokens),
                    parent=parents[request_idx],
                    children=children[request_idx],
                    depth=depths[request_idx],
                    rope_positions=[
                        root_position + depths[request_idx][node]
                        for node in range(1, len(parents[request_idx]))
                    ],
                    bfs_to_node=list(range(1, len(parents[request_idx]))),
                )
            )
        probability_rows = [
            row for request_rows in node_probabilities for row in request_rows
        ]
        probabilities = (
            torch.stack(probability_rows)
            if probability_rows
            else temperatures.new_empty((0, self.target_vocab_size))
        )
        self._tree_kv_managers = {
            seq.seq_id: manager for seq, manager in zip(seqs, managers)
        }
        self._tree_node_tables = {
            seq.seq_id: tables for seq, tables in zip(seqs, node_tables)
        }
        return DraftProposal(
            token_ids=node_tokens,
            probabilities=probabilities,
            lengths=[len(tokens) for tokens in node_tokens],
            tree_topologies=topologies,
            unused_draft_block_ids=[list(manager.pool) for manager in managers],
        )

    def _select_tree_children(self, probabilities: torch.Tensor) -> list[int]:
        top_probs, top_ids = torch.topk(
            probabilities,
            k=min(self.tree_top_k, probabilities.numel()),
        )
        if self.tree_prune_ratio >= 1:
            top_ids = top_ids[:1]
        elif self.tree_prune_ratio > 0:
            top_ids = top_ids[top_probs >= top_probs[0] * self.tree_prune_ratio]
        return top_ids.tolist()

    def _run_tree_step(
        self,
        nodes: list[_TreeDraftNode],
        *,
        root: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not nodes:
            raise ValueError("tree draft step requires at least one node")
        device = nodes[0].hidden.device
        input_ids = torch.tensor(
            [node.token_id for node in nodes], dtype=torch.long, device=device
        )
        positions = torch.tensor(
            [
                self.states_by_request_position(node, root)
                for node in nodes
            ],
            dtype=torch.long,
            device=device,
        )
        slots = []
        context_lens = []
        tables = []
        for node, position in zip(nodes, positions.tolist()):
            table = node.table
            if position // self.block_size >= len(table):
                raise ValueError("tree draft position exceeds COW block table")
            block_id = table[position // self.block_size]
            if block_id < 0:
                raise ValueError("tree draft table has an unassigned write block")
            slots.append(block_id * self.block_size + position % self.block_size)
            context_lens.append(position + 1)
            tables.append(table)
        max_blocks = max(len(table) for table in tables)
        padded_tables = [table + [-1] * (max_blocks - len(table)) for table in tables]
        set_context(
            False,
            slot_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
            block_tables=torch.tensor(padded_tables, dtype=torch.int32, device=device),
        )
        try:
            hidden = torch.stack([node.hidden for node in nodes])
            if root:
                hidden = self.model.combine_hidden_states(hidden)
            logits_hidden, next_hidden = self.model(input_ids, positions, hidden)
            logits = self.model.compute_logits(logits_hidden)
        finally:
            reset_context()
        return logits, next_hidden

    def states_by_request_position(self, node: _TreeDraftNode, root: bool) -> int:
        state = self.states_for_request(node.request_idx)
        return state.draft_num_computed_tokens + (0 if root else node.depth)

    def states_for_request(self, request_idx: int) -> Eagle3RequestState:
        # `_run_tree_step` receives nodes built only by `_propose_tree`; retain
        # the request/state association without exposing it in public APIs.
        return self._tree_states[request_idx]

    def _run_step(
        self,
        seqs: list[Sequence],
        reservations: list[SpecReservation],
        active: list[int],
        step: int,
        current_tokens: list[int],
        current_hidden: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = current_hidden[active[0]].device
        input_ids = torch.tensor(
            [current_tokens[index] for index in active],
            dtype=torch.long,
            device=device,
        )
        positions = []
        slots = []
        context_lens = []
        block_tables = []
        for index in active:
            state = self.states[seqs[index].seq_id]
            position = state.draft_num_computed_tokens + step
            table = seqs[index].block_table + reservations[index].new_block_ids
            if position // self.block_size >= len(table):
                raise ValueError("draft position exceeds reserved block table")
            positions.append(position)
            slots.append(
                table[position // self.block_size] * self.block_size
                + position % self.block_size
            )
            context_lens.append(position + 1)
            block_tables.append(table)
        max_blocks = max(len(table) for table in block_tables)
        padded_tables = [
            table + [-1] * (max_blocks - len(table)) for table in block_tables
        ]
        positions_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        set_context(
            False,
            slot_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
            context_lens=torch.tensor(
                context_lens, dtype=torch.int32, device=device
            ),
            block_tables=torch.tensor(
                padded_tables, dtype=torch.int32, device=device
            ),
        )
        try:
            hidden = torch.stack([current_hidden[index] for index in active])
            if step == 0:
                hidden = self.model.combine_hidden_states(hidden)
            logits_hidden, next_hidden = self.model(
                input_ids, positions_tensor, hidden
            )
            logits = self.model.compute_logits(logits_hidden)
        finally:
            reset_context()
        return logits, next_hidden

    def commit(
        self,
        seqs: list[Sequence],
        verification_auxiliary_hidden_states: list[torch.Tensor],
        accepted_draft_counts: list[int],
        new_target_num_computed_tokens: list[int],
    ) -> None:
        self._validate_batch_size(
            "commit",
            len(seqs),
            verification_auxiliary_hidden_states,
            accepted_draft_counts,
            new_target_num_computed_tokens,
        )
        anchors = select_next_anchor(
            verification_auxiliary_hidden_states, accepted_draft_counts
        )
        updates = []
        for seq, accepted_count, target_length, anchor in zip(
            seqs,
            accepted_draft_counts,
            new_target_num_computed_tokens,
            anchors,
        ):
            state = self.states.get(seq.seq_id)
            if state is None:
                raise ValueError(f"missing EAGLE3 state for sequence {seq.seq_id}")
            self._assert_stable_state(seq, state)
            expected_target_length = seq.num_computed_tokens + accepted_count + 1
            if target_length != expected_target_length:
                raise ValueError(
                    "new target computed length must advance by pending root plus "
                    "accepted drafts"
                )
            new_draft_length = committed_draft_length(target_length)
            if new_draft_length != state.draft_num_computed_tokens + accepted_count + 1:
                raise ValueError("draft committed length advance is inconsistent")
            updates.append((state, anchor.detach(), new_draft_length))
        for state, anchor, new_draft_length in updates:
            state.anchor_hidden_states = anchor
            state.draft_num_computed_tokens = new_draft_length
            state.valid = True

    def commit_tree(
        self,
        seqs: list[Sequence],
        verification_auxiliary_hidden_states: torch.Tensor,
        result,
        topologies: list[TreeTopology],
        verify_row_indices: list[list[int]],
        new_target_num_computed_tokens: list[int],
        persistent_draft_block_tables: list[list[int]],
    ) -> None:
        """Carry the Target row that generated the next pending root forward."""
        accepted_paths = result.accepted_paths
        if accepted_paths is None:
            raise ValueError("tree commit requires accepted paths")
        self._validate_batch_size(
            "tree commit",
            len(seqs),
            topologies,
            verify_row_indices,
            accepted_paths,
            new_target_num_computed_tokens,
            persistent_draft_block_tables,
        )
        if verification_auxiliary_hidden_states.ndim != 2:
            raise ValueError("tree verification auxiliary states must be rank 2")

        updates = []
        for (
            seq,
            topology,
            row_indices,
            accepted_path,
            accepted_count,
            target_length,
            persistent_table,
        ) in zip(
            seqs,
            topologies,
            verify_row_indices,
            accepted_paths,
            result.accepted_draft_counts,
            new_target_num_computed_tokens,
            persistent_draft_block_tables,
        ):
            if len(accepted_path) != accepted_count:
                raise ValueError("tree accepted path/count mismatch")
            state = self.states.get(seq.seq_id)
            if state is None:
                raise ValueError(f"missing EAGLE3 state for sequence {seq.seq_id}")
            self._assert_stable_state(seq, state)
            expected_target_length = seq.num_computed_tokens + accepted_count + 1
            if target_length != expected_target_length:
                raise ValueError("tree target computed length advance is inconsistent")
            if accepted_path:
                manager = self._tree_kv_managers.get(seq.seq_id)
                node_tables = self._tree_node_tables.get(seq.seq_id)
                if manager is None or node_tables is None:
                    raise ValueError("missing Draft COW state for tree commit")
                manager.commit_table(node_tables[accepted_path[-1]], persistent_table)
            selected_node = accepted_path[-1] if accepted_path else 0
            if selected_node == 0:
                row = row_indices[0]
            else:
                row = row_indices[1 + topology.bfs_to_node.index(selected_node)]
            if row < 0 or row >= verification_auxiliary_hidden_states.size(0):
                raise ValueError("tree anchor row is outside verification output")
            updates.append(
                (
                    state,
                    verification_auxiliary_hidden_states[row].detach(),
                    committed_draft_length(target_length),
                )
            )
        for state, anchor, new_draft_length in updates:
            state.anchor_hidden_states = anchor
            state.draft_num_computed_tokens = new_draft_length
            state.valid = True
        for seq in seqs:
            self._tree_kv_managers.pop(seq.seq_id, None)
            self._tree_node_tables.pop(seq.seq_id, None)

    def release(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            self.states.pop(seq_id, None)
            self._tree_kv_managers.pop(seq_id, None)
            self._tree_node_tables.pop(seq_id, None)
