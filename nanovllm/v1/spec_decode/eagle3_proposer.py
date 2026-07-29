import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context, set_context
from nanovllm.v1.spec_decode.types import (
    DraftProposal,
    Eagle3RequestState,
    SpecReservation,
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


class Eagle3Proposer:
    def __init__(self, model, block_size: int):
        self.model = model
        self.block_size = block_size
        self.target_vocab_size = model.config.vocab_size
        self.states: dict[int, Eagle3RequestState] = {}

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

    def release(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            self.states.pop(seq_id, None)
