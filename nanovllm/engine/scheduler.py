from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.v1.spec_decode.types import (
    DraftProposal,
    SpecDecodeResult,
    SpecDecodeMetrics,
    SpecReservation,
)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.speculative_method = (
            config.speculative_config.method
            if config.speculative_config is not None
            else None
        )
        self.is_spec_decoding = self.speculative_method is not None
        self.num_speculative_tokens = (
            config.speculative_config.num_speculative_tokens
            if config.speculative_config is not None
            else 0
        )
        self._num_accepted_draft_tokens = 0
        self._num_proposed_draft_tokens = 0
        self._num_spec_decode_requests = 0
        self._fallback_decode_count = 0
        self._preempted_seq_ids: list[int] = []
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_cache,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    @property
    def acceptance_rate(self) -> float:
        if self._num_proposed_draft_tokens == 0:
            return 0.0
        return self._num_accepted_draft_tokens / self._num_proposed_draft_tokens

    def reset_spec_decode_metrics(self):
        self._num_accepted_draft_tokens = 0
        self._num_proposed_draft_tokens = 0
        self._num_spec_decode_requests = 0
        self._fallback_decode_count = 0

    def get_spec_decode_metrics(
        self,
        draft_time_ms: float = 0.0,
        verify_time_ms: float = 0.0,
        sampling_time_ms: float = 0.0,
    ) -> SpecDecodeMetrics:
        mean_effective_draft_length = (
            self._num_proposed_draft_tokens / self._num_spec_decode_requests
            if self._num_spec_decode_requests
            else 0.0
        )
        return SpecDecodeMetrics(
            draft_tokens_proposed=self._num_proposed_draft_tokens,
            draft_tokens_accepted=self._num_accepted_draft_tokens,
            mean_effective_draft_length=mean_effective_draft_length,
            fallback_decode_count=self._fallback_decode_count,
            draft_time_ms=draft_time_ms,
            verify_time_ms=verify_time_ms,
            sampling_time_ms=sampling_time_ms,
        )

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if (
                num_batched_tokens + len(seq) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self._preempted_seq_ids.append(seq.seq_id)
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def pop_preempted_seq_ids(self) -> list[int]:
        preempted_seq_ids = list(self._preempted_seq_ids)
        self._preempted_seq_ids.clear()
        return preempted_seq_ids

    def get_eagle3_requested_lengths(
        self,
        seqs: list[Sequence],
    ) -> list[int]:
        requested_lengths = []
        for seq in seqs:
            remaining_output_budget = seq.max_tokens - seq.num_completion_tokens
            remaining_context = self.max_model_len - len(seq)
            requested = max(
                0,
                min(
                    self.num_speculative_tokens,
                    remaining_output_budget - 1,
                    remaining_context,
                ),
            )
            requested_lengths.append(requested)
        return requested_lengths

    def reserve_spec_budget(
        self,
        seqs: list[Sequence],
        requested_lengths: list[int],
    ) -> list[SpecReservation]:
        reservations = []
        for seq, requested in zip(seqs, requested_lengths):
            draft_len = min(
                requested,
                self.block_manager.get_num_appendable_tokens(seq),
            )
            reservations.append(
                SpecReservation(
                    draft_len=draft_len,
                    new_block_ids=self.block_manager.reserve_spec_append(
                        seq,
                        draft_len,
                    ),
                )
            )
        return reservations

    def reserve_spec_decode(
        self,
        seqs: list[Sequence],
        draft_token_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[dict[str, list[int] | int]]]:
        reservations = self.reserve_spec_budget(
            seqs,
            [len(token_ids) for token_ids in draft_token_ids],
        )
        reserved_draft_token_ids = [
            token_ids[:reservation.draft_len]
            for token_ids, reservation in zip(draft_token_ids, reservations)
        ]
        legacy_reservations = [
            {
                "draft_len": reservation.draft_len,
                "new_block_ids": reservation.new_block_ids,
            }
            for reservation in reservations
        ]
        return reserved_draft_token_ids, legacy_reservations

    def postprocess_spec_decode(
        self,
        seqs: list[Sequence],
        result: SpecDecodeResult,
        proposal: DraftProposal | list[list[int]],
        reservations: list[SpecReservation | dict[str, list[int] | int]],
    ) -> int:
        proposal_lengths = (
            proposal.lengths
            if isinstance(proposal, DraftProposal)
            else [len(token_ids) for token_ids in proposal]
        )
        batch_size = len(seqs)
        if not (
            len(result.output_token_ids)
            == len(result.accepted_draft_counts)
            == len(proposal_lengths)
            == len(reservations)
            == batch_size
        ):
            raise ValueError("speculative decode batch size mismatch")
        for accepted_count, proposal_length in zip(
            result.accepted_draft_counts,
            proposal_lengths,
        ):
            if not 0 <= accepted_count <= proposal_length:
                raise ValueError(
                    f"accepted count {accepted_count} outside proposal length "
                    f"{proposal_length}"
                )

        num_tokens = 0
        self._num_spec_decode_requests += batch_size
        self._fallback_decode_count += sum(
            proposal_length == 0 for proposal_length in proposal_lengths
        )
        for (
            seq,
            seq_token_ids,
            accepted_draft_count,
            proposal_length,
            reservation,
        ) in zip(
            seqs,
            result.output_token_ids,
            result.accepted_draft_counts,
            proposal_lengths,
            reservations,
        ):
            old_len = len(seq)
            self._num_proposed_draft_tokens += proposal_length

            num_appended_accepted_drafts = 0
            for token_idx, token_id in enumerate(seq_token_ids):
                num_tokens += 1
                seq.append_token(token_id)
                if token_idx < accepted_draft_count:
                    num_appended_accepted_drafts += 1
                if (
                    (not seq.ignore_eos and token_id == self.eos)
                    or seq.num_completion_tokens >= seq.max_tokens
                ):
                    seq.status = SequenceStatus.FINISHED
                    break

            self._num_accepted_draft_tokens += num_appended_accepted_drafts
            num_computed_tokens = old_len + num_appended_accepted_drafts
            new_block_ids = (
                reservation.new_block_ids
                if isinstance(reservation, SpecReservation)
                else reservation["new_block_ids"]
            )
            assert isinstance(new_block_ids, list)
            self.block_manager.commit_spec_append(
                seq,
                new_block_ids,
                num_computed_tokens,
            )
            seq.num_computed_tokens = num_computed_tokens

            if seq.is_finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return num_tokens

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int] | list[list[int]],
    ) -> int:
        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens = len(seq)
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens >= seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return len(seqs)
