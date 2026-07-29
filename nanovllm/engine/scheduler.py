from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.is_spec_decoding = config.speculative_config is not None
        self._num_accepted_draft_tokens = 0
        self._num_proposed_draft_tokens = 0
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    @property
    def acceptance_rate(self) -> float:
        if self._num_proposed_draft_tokens == 0:
            return 0.0
        return self._num_accepted_draft_tokens / self._num_proposed_draft_tokens

    def reset_spec_decode_metrics(self):
        self._num_accepted_draft_tokens = 0
        self._num_proposed_draft_tokens = 0

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def reserve_spec_decode(
        self,
        seqs: list[Sequence],
        draft_token_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[dict[str, list[int] | int]]]:
        reserved_draft_token_ids: list[list[int]] = []
        reservations: list[dict[str, list[int] | int]] = []
        for seq, seq_draft_token_ids in zip(seqs, draft_token_ids):
            max_num_appendable_tokens = self.block_manager.get_num_appendable_tokens(seq)
            reserved_num_draft_tokens = min(
                len(seq_draft_token_ids), max_num_appendable_tokens
            )
            reserved_draft_token_ids.append(
                seq_draft_token_ids[:reserved_num_draft_tokens]
            )
            reservations.append(
                {
                    "draft_len": reserved_num_draft_tokens,
                    "new_block_ids": self.block_manager.reserve_spec_append(
                        seq, reserved_num_draft_tokens
                    ),
                }
            )
        return reserved_draft_token_ids, reservations

    def postprocess_spec_decode(
        self,
        seqs: list[Sequence],
        token_ids: list[list[int]],
        draft_token_ids: list[list[int]],
        reservations: list[dict[str, list[int] | int]],
    ) -> int:
        num_tokens = 0
        for seq, seq_token_ids, seq_draft_token_ids, reservation in zip(
            seqs, token_ids, draft_token_ids, reservations
        ):
            old_len = len(seq)
            accepted_draft_tokens = 0
            for draft_token_id, token_id in zip(seq_draft_token_ids, seq_token_ids):
                if draft_token_id != token_id:
                    break
                accepted_draft_tokens += 1
            self._num_accepted_draft_tokens += accepted_draft_tokens
            self._num_proposed_draft_tokens += len(seq_draft_token_ids)

            for token_id in seq_token_ids:
                if seq.num_completion_tokens >= seq.max_tokens:
                    break
                num_tokens += 1
                seq.append_token(token_id)
                if (
                    (not seq.ignore_eos and token_id == self.eos)
                    or seq.num_completion_tokens >= seq.max_tokens
                ):
                    seq.status = SequenceStatus.FINISHED
                    break

            num_computed_tokens = old_len + accepted_draft_tokens
            self.block_manager.commit_spec_append(
                seq, reservation["new_block_ids"], num_computed_tokens
            )
            seq.num_computed_tokens = num_computed_tokens
            seq.num_cached_tokens = num_computed_tokens
            seq.num_scheduled_tokens = 0

            if seq.is_finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return num_tokens

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_computed_tokens = seq.num_cached_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
