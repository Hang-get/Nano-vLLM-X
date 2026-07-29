from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpecReservation:
    draft_len: int
    new_block_ids: list[int]

    def __post_init__(self):
        if self.draft_len < 0:
            raise ValueError("draft_len must be non-negative")


@dataclass
class DraftProposal:
    token_ids: list[list[int]]
    probabilities: torch.Tensor
    lengths: list[int]

    def __post_init__(self):
        if self.lengths != [len(row) for row in self.token_ids]:
            raise ValueError("lengths must match token_ids")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must be rank 2")
        if self.probabilities.size(0) != sum(self.lengths):
            raise ValueError("probability rows must equal sum(lengths)")

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
        return DraftProposal(token_ids, probabilities, new_lengths)


@dataclass
class SpecDecodeResult:
    output_token_ids: list[list[int]]
    accepted_draft_counts: list[int]

    def __post_init__(self):
        if len(self.output_token_ids) != len(self.accepted_draft_counts):
            raise ValueError("accepted-count batch size mismatch")
        if any(count < 0 for count in self.accepted_draft_counts):
            raise ValueError("accepted counts must be non-negative")


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
