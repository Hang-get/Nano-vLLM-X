import pytest
import torch

from nanovllm.v1.spec_decode.types import (
    DraftProposal,
    SpecDecodeResult,
    SpecReservation,
)


def test_draft_proposal_truncate_preserves_request_major_rows():
    proposal = DraftProposal(
        token_ids=[[10, 11, 12], [20, 21]],
        probabilities=torch.arange(5 * 7, dtype=torch.float32).reshape(5, 7),
        lengths=[3, 2],
    )

    truncated = proposal.truncate([2, 1])

    assert truncated.token_ids == [[10, 11], [20]]
    assert truncated.lengths == [2, 1]
    torch.testing.assert_close(
        truncated.probabilities,
        torch.cat([proposal.probabilities[0:2], proposal.probabilities[3:4]]),
    )


def test_draft_proposal_rejects_token_length_mismatch():
    with pytest.raises(ValueError, match="lengths must match token_ids"):
        DraftProposal(
            token_ids=[[1, 2]],
            probabilities=torch.zeros(2, 4),
            lengths=[1],
        )


def test_draft_proposal_rejects_probability_row_mismatch():
    with pytest.raises(
        ValueError, match=r"probability rows must equal sum\(lengths\)"
    ):
        DraftProposal(
            token_ids=[[1, 2]],
            probabilities=torch.zeros(1, 4),
            lengths=[2],
        )


def test_draft_proposal_rejects_non_matrix_probabilities():
    with pytest.raises(ValueError, match="probabilities must be rank 2"):
        DraftProposal(
            token_ids=[[1]],
            probabilities=torch.zeros(1),
            lengths=[1],
        )


def test_draft_proposal_truncate_rejects_invalid_lengths():
    proposal = DraftProposal(
        token_ids=[[1, 2], [3]],
        probabilities=torch.zeros(3, 4),
        lengths=[2, 1],
    )

    with pytest.raises(ValueError, match="new_lengths batch size mismatch"):
        proposal.truncate([1])
    with pytest.raises(ValueError, match="new length exceeds proposed length"):
        proposal.truncate([3, 1])
    with pytest.raises(ValueError, match="new length exceeds proposed length"):
        proposal.truncate([-1, 1])


def test_spec_decode_result_rejects_accepted_count_batch_mismatch():
    with pytest.raises(ValueError, match="accepted-count batch size mismatch"):
        SpecDecodeResult(output_token_ids=[[1]], accepted_draft_counts=[])


def test_spec_decode_result_rejects_negative_accepted_count():
    with pytest.raises(ValueError, match="accepted counts must be non-negative"):
        SpecDecodeResult(output_token_ids=[[1]], accepted_draft_counts=[-1])


def test_spec_reservation_rejects_negative_draft_length():
    with pytest.raises(ValueError, match="draft_len must be non-negative"):
        SpecReservation(draft_len=-1, new_block_ids=[])
