import numpy as np
import pytest

from nanovllm.v1.spec_decode.ngram_proposer import NgramProposer


def test_proposes_continuation_after_repeated_suffix():
    proposer = NgramProposer(
        prompt_lookup_min=2,
        prompt_lookup_max=3,
        num_speculative_tokens=2,
        max_model_len=16,
        max_num_seqs=2,
    )
    token_ids = np.zeros((1, 16), dtype=np.int32)
    token_ids[0, :8] = [1, 2, 3, 4, 1, 2, 3, 4]

    proposals = proposer.propose(np.array([8], dtype=np.int32), token_ids)

    assert proposals == [[1, 2]]


def test_returns_no_proposal_without_a_repeated_ngram():
    proposer = NgramProposer(
        prompt_lookup_min=2,
        prompt_lookup_max=3,
        num_speculative_tokens=2,
        max_model_len=16,
        max_num_seqs=1,
    )
    token_ids = np.zeros((1, 16), dtype=np.int32)
    token_ids[0, :5] = [1, 2, 3, 4, 5]

    assert proposer.propose(np.array([5], dtype=np.int32), token_ids) == [[]]


def test_rejects_mismatched_batch_sizes():
    proposer = NgramProposer(
        max_model_len=8,
        max_num_seqs=2,
    )

    with pytest.raises(ValueError, match="batch size"):
        proposer.propose(
            np.array([1, 2], dtype=np.int32),
            np.zeros((1, 8), dtype=np.int32),
        )
