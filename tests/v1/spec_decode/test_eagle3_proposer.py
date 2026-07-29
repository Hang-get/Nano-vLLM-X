from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from nanovllm.v1.spec_decode.eagle3_proposer import (
    Eagle3Proposer,
    build_draft_prefill_inputs,
    committed_draft_length,
    draft_position_for_target_position,
    select_next_anchor,
)
from nanovllm.v1.spec_decode.types import Eagle3RequestState, SpecReservation


class FakeSequence:
    def __init__(self, seq_id, token_ids, num_computed_tokens, block_table=None):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)
        self.last_token = self.token_ids[-1]
        self.num_computed_tokens = num_computed_tokens
        self.block_table = list(block_table or [0])

    def __len__(self):
        return len(self.token_ids)


class RecordingModel:
    def __init__(self, vocab_size=11, hidden_size=2):
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.calls = []

    def combine_hidden_states(self, hidden):
        self.calls.append(("combine", hidden.detach().clone()))
        return hidden[:, : self.config.hidden_size]

    def __call__(self, input_ids, positions, hidden):
        self.calls.append(
            (
                "forward",
                input_ids.detach().clone(),
                positions.detach().clone(),
                hidden.detach().clone(),
            )
        )
        return hidden, hidden

    def compute_logits(self, hidden):
        return hidden.new_zeros((hidden.size(0), self.config.vocab_size))


class DeterministicProposer(Eagle3Proposer):
    def __init__(self, model, block_size=4):
        super().__init__(model, block_size)
        self.forward_counts = Counter()
        self.positions = []

    def _run_step(
        self,
        seqs,
        reservations,
        active,
        step,
        current_tokens,
        current_hidden,
    ):
        del reservations, current_tokens, current_hidden
        logits = torch.full(
            (len(active), self.target_vocab_size),
            float("-inf"),
        )
        next_hidden = torch.empty((len(active), 2))
        for row, request_idx in enumerate(active):
            seq = seqs[request_idx]
            self.forward_counts[seq.seq_id] += 1
            self.positions.append(
                (
                    seq.seq_id,
                    self.states[seq.seq_id].draft_num_computed_tokens + step,
                )
            )
            token_id = (seq.seq_id + step + 1) % self.target_vocab_size
            logits[row, token_id] = 0
            next_hidden[row].fill_(seq.seq_id + step)
        return logits, next_hidden


def install_states(proposer, seqs):
    for seq in seqs:
        proposer.states[seq.seq_id] = Eagle3RequestState(
            anchor_hidden_states=torch.zeros(6),
            draft_num_computed_tokens=committed_draft_length(
                seq.num_computed_tokens
            ),
        )


def test_pure_alignment_helpers_shift_tokens_features_and_positions():
    auxiliary = torch.arange(4 * 6).reshape(4, 6)

    tokens, features, positions = build_draft_prefill_inputs(
        token_ids=[10, 11, 12, 13],
        target_auxiliary_hidden_states=auxiliary,
    )

    assert tokens == [11, 12, 13]
    torch.testing.assert_close(features, auxiliary[:-1])
    assert positions == [0, 1, 2]
    assert draft_position_for_target_position(1) == 0
    assert draft_position_for_target_position(7) == 6
    assert committed_draft_length(0) == 0
    assert committed_draft_length(8) == 7


def test_one_token_prefill_stores_anchor_without_calling_draft_model():
    model = RecordingModel()
    proposer = Eagle3Proposer(model, block_size=4)
    seq = FakeSequence(1, [10], num_computed_tokens=0)
    auxiliary = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    proposer.prefill([seq], [auxiliary], [1])

    assert model.calls == []
    torch.testing.assert_close(proposer.states[1].anchor_hidden_states, auxiliary[0])
    assert proposer.states[1].draft_num_computed_tokens == 0


def test_shifted_prefill_records_request_major_inputs():
    model = RecordingModel()
    proposer = Eagle3Proposer(model, block_size=4)
    seq = FakeSequence(1, [10, 11, 12, 13], num_computed_tokens=0)
    auxiliary = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    proposer.prefill([seq], [auxiliary], [4])

    assert model.calls[0][0] == "combine"
    forward = model.calls[1]
    assert forward[0] == "forward"
    assert forward[1].tolist() == [11, 12, 13]
    assert forward[2].tolist() == [0, 1, 2]
    assert proposer.states[1].draft_num_computed_tokens == 3


def test_ragged_proposal_is_request_major_and_performs_cache_fill():
    seqs = [
        FakeSequence(1, [1, 2, 3, 4, 5], 4),
        FakeSequence(7, [6, 7, 8, 9, 10], 4),
    ]
    proposer = DeterministicProposer(RecordingModel())
    install_states(proposer, seqs)

    proposal = proposer.propose(
        seqs,
        [SpecReservation(3, []), SpecReservation(1, [])],
        torch.ones(2),
    )

    assert proposal.lengths == [3, 1]
    assert proposer.forward_counts == Counter({1: 4, 7: 2})
    assert proposal.probabilities.shape == (4, 11)
    assert proposal.probabilities.argmax(dim=-1).tolist() == [2, 3, 4, 8]
    assert proposer.positions == [(1, 3), (7, 3), (1, 4), (7, 4), (1, 5), (1, 6)]


def test_ragged_batch_matches_separate_requests():
    seqs = [
        FakeSequence(1, [1, 2, 3, 4, 5], 4),
        FakeSequence(7, [6, 7, 8, 9, 10], 4),
    ]
    batched = DeterministicProposer(RecordingModel())
    install_states(batched, seqs)
    batch_result = batched.propose(
        seqs,
        [SpecReservation(3, []), SpecReservation(1, [])],
        torch.ones(2),
    )

    separate_rows = []
    separate_tokens = []
    for seq, length in zip(seqs, [3, 1]):
        proposer = DeterministicProposer(RecordingModel())
        install_states(proposer, [seq])
        result = proposer.propose(
            [seq], [SpecReservation(length, [])], torch.ones(1)
        )
        separate_tokens.append(result.token_ids[0])
        separate_rows.append(result.probabilities)

    assert batch_result.token_ids == separate_tokens
    torch.testing.assert_close(
        batch_result.probabilities, torch.cat(separate_rows, dim=0)
    )


def test_zero_length_reservation_still_runs_one_cache_fill_step():
    seq = FakeSequence(3, [1, 2, 3], 2)
    proposer = DeterministicProposer(RecordingModel())
    install_states(proposer, [seq])

    proposal = proposer.propose(
        [seq], [SpecReservation(0, [])], torch.ones(1)
    )

    assert proposal.token_ids == [[]]
    assert proposal.probabilities.shape == (0, 11)
    assert proposer.forward_counts == Counter({3: 1})


def test_commit_selects_anchor_and_advances_stable_length():
    seqs = [
        FakeSequence(1, [1, 2, 3, 4, 5], 4),
        FakeSequence(7, [6, 7, 8, 9, 10], 4),
    ]
    proposer = Eagle3Proposer(RecordingModel(), block_size=4)
    install_states(proposer, seqs)
    verification = [torch.arange(18).reshape(3, 6), torch.arange(6).reshape(1, 6)]

    proposer.commit(seqs, verification, [2, 0], [7, 5])

    torch.testing.assert_close(proposer.states[1].anchor_hidden_states, verification[0][2])
    torch.testing.assert_close(proposer.states[7].anchor_hidden_states, verification[1][0])
    assert proposer.states[1].draft_num_computed_tokens == 6
    assert proposer.states[7].draft_num_computed_tokens == 4


def test_commit_is_atomic_on_invalid_length_and_release_is_idempotent():
    seq = FakeSequence(1, [1, 2, 3, 4, 5], 4)
    proposer = Eagle3Proposer(RecordingModel(), block_size=4)
    install_states(proposer, [seq])
    original = proposer.states[1]

    with pytest.raises(ValueError, match="new target computed length"):
        proposer.commit([seq], [torch.zeros(2, 6)], [1], [99])

    assert proposer.states[1] is original
    assert original.draft_num_computed_tokens == 3
    proposer.release([1, 1, 999])
    assert proposer.states == {}


def test_select_next_anchor_uses_accepted_count_as_row_index():
    rows = [torch.arange(18).reshape(3, 6), torch.arange(24).reshape(4, 6)]

    anchors = select_next_anchor(rows, [0, 2])

    torch.testing.assert_close(anchors[0], rows[0][0])
    torch.testing.assert_close(anchors[1], rows[1][2])


def test_public_lifecycle_methods_reject_batch_size_mismatches():
    seq = FakeSequence(1, [1, 2], 1)
    proposer = Eagle3Proposer(RecordingModel(), block_size=4)

    with pytest.raises(ValueError, match="prefill batch size mismatch"):
        proposer.prefill([seq], [], [2])

    install_states(proposer, [seq])
    with pytest.raises(ValueError, match="proposal batch size mismatch"):
        proposer.propose([seq], [], torch.ones(1))
    with pytest.raises(ValueError, match="proposal temperatures must be rank 1"):
        proposer.propose([seq], [SpecReservation(0, [])], torch.ones(1, 1))
    with pytest.raises(ValueError, match="commit batch size mismatch"):
        proposer.commit([seq], [], [0], [2])
