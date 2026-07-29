from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.v1.spec_decode.types import (
    DraftProposal,
    SpecDecodeResult,
    SpecReservation,
)


@pytest.fixture(autouse=True)
def small_sequence_blocks(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)


def make_scheduler(
    *,
    num_blocks=16,
    block_size=4,
    max_model_len=32,
    speculative_method="eagle3",
    num_speculative_tokens=5,
    eos=99,
):
    speculative_config = SimpleNamespace(
        method=speculative_method,
        num_speculative_tokens=num_speculative_tokens,
    )
    config = SimpleNamespace(
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=max_model_len,
        eos=eos,
        speculative_config=speculative_config,
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=block_size,
        enable_prefix_cache=speculative_method != "eagle3",
    )
    return Scheduler(config)


def make_sequence(token_ids, *, max_tokens=16, ignore_eos=False):
    return Sequence(
        token_ids,
        SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos),
    )


def add_running(scheduler, seqs):
    for seq in seqs:
        scheduler.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)


def test_eagle3_requested_lengths_use_output_and_context_budgets():
    scheduler = make_scheduler(max_model_len=10)
    full_budget = make_sequence([1, 2], max_tokens=10)
    output_limited = make_sequence([3, 4], max_tokens=6)
    output_limited.append_tokens([5, 6])
    no_budget = make_sequence([7, 8], max_tokens=1)

    assert scheduler.speculative_method == "eagle3"
    assert scheduler.get_eagle3_requested_lengths(
        [full_budget, output_limited, no_budget]
    ) == [5, 3, 0]


def test_spec_budget_reservations_consume_capacity_sequentially():
    scheduler = make_scheduler(num_blocks=4)
    seqs = [make_sequence([index, 1, 2]) for index in range(3)]
    add_running(scheduler, seqs)

    reservations = scheduler.reserve_spec_budget(seqs, [5, 3, 0])

    assert reservations == [
        SpecReservation(draft_len=5, new_block_ids=[3]),
        SpecReservation(draft_len=1, new_block_ids=[]),
        SpecReservation(draft_len=0, new_block_ids=[]),
    ]


def test_ngram_reservation_wrapper_preserves_legacy_shape():
    scheduler = make_scheduler(num_blocks=4, speculative_method="ngram")
    seqs = [make_sequence([index, 1, 2]) for index in range(3)]
    add_running(scheduler, seqs)
    draft_token_ids = [
        [10, 11, 12, 13, 14],
        [20, 21, 22],
        [],
    ]

    reserved_token_ids, reservations = scheduler.reserve_spec_decode(
        seqs, draft_token_ids
    )

    assert reserved_token_ids == [draft_token_ids[0], [20], []]
    assert [reservation["draft_len"] for reservation in reservations] == [5, 1, 0]
    assert reservations[0]["new_block_ids"] == [3]

    old_lengths = [len(seq) for seq in seqs]
    result = SpecDecodeResult(
        output_token_ids=[[10], [20, 30], [40]],
        accepted_draft_counts=[0, 1, 0],
    )

    num_tokens = scheduler.postprocess_spec_decode(
        seqs,
        result,
        reserved_token_ids,
        reservations,
    )

    assert num_tokens == 4
    assert [seq.num_computed_tokens for seq in seqs] == [
        old_lengths[0],
        old_lengths[1] + 1,
        old_lengths[2],
    ]
    metrics = scheduler.get_spec_decode_metrics()
    assert metrics.draft_tokens_proposed == 6
    assert metrics.draft_tokens_accepted == 1
    assert metrics.mean_effective_draft_length == pytest.approx(2.0)
    assert metrics.fallback_decode_count == 1


def test_spec_postprocess_uses_explicit_accepted_counts():
    scheduler = make_scheduler()
    seqs = [make_sequence([index, 1, 2]) for index in range(3)]
    add_running(scheduler, seqs)
    proposal = DraftProposal(
        token_ids=[[7], [8, 9], [10]],
        probabilities=torch.zeros((4, 32)),
        lengths=[1, 2, 1],
    )
    reservations = scheduler.reserve_spec_budget(seqs, proposal.lengths)
    old_lengths = [len(seq) for seq in seqs]
    result = SpecDecodeResult(
        output_token_ids=[[7], [8, 9, 20], [30]],
        accepted_draft_counts=[0, 2, 0],
    )

    num_tokens = scheduler.postprocess_spec_decode(
        seqs, result, proposal, reservations
    )

    assert num_tokens == 5
    assert [seq.num_computed_tokens for seq in seqs] == [
        old_lengths[0],
        old_lengths[1] + 2,
        old_lengths[2],
    ]
    assert scheduler.acceptance_rate == pytest.approx(0.5)
    metrics = scheduler.get_spec_decode_metrics(1.0, 2.0, 3.0)
    assert metrics.draft_tokens_proposed == 4
    assert metrics.draft_tokens_accepted == 2
    assert metrics.mean_effective_draft_length == pytest.approx(4 / 3)
    assert metrics.fallback_decode_count == 0
    assert metrics.draft_time_ms == 1.0
    assert metrics.verify_time_ms == 2.0
    assert metrics.sampling_time_ms == 3.0


@pytest.mark.parametrize(
    ("output_token_ids", "max_tokens", "eos"),
    [
        ([99, 11, 12], 16, 99),
        ([10, 11, 12], 1, 99),
    ],
    ids=["eos", "max_tokens"],
)
def test_spec_postprocess_commits_only_accepted_tokens_appended_before_stop(
    monkeypatch,
    output_token_ids,
    max_tokens,
    eos,
):
    scheduler = make_scheduler(eos=eos)
    seq = make_sequence([1, 2, 3], max_tokens=max_tokens)
    add_running(scheduler, [seq])
    proposal = [[output_token_ids[0], output_token_ids[1]]]
    reservations = scheduler.reserve_spec_budget([seq], [2])
    old_length = len(seq)
    committed_lengths = []
    original_commit = scheduler.block_manager.commit_spec_append

    def record_commit(seq, new_block_ids, num_computed_tokens):
        committed_lengths.append(num_computed_tokens)
        original_commit(seq, new_block_ids, num_computed_tokens)

    monkeypatch.setattr(
        scheduler.block_manager,
        "commit_spec_append",
        record_commit,
    )
    result = SpecDecodeResult(
        output_token_ids=[output_token_ids],
        accepted_draft_counts=[2],
    )

    num_tokens = scheduler.postprocess_spec_decode(
        [seq], result, proposal, reservations
    )

    assert num_tokens == 1
    assert seq.completion_token_ids == [output_token_ids[0]]
    assert seq.is_finished
    assert committed_lengths == [old_length + 1]
    assert scheduler.acceptance_rate == pytest.approx(0.5)


def test_spec_postprocess_rejects_count_above_proposal_length():
    scheduler = make_scheduler()
    seq = make_sequence([1, 2, 3])
    add_running(scheduler, [seq])
    reservations = scheduler.reserve_spec_budget([seq], [1])
    result = SpecDecodeResult(
        output_token_ids=[[10]],
        accepted_draft_counts=[2],
    )

    with pytest.raises(ValueError, match="accepted count"):
        scheduler.postprocess_spec_decode(
            [seq], result, [[10]], reservations
        )

    assert seq.completion_token_ids == []


def test_preempted_sequence_ids_are_reported_once():
    scheduler = make_scheduler()
    seq = make_sequence([1, 2, 3])
    add_running(scheduler, [seq])
    scheduler.running.remove(seq)

    scheduler.preempt(seq)

    assert scheduler.pop_preempted_seq_ids() == [seq.seq_id]
    assert scheduler.pop_preempted_seq_ids() == []
