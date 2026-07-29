from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler():
    Sequence.block_size = 4
    return Scheduler(
        SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=64,
            eos=99,
            kvcache_block_size=4,
            num_kvcache_blocks=8,
            speculative_config={"method": "ngram"},
        )
    )


def make_running_sequence(tokens, max_tokens=8):
    sequence = Sequence(tokens, SamplingParams(max_tokens=max_tokens))
    sequence.status = SequenceStatus.RUNNING
    sequence.is_prefill = False
    return sequence


def test_postprocess_spec_decode_commits_accepted_draft_prefix():
    scheduler = make_scheduler()
    sequence = make_running_sequence([1, 2, 3, 4])
    scheduler.block_manager.allocate(sequence, 0)
    sequence.num_cached_tokens = len(sequence)
    sequence.num_computed_tokens = len(sequence)
    scheduler.running.append(sequence)
    drafts, reservations = scheduler.reserve_spec_decode([sequence], [[5, 6]])

    decoded_tokens = scheduler.postprocess_spec_decode(
        [sequence], [[5, 6, 7]], drafts, reservations
    )

    assert decoded_tokens == 3
    assert sequence.completion_token_ids == [5, 6, 7]
    assert sequence.num_computed_tokens == 6
    assert scheduler.acceptance_rate == 1.0


def test_postprocess_spec_decode_keeps_only_prefix_before_rejection():
    scheduler = make_scheduler()
    sequence = make_running_sequence([1, 2, 3, 4])
    scheduler.block_manager.allocate(sequence, 0)
    sequence.num_cached_tokens = len(sequence)
    sequence.num_computed_tokens = len(sequence)
    scheduler.running.append(sequence)
    drafts, reservations = scheduler.reserve_spec_decode([sequence], [[5, 6]])

    scheduler.postprocess_spec_decode([sequence], [[5, 8]], drafts, reservations)

    assert sequence.completion_token_ids == [5, 8]
    assert sequence.num_computed_tokens == 5
    assert scheduler.acceptance_rate == 0.5
