from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.models.model_output import TargetModelOutput
from nanovllm.v1.spec_decode.types import (
    DraftProposal,
    SpecDecodeResult,
    SpecReservation,
)


class RecordingTargetModel:
    def __init__(self, events):
        self.events = events
        self.lm_head = self

    def __call__(
        self,
        input_ids,
        positions=None,
        auxiliary_layer_ids=(),
        return_all_logits=False,
    ):
        if return_all_logits:
            return torch.zeros((input_ids.size(0), 7))
        self.events.append("target_with_aux")
        rows = input_ids.numel()
        hidden = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
        auxiliary = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6)
        return TargetModelOutput(hidden, auxiliary)

    def compute_logits(self, hidden_states):
        return torch.zeros((hidden_states.size(0), 7))

class RecordingProposer:
    def __init__(self, events, proposal=None):
        self.events = events
        self.proposal = proposal
        self.commits = []

    def prefill(self, seqs, auxiliary, target_lengths):
        self.events.append("draft_prefill_and_store_anchor")
        assert [rows.size(0) for rows in auxiliary] == [len(seq) for seq in seqs]
        assert target_lengths == [len(seq) for seq in seqs]

    def propose(self, seqs, reservations, temperatures):
        self.events.append("draft_propose")
        return self.proposal

    def commit(self, seqs, auxiliary, accepted_counts, target_lengths):
        self.events.append("select_next_anchor")
        selected = [
            rows[accepted]
            for rows, accepted in zip(auxiliary, accepted_counts)
        ]
        self.commits.append(
            (auxiliary, accepted_counts, target_lengths, selected)
        )

    def release(self, seq_ids):
        self.events.append(("release", list(seq_ids)))


class RecordingRejectionSampler:
    def __init__(self, events, result):
        self.events = events
        self.result = result

    def __call__(self, **kwargs):
        self.events.append("rejection_sample")
        assert kwargs["draft_probs"].shape[0] == sum(
            len(row) for row in kwargs["draft_token_ids"]
        )
        return self.result


class FakeSequence:
    def __init__(self, seq_id, token_ids, num_computed_tokens):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_computed_tokens = num_computed_tokens
        self.last_token = token_ids[-1]
        self.block_table = [0]
        self.temperature = 1.0
        self.is_finished = False

    def __len__(self):
        return len(self.token_ids)

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]


def test_runner_eagle3_prefill_orders_target_draft_and_root_sampling(monkeypatch):
    events = []
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.model = RecordingTargetModel(events)
    runner.eagle3_proposer = RecordingProposer(events)
    runner.speculative_config = SimpleNamespace(auxiliary_layer_ids=(2, 18, 33))
    runner.prepare_prefill = lambda seqs: (torch.tensor([1, 2, 3]), torch.arange(3))
    runner.prepare_sample = lambda seqs: torch.ones(len(seqs))

    class RootSampler:
        def __call__(self, logits, temperatures):
            events.append("sample_root")
            return torch.tensor([6])

    runner.sampler = RootSampler()
    monkeypatch.setattr("nanovllm.engine.model_runner.reset_context", lambda: None)
    seq = FakeSequence(1, [1, 2, 3], 0)

    token_ids = runner.run_eagle3_prefill([seq])

    assert token_ids == [6]
    assert events == [
        "target_with_aux",
        "draft_prefill_and_store_anchor",
        "sample_root",
    ]


@pytest.mark.parametrize("accepted_count", [0, 1, 2])
def test_runner_eagle3_verify_commits_anchor_by_explicit_accepted_count(
    monkeypatch,
    accepted_count,
):
    events = []
    proposal = DraftProposal(
        token_ids=[[4, 5]],
        probabilities=torch.full((2, 7), 1 / 7),
        lengths=[2],
    )
    output = proposal.token_ids[0][:accepted_count] + [9]
    result = SpecDecodeResult([output], [accepted_count])
    proposer = RecordingProposer(events, proposal)
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.model = RecordingTargetModel(events)
    runner.eagle3_proposer = proposer
    runner.rejection_sampler = RecordingRejectionSampler(events, result)
    runner.speculative_config = SimpleNamespace(auxiliary_layer_ids=(2, 18, 33))
    runner.prepare_sample = lambda seqs: torch.ones(len(seqs))
    runner.prepare_spec_decode = lambda seqs, tokens, reservations: (
        torch.tensor([3, 4, 5]),
        torch.arange(3),
        torch.tensor([0, 1, 2]),
        [3],
    )
    monkeypatch.setattr("nanovllm.engine.model_runner.reset_context", lambda: None)
    seq = FakeSequence(1, [1, 2, 3], 2)

    actual_proposal, actual_result = runner.run_eagle3_spec_decode(
        [seq], [SpecReservation(2, [])]
    )

    assert actual_proposal is proposal
    assert actual_result is result
    assert events == [
        "draft_propose",
        "target_with_aux",
        "rejection_sample",
        "select_next_anchor",
    ]
    auxiliary, accepted, target_lengths, selected = proposer.commits[0]
    assert accepted == [accepted_count]
    assert target_lengths == [3 + accepted_count]
    torch.testing.assert_close(
        selected[0],
        auxiliary[0][accepted_count],
    )


class RecordingEngineRunner:
    def __init__(self, events, proposal, result):
        self.events = events
        self.proposal = proposal
        self.result = result

    def call(self, method, *args):
        if method == "release_eagle_states":
            self.events.append(("release", list(args[0])))
            return None
        if method == "run_eagle3_spec_decode":
            self.events.extend(
                [
                    "draft_propose",
                    "target_verify_with_aux",
                    "rejection_sample",
                    "select_next_anchor",
                ]
            )
            return self.proposal, self.result
        raise AssertionError(method)


class RecordingEngineScheduler:
    speculative_method = "eagle3"

    def __init__(self, events, seq, reservation):
        self.events = events
        self.seq = seq
        self.reservation = reservation

    def schedule(self):
        return [self.seq], False

    def pop_preempted_seq_ids(self):
        return [41]

    def get_eagle3_requested_lengths(self, seqs):
        self.events.append("requested_lengths")
        return [self.reservation.draft_len]

    def reserve_spec_budget(self, seqs, requested):
        self.events.append("reserve_budget")
        return [self.reservation]

    def postprocess_spec_decode(self, seqs, result, proposal, reservations):
        self.events.append("scheduler_commit")
        self.seq.is_finished = True
        return len(result.output_token_ids[0])


def test_engine_orders_reservation_and_releases_preempted_and_finished_states():
    events = []
    seq = FakeSequence(7, [1, 2, 3], 2)
    proposal = DraftProposal([[4]], torch.full((1, 7), 1 / 7), [1])
    result = SpecDecodeResult([[9]], [0])
    reservation = SpecReservation(1, [])
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = RecordingEngineScheduler(events, seq, reservation)
    engine.model_runner = RecordingEngineRunner(events, proposal, result)

    outputs, num_tokens = engine.step()

    assert outputs == [(7, [])]
    assert num_tokens == -1
    assert events == [
        ("release", [41]),
        "requested_lengths",
        "reserve_budget",
        "draft_propose",
        "target_verify_with_aux",
        "rejection_sample",
        "select_next_anchor",
        "scheduler_commit",
        ("release", [7]),
    ]


def test_engine_exposes_combined_spec_decode_metrics():
    engine = LLMEngine.__new__(LLMEngine)

    class TimingRunner:
        def call(self, method, *args):
            if method == "get_spec_decode_timings":
                return (1.5, 2.5, 3.5)
            raise AssertionError(method)

    class MetricsScheduler:
        def get_spec_decode_metrics(self, *timings):
            return timings

    engine.model_runner = TimingRunner()
    engine.scheduler = MetricsScheduler()

    assert engine.spec_decode_metrics == (1.5, 2.5, 3.5)
