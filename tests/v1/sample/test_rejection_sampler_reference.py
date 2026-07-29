import pytest
import torch

from nanovllm.v1.sample.rejection_sampler import reference_rejection_sample


def logits(probabilities):
    return torch.log(torch.tensor(probabilities, dtype=torch.float32))


def test_reference_all_accepts_and_samples_bonus():
    target = torch.stack([logits([0.4, 0.3, 0.2, 0.1])] * 3)
    draft_probs = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]]
    )

    result = reference_rejection_sample(
        [[0, 1]],
        target,
        torch.ones(1),
        draft_probs,
        torch.tensor([0.5, 0.5]),
        torch.tensor([[4.0, 3.0, 2.0, 0.1]]),
    )

    assert result.output_token_ids == [[0, 1, 3]]
    assert result.accepted_draft_counts == [2]


def test_reference_rejects_first_token_and_samples_recovered_distribution():
    result = reference_rejection_sample(
        [[0]],
        torch.stack([logits([0.1, 0.2, 0.3, 0.4])] * 2),
        torch.ones(1),
        torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
        torch.tensor([0.9]),
        torch.ones(1, 4),
    )

    assert result.output_token_ids == [[3]]
    assert result.accepted_draft_counts == [0]


def test_reference_rejects_middle_token():
    result = reference_rejection_sample(
        [[0, 1]],
        torch.stack(
            [
                logits([0.6, 0.2, 0.1, 0.1]),
                logits([0.1, 0.1, 0.3, 0.5]),
                logits([0.25, 0.25, 0.25, 0.25]),
            ]
        ),
        torch.ones(1),
        torch.tensor(
            [[0.6, 0.2, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]
        ),
        torch.tensor([0.5, 0.9]),
        torch.ones(1, 4),
    )

    assert result.output_token_ids == [[0, 3]]
    assert result.accepted_draft_counts == [1]


def test_reference_handles_zero_drafts_and_ragged_batch():
    zero = reference_rejection_sample(
        [[], []],
        torch.stack([logits([0.1, 0.2, 0.3, 0.4]), logits([0.4, 0.3, 0.2, 0.1])]),
        torch.ones(2),
        None,
        torch.empty(0),
        torch.tensor([[1.0, 1.0, 1.0, 0.1], [0.1, 1.0, 1.0, 1.0]]),
    )
    assert zero.output_token_ids == [[3], [0]]
    assert zero.accepted_draft_counts == [0, 0]

    ragged = reference_rejection_sample(
        [[0, 1], [2]],
        torch.stack(
            [
                logits([0.8, 0.1, 0.05, 0.05]),
                logits([0.05, 0.8, 0.1, 0.05]),
                logits([0.05, 0.05, 0.8, 0.1]),
                logits([0.25] * 4),
                logits([0.25] * 4),
            ]
        ),
        torch.ones(2),
        torch.tensor(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.05, 0.8, 0.1, 0.05],
                [0.05, 0.05, 0.8, 0.1],
            ]
        ),
        torch.tensor([0.1, 0.1, 0.1]),
        torch.tensor([[1.0, 1.0, 1.0, 0.1], [0.1, 1.0, 1.0, 1.0]]),
    )
    assert ragged.output_token_ids == [[0, 1, 3], [2, 0]]
    assert ragged.accepted_draft_counts == [2, 1]


def test_reference_applies_per_request_temperature():
    raw_logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]
    )
    result = reference_rejection_sample(
        [[], []],
        raw_logits,
        torch.tensor([0.25, 4.0]),
        None,
        torch.empty(0),
        torch.tensor([[1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 2.0]]),
    )
    assert result.output_token_ids == [[3], [2]]


def test_reference_preserves_ngram_sampling_without_draft_probabilities():
    result = reference_rejection_sample(
        [[0]],
        torch.stack(
            [
                logits([0.2, 0.3, 0.4, 0.1]),
                logits([0.25, 0.25, 0.25, 0.25]),
            ]
        ),
        torch.ones(1),
        None,
        torch.tensor([0.9]),
        torch.ones(1, 4),
    )

    assert result.output_token_ids == [[2]]
    assert result.accepted_draft_counts == [0]


@pytest.mark.parametrize(
    "acceptance_uniforms",
    [torch.tensor([-0.1]), torch.tensor([1.1]), torch.tensor([float("nan")])],
)
def test_reference_rejects_invalid_acceptance_uniforms(acceptance_uniforms):
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        reference_rejection_sample(
            [[0]],
            torch.stack([logits([0.25] * 4)] * 2),
            torch.ones(1),
            torch.tensor([[0.25] * 4]),
            acceptance_uniforms,
            torch.ones(1, 4),
        )


@pytest.mark.parametrize(
    "recovery_uniforms",
    [torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.full((1, 4), float("inf"))],
)
def test_reference_requires_positive_finite_exponential_noise(recovery_uniforms):
    with pytest.raises(ValueError, match="finite and positive"):
        reference_rejection_sample(
            [[0]],
            torch.stack([logits([0.25] * 4)] * 2),
            torch.ones(1),
            torch.tensor([[0.25] * 4]),
            torch.tensor([0.5]),
            recovery_uniforms,
        )


def test_reference_output_distribution_matches_target_distribution():
    generator = torch.Generator().manual_seed(1234)
    target = torch.tensor([0.1, 0.2, 0.3, 0.4])
    draft = torch.tensor([0.4, 0.3, 0.2, 0.1])
    target_logits = torch.log(target).repeat(2, 1)
    counts = torch.zeros(4, dtype=torch.int64)

    for _ in range(50_000):
        draft_token = int(torch.multinomial(draft, 1, generator=generator))
        acceptance = torch.rand(1, generator=generator)
        recovery_noise = torch.empty(1, 4).exponential_(generator=generator)
        result = reference_rejection_sample(
            [[draft_token]],
            target_logits,
            torch.ones(1),
            draft.unsqueeze(0),
            acceptance,
            recovery_noise,
        )
        counts[result.output_token_ids[0][0]] += 1

    frequencies = counts.to(torch.float32) / counts.sum()
    torch.testing.assert_close(frequencies, target, atol=0.015, rtol=0)
