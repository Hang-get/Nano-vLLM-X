import pytest
import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.v1.sample.rejection_sampler import (
    RejectionSampler,
    reference_rejection_sample,
)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_sampler_matches_reference_with_injected_randomness():
    device = torch.device("cuda")
    draft_token_ids = [[0, 1], [2]]
    logits = torch.log(
        torch.tensor(
            [
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.1, 0.3, 0.5],
                [0.2, 0.2, 0.5, 0.1],
                [0.25, 0.25, 0.25, 0.25],
                [0.4, 0.3, 0.2, 0.1],
            ],
            device=device,
        )
    )
    draft_probs = torch.tensor(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.2, 0.2, 0.5, 0.1],
        ],
        device=device,
    )
    temperatures = torch.ones(2, device=device)
    acceptance = torch.tensor([0.5, 0.9, 0.5], device=device)
    recovery_noise = torch.tensor(
        [[1.0, 1.0, 1.0, 0.1], [0.1, 1.0, 1.0, 1.0]],
        device=device,
    )

    reference = reference_rejection_sample(
        draft_token_ids,
        logits,
        temperatures,
        draft_probs,
        acceptance,
        recovery_noise,
    )
    actual = RejectionSampler(Sampler())(
        draft_token_ids,
        logits,
        temperatures,
        draft_probs,
        acceptance_uniforms=acceptance,
        recovery_noise=recovery_noise,
    )

    assert actual.output_token_ids == reference.output_token_ids
    assert actual.accepted_draft_counts == reference.accepted_draft_counts


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_sampler_validates_injected_acceptance_uniforms():
    device = torch.device("cuda")
    sampler = RejectionSampler(Sampler())
    logits = torch.log(torch.full((2, 4), 0.25, device=device))
    draft_probs = torch.full((1, 4), 0.25, device=device)

    with pytest.raises(ValueError, match="must have shape"):
        sampler(
            [[0]],
            logits,
            torch.ones(1, device=device),
            draft_probs,
            acceptance_uniforms=torch.tensor([0.5, 0.5], device=device),
        )

    with pytest.raises(ValueError, match="on the logits device"):
        sampler(
            [[0]],
            logits,
            torch.ones(1, device=device),
            draft_probs,
            acceptance_uniforms=torch.tensor([0.5]),
        )

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        sampler(
            [[0]],
            logits,
            torch.ones(1, device=device),
            draft_probs,
            acceptance_uniforms=torch.tensor([1.1], device=device),
        )
