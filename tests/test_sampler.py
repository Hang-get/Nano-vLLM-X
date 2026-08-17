import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def test_zero_temperature_samples_the_logit_argmax():
    logits = torch.tensor([[1.0, 4.0, 2.0], [3.0, 1.0, 2.0]])
    temperatures = torch.tensor([0.0, 0.0])

    token_ids = Sampler()(logits, temperatures)

    assert token_ids.tolist() == [1, 0]


def test_sampling_params_accepts_zero_temperature():
    params = SamplingParams(temperature=0.0)

    assert params.temperature == 0.0
