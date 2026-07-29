from dataclasses import dataclass

import torch


@dataclass
class TargetModelOutput:
    hidden_states: torch.Tensor
    auxiliary_hidden_states: torch.Tensor
