from typing import Any, Dict
import torch
import torch.nn as nn
from torch import Tensor

from lscaleuq.models import AbstractConfidNetVLM
import lscaleuq.lib as lib

Kwargs = Dict[str, Any]


class ConfidNetVLMEntropy(AbstractConfidNetVLM):
    def __init__(
        self,
        entropy_dim: int = 128,
        eps: float = 1e-7,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.entropy_dim = entropy_dim
        self.eps = eps

        self.fc_entropy = nn.Linear(1, entropy_dim)
        self.mlp = self.build_mlp(self.visual_dim + self.entropy_dim)

    def forward(
        self,
        v: Tensor,
        t: Tensor,
    ) -> Tensor:
        logits = self.logit_scale * v @ t.t()

        entropy = lib.softmax_entropy(logits, self.eps).unsqueeze(1)
        entropy = self.fc_entropy(entropy)
        x = torch.cat((v, entropy), dim=1)
        x = self.mlp(x)

        return x
