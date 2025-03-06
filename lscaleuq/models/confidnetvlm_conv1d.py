from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lscaleuq.models import AbstractConfidNetVLM

Kwargs = Dict[str, Any]


class L2NormPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


class AdaptativeAvgMaxPooling(nn.Module):
    def __init__(
        self,
        output_size: int,
    ) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        avg_pool = F.adaptive_avg_pool1d(x, self.output_size)
        max_pool = F.adaptive_max_pool1d(x, self.output_size)
        return torch.cat((avg_pool, max_pool), dim=1)


class ConfidNetVLMConv1d(AbstractConfidNetVLM):
    """
    Use conv1D on probabilities to be independant of the number of probabilities
    """

    def __init__(
        self,
        kernel_size: int = 3,
        use_activation: bool = True,
        pooling_type: str = "avg",  # Pooling type: "avg", "max", "l2", "avg+max",
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.use_activation = use_activation
        self.pooling_type = pooling_type

        self.pooling = self.get_pooling()
        self.mlp = self.build_mlp(2 * self.visual_dim)
        self.conv = nn.Conv1d(1, self.visual_dim, kernel_size=self.kernel_size, padding=1)

    def forward(self, v: Tensor, t: Tensor) -> Tensor:
        prob = F.softmax(self.logit_scale * v @ t.t(), dim=-1)
        prob = prob.unsqueeze(1)
        if self.use_activation:
            prob = self.activation_fn(self.conv(prob))
        else:
            prob = self.conv(prob)

        prob = self.pooling(prob).squeeze()

        x = torch.cat((v, prob), dim=1)
        x = self.mlp(x)
        return x

    def get_pooling(self) -> nn.Module:
        if self.pooling_type == "avg":
            pooling = nn.AdaptiveAvgPool1d(1)
        elif self.pooling_type == "max":
            pooling = nn.AdaptiveMaxPool1d(1)
        elif self.pooling_type == "l2":
            pooling = L2NormPooling()
        elif self.pooling_type == "avg+max":
            pooling = AdaptativeAvgMaxPooling(1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        return pooling
