from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch import Tensor

from lscaleuq.models import AbstractConfidNetVLM

Kwargs = Dict[str, Any]


class ConfidNetVLMProb(AbstractConfidNetVLM):
    def __init__(
        self,
        num_classes: int = 1000,
        only_probs: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.only_probs = only_probs

        if self.only_probs:
            self.mlp = self.build_mlp(self.num_classes)
        else:
            self.mlp = self.build_mlp(self.visual_dim + self.num_classes)

    def forward(
        self,
        v: Tensor,
        t: Tensor,
    ) -> Tensor:
        probs = F.softmax(self.logit_scale * v @ t.t(), dim=-1)

        if self.only_probs:
            x = probs
        else:
            x = torch.cat((v, probs), dim=1)

        return self.mlp(x)


class ConfidNetVLMProbTopk(AbstractConfidNetVLM):
    def __init__(
        self,
        topk: int,
        only_probs: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.topk = topk
        self.only_probs = only_probs

        if self.only_probs:
            self.mlp = self.build_mlp(self.num_classes)
        else:
            self.mlp = self.build_mlp(self.visual_dim + self.num_classes)

    def forward(
        self,
        v: Tensor,
        t: Tensor,
    ) -> Tensor:
        probs = F.softmax(self.logit_scale * v @ t.t(), dim=-1)
        probs = torch.sort(probs, dim=1, descending=True)[0]
        probs = probs[:, :self.topk]

        if self.only_probs:
            x = probs
        else:
            x = torch.cat((v, probs), dim=1)

        return self.mlp(x)
