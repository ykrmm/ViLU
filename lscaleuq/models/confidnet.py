from typing import Any, Dict, Optional
from torch import Tensor

from lscaleuq.models import AbstractConfidNetVLM

Kwargs = Dict[str, Any]


class ConfidNet(AbstractConfidNetVLM):
    def __init__(
        self,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.mlp = self.build_mlp(self.visual_dim)

    def forward(
        self,
        v: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mlp(v)
