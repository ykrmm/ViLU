import torch
import torch.nn as nn
from torch import Tensor

from lumen.lib import entropy


class MSELoss(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        weighting_type: str = "static",
        reduction: str = "mean",
        use_entropy: bool = False,
        entropy_coef: float = 0.1,
    ) -> torch.Tensor:
        super().__init__()
        self.weight = weight
        self.weighting_type = weighting_type
        self.reduction = reduction
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef

    def forward(
        self,
        input: Tensor,
        probs: Tensor,
        labels: Tensor,
    ) -> Tensor:
        weights = torch.ones_like(probs[:, 0])
        correct = labels == probs.argmax(dim=-1)

        if self.weighting_type == "adaptative":
            weights[~correct] *= (1 + torch.exp(4 * (correct.mean() - 0.5))) / 2
        else:
            weights[~correct] *= self.weight

        target = probs[torch.arange(probs.size(0)), labels]

        if self.use_entropy:
            target = target + self.entropy_coef * entropy(probs)

        loss = weights * (input - target) ** 2

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
