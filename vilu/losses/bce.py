import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        weighting_type: str = "static",
        reduction: str = "mean",
        batchwise_train: bool = False,
    ) -> Tensor:
        super().__init__()
        self.weight = weight
        self.weighting_type = weighting_type
        self.reduction = reduction
        self.batchwise_train = batchwise_train

    def forward(
        self,
        input: Tensor,
        probs: Tensor,
        labels: Tensor,
    ) -> Tensor:
        weights = torch.ones_like(probs[:, 0])

        if self.batchwise_train:
            correct = probs.diag() == probs.max(dim=-1).values
        else:
            correct = labels == probs.argmax(dim=-1)

        if self.weighting_type == "adaptative":
            acc = correct.float().mean()
            weights[~correct] *= torch.log(1 + (acc / (1 - acc)))
        else:
            weights[~correct] *= self.weight

        target = correct.float()
        loss = F.binary_cross_entropy_with_logits(input, target, weights, reduction=self.reduction)
        return loss
