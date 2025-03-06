from typing import Optional, List
import torch.nn as nn
from torch import Tensor


class AbstractLuMen(nn.Module):
    def __init__(
        self,
        visual_dim: int = 512,
        textual_dim: int = 512,
        layers: List[int] = [512, 256, 128, 1],
        activation: str = "relu",
        negative_slope: float = 0.01,
        use_sigmoid: bool = False,
        logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.textual_dim = textual_dim
        self.layers = layers
        self.activation = activation
        self.negative_slope = negative_slope
        self.use_sigmoid = use_sigmoid
        self.logit_scale = logit_scale

        self.activation_fn = self.get_activation()

    def forward(
        self,
        v: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    def build_mlp(self, input_dim: int) -> nn.Module:
        modules = [nn.Linear(input_dim, self.layers[0]), self.activation_fn]
        for i in range(len(self.layers) - 1):
            modules.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                modules.append(self.activation_fn)
        if self.use_sigmoid:
            modules.append(nn.Sigmoid())
        mlp = nn.Sequential(*modules)

        return mlp

    def get_activation(self) -> nn.Module:
        if self.activation == "relu":
            activation_fn = nn.ReLU()
        elif self.activation == "leaky_relu":
            activation_fn = nn.LeakyReLU(self.negative_slope)
        else:
            raise ValueError("Activation function not supported")

        return activation_fn
