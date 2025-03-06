from typing import Any, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lscaleuq.models import AbstractConfidNetVLM

Kwargs = Dict[str, Any]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class ConfidNetVLMTrsf(AbstractConfidNetVLM):
    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 8,
        use_predicted_caption: bool = False,
        use_visual: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_predicted_caption = use_predicted_caption
        self.use_visual = use_visual
        input_dim = self.visual_dim
        if self.use_predicted_caption:
            input_dim += self.visual_dim
        if self.use_visual:
            input_dim += self.visual_dim

        self.transformer = Transformer(self.visual_dim, self.n_layers, self.n_heads)
        self.mlp = self.build_mlp(input_dim)
        scale = self.visual_dim**-0.5
        self.uncertainty_embedding = nn.Parameter(scale * torch.randn(self.visual_dim))
        self.ln_pre = LayerNorm(self.visual_dim)
        self.ln_post = LayerNorm(self.visual_dim)

    def forward(
        self,
        v: Tensor,
        t: Tensor = None,
    ) -> Tensor:
        if self.use_predicted_caption:
            prob = F.softmax(self.logit_scale * v @ t.t(), dim=-1)
            _, pred = prob.topk(1, 1, True, True)

        t = t + torch.zeros(v.shape[0], t.shape[0], v.shape[-1], dtype=v.dtype, device=v.device)
        v = v.unsqueeze(1)

        x = torch.cat((v, t), dim=1)
        x = torch.cat(
            [
                self.uncertainty_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )

        # Must be B,N+2,d
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)

        out = [x[:, 0, :]]  # uncertainty token
        if self.use_predicted_caption:
            t = x[:, 2:, :]
            out.append(t[torch.range(0, v.shape[0] - 1).int(), pred[:, 0]])

        if self.use_visual:
            out.append(x[:, 1, :])

        x = torch.cat(out, dim=-1)
        x = self.mlp(x)
        return x
