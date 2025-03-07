from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vilu.models import Abstractvilu

Kwargs = Dict[str, Any]


class viluAttention(Abstractvilu):
    def __init__(
        self,
        concat: bool = False,
        identity_init: bool = True,
        n_iter_freeze_proj: int = 200,
        keep_frozen: bool = False,
        use_predicted_caption: bool = False,
        use_attention: bool = True,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.concat = concat
        self.identity_init = identity_init
        self.n_iter_freeze_proj = n_iter_freeze_proj
        self.keep_frozen = keep_frozen
        self.use_predicted_caption = use_predicted_caption
        self.use_attention = use_attention
        self.register_buffer("iteration_count", torch.tensor(0))

        if not self.concat:
            assert self.visual_dim == self.textual_dim

        if (
            not self.use_attention or not self.concat
        ) and not self.use_predicted_caption:
            input_dim = self.visual_dim
        elif self.use_attention and self.use_predicted_caption and self.concat:
            input_dim = self.visual_dim + 2 * self.textual_dim
        else:
            input_dim = self.visual_dim + self.textual_dim

        self.mlp = self.build_mlp(input_dim)

        if self.use_attention:
            self.in_proj_v = nn.Linear(self.visual_dim, self.visual_dim, bias=False)
            self.in_proj_t = nn.Linear(self.textual_dim, self.textual_dim, bias=False)
            if self.identity_init:
                nn.init.eye_(self.in_proj_v.weight)
                nn.init.eye_(self.in_proj_t.weight)

    def cross_attention(
        self,
        v: Tensor,
        t: Tensor = None,
    ) -> Tensor:
        if self.training:
            self.iteration_count += 1

        if self.iteration_count > self.n_iter_freeze_proj and not self.keep_frozen:
            v = self.in_proj_v(v)
            t = self.in_proj_t(t)
        else:
            v = self.in_proj_v(v) * 0 + v
            t = self.in_proj_t(t) * 0 + t

        return F.softmax(self.logit_scale * v @ t.t(), dim=-1) @ t

    def forward(
        self,
        v: Tensor,
        t: Tensor = None,
    ) -> Tensor:
        if self.use_attention:
            if self.concat:
                x = torch.cat((v, self.cross_attention(v, t)), dim=1)
            else:
                x = v + self.cross_attention(v, t)  # standard
        else:
            x = v
        if self.use_predicted_caption:
            probs = F.softmax(self.logit_scale * v @ t.t(), dim=-1)
            _, preds = probs.topk(1, 1, True, True)
            t_preds = t[preds[:, 0]]
            x = torch.cat((x, t_preds), dim=1)

        x = self.mlp(x)
        return x
