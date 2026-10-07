from vilu.models.abstract_vilu import AbstractViLU
from vilu.models.vilu_attn import ViLUAttention

# Former names, kept so existing code and notebooks keep working.
Abstractvilu = AbstractViLU
viluAttention = ViLUAttention

__all__ = [
    "AbstractViLU",
    "ViLUAttention",
]
