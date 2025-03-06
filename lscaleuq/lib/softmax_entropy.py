from torch import Tensor


def entropy(probs: Tensor, eps: float = 1e-6) -> Tensor:
    """Entropy of softmax distribution from probabilities."""
    return -(probs * (probs + eps).log()).sum(dim=-1)


def softmax_entropy(logits: Tensor, eps: float = 1e-6) -> Tensor:
    """Entropy of softmax distribution from logits."""
    return -((logits + eps).softmax(-1) * (logits + eps).log_softmax(-1)).sum(-1)
