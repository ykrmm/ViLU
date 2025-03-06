import torch
from lightning.fabric import Fabric


def gather_features(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    fabric: Fabric,
    gather_with_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fabric.world_size <= 1:
        return img_features, text_features

    if gather_with_grad:
        gathered_img_features = fabric.all_gather(img_features)
        gathered_text_features = fabric.all_gather(text_features)
    else:
        gathered_img_features = fabric.all_gather(img_features.detach())
        gathered_text_features = fabric.all_gather(text_features.detach())

    ws = fabric.world_size
    bs_local = gathered_img_features.shape[1]

    gathered_img_features = gathered_img_features.reshape(ws * bs_local, *gathered_img_features.shape[2:])
    gathered_text_features = gathered_text_features.reshape(ws * bs_local, *gathered_text_features.shape[2:])

    return gathered_img_features, gathered_text_features
