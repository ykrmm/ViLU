from typing import Union
import os
import omegaconf
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from lumen.lib.meters import DictAverage


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    scaler: Union[GradScaler, None],
    train_meter: DictAverage,
    config: DictConfig,
) -> None:
    conf = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    state = {
        "epoch": epoch,
        "train_meter": {k: v.avg for k, v in train_meter.items()},
        "state_dict": model.state_dict()
        if not hasattr(model, "trainable_state_dict")
        else model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "args": conf,
    }

    save_dir = os.path.join(save_dir, config.wandbconf.name)
    os.makedirs(save_dir, exist_ok=True)

    # running checkpoint
    model_name = (
        f"model_seed{config.engine.seed}.ckpt"
        if config.engine.seed is not None
        else "model.ckpt"
    )
    torch.save(state, os.path.join(save_dir, model_name))

    # epoch checkpoint
    if ((epoch % config.engine.save_freq == 0) and (epoch > 0)) or (
        epoch + 1 == config.engine.n_epochs
    ):
        model_name = (
            f"model_seed{config.engine.seed}_epoch{epoch}.ckpt"
            if config.engine.seed is not None
            else f"model_epoch{epoch}.ckpt"
        )
        torch.save(state, os.path.join(save_dir, model_name))
