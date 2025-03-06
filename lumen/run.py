import os
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ChainedScheduler
from torch.utils.data import Dataset, DataLoader

from lumen.engine import Trainer
from lumen.lib.setup_experiment import setup_experiment
from lumen.models.clip import load as load_clip
from lumen.models import AbstractLuMen


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def main(config: DictConfig) -> None:
    hydra_config = HydraConfig.get()

    n_nodes = int(os.getenv("SLURM_NNODES", 1))
    n_gpus_per_node = int(os.getenv("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))
    config.optimizer.lr = config.optimizer.lr * n_nodes * n_gpus_per_node

    fabric, save_dir = setup_experiment(config, hydra_config)

    # Datasets
    with fabric.rank_zero_first():
        dataset_train: Dataset = instantiate(config.dataset.train)
        dataset_val: Dataset = instantiate(config.dataset.val)


    # Dataloaders
    if config.dataset.web_dataset:
        train_loader: DataLoader = instantiate(config.loader_wds, dataset_train)
        val_loader: DataLoader = instantiate(config.loader_wds, dataset_val)
    else:
        train_loader: DataLoader = instantiate(config.train_loader, dataset_train)
        val_loader: DataLoader = instantiate(config.val_loader, dataset_val)



    # Load CLIP
    with fabric.rank_zero_first():
        clip_model, _ = load_clip(config.backbone.name)

    # Model & Optimizer
    model: AbstractLuMen = instantiate(config.model, logit_scale=clip_model.logit_scale.exp().item())
    optimizer: Optimizer = instantiate(config.optimizer, params=model.parameters())

    # Distrubuted Training with Fabric
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)


    clip_model.requires_grad_(False)
    clip_model.visual.requires_grad_(False)
    clip_model.transformer.requires_grad_(False)


    model, optimizer = fabric.setup(model, optimizer)
    lr_scheduler: _LRScheduler = instantiate(config.lr_scheduler, optimizer)

    if config.lr_warmup.total_iters > 0:
        linear_warmup: _LRScheduler = instantiate(config.lr_warmup, optimizer)
        lr_scheduler = ChainedScheduler([linear_warmup, lr_scheduler])

    loss_fn = instantiate(config.loss)

    trainer: Trainer = instantiate(
        config.engine,
        fabric=fabric,
        clip_model=clip_model,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        save_dir=save_dir,
    )
    trainer.fit()



if __name__ == "__main__":
    main()
