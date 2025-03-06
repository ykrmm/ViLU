import os
from typing import Dict
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ChainedScheduler
from torch.utils.data import Dataset, DataLoader

from lscaleuq.engine import Trainer
from lscaleuq.lib.setup_experiment import setup_experiment
from lscaleuq.models.clip import load as load_clip
from lscaleuq.models import AbstractConfidNetVLM


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def main(config: DictConfig) -> None:
    hydra_config = HydraConfig.get()

    n_nodes = int(os.getenv("SLURM_NNODES", 1))
    n_gpus_per_node = int(os.getenv("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))
    config.optimizer.lr = config.optimizer.lr * n_nodes * n_gpus_per_node

    fabric, wandb_run, save_dir = setup_experiment(config, hydra_config)

    # Datasets
    with fabric.rank_zero_first():
        dataset_train: Dataset = instantiate(config.dataset.train)
        dataset_val: Dataset = instantiate(config.dataset.val)
        if config.engine.eval_zs:
            datasets_zs: Dict[Dataset] = instantiate(config.zs_datasets.dst)

    # Dataloaders
    if config.dataset.web_dataset:
        train_loader: DataLoader = instantiate(config.loader_wds, dataset_train)
        val_loader: DataLoader = instantiate(config.loader_wds, dataset_val)
    else:
        train_loader: DataLoader = instantiate(config.train_loader, dataset_train)
        val_loader: DataLoader = instantiate(config.val_loader, dataset_val)

    if config.engine.eval_zs:
        zs_loaders: Dict[DataLoader] = {}
        for dts_name, dts in datasets_zs.items():
            zs_loaders[dts_name] = instantiate(config.val_loader, dts)

    # Load CLIP
    with fabric.rank_zero_first():
        clip_model, _ = load_clip(config.backbone.name)

    # Model & Optimizer
    model: AbstractConfidNetVLM = instantiate(config.model, logit_scale=clip_model.logit_scale.exp().item())
    optimizer: Optimizer = instantiate(config.optimizer, params=model.parameters())

    # Distrubuted Training with Fabric
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Maybe not necesary:
    clip_model.requires_grad_(False)
    clip_model.visual.requires_grad_(False)
    clip_model.transformer.requires_grad_(False)
    ###

    model, optimizer = fabric.setup(model, optimizer)
    lr_scheduler: _LRScheduler = instantiate(config.lr_scheduler, optimizer)

    if config.lr_warmup.total_iters > 0:
        linear_warmup: _LRScheduler = instantiate(config.lr_warmup, optimizer)
        lr_scheduler = ChainedScheduler([linear_warmup, lr_scheduler])

    loss_fn = instantiate(config.loss)

    trainer: Trainer = instantiate(
        config.engine,
        fabric=fabric,
        wandb_run=wandb_run,
        clip_model=clip_model,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        zs_loaders=zs_loaders if config.engine.eval_zs else None,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        save_dir=save_dir,
    )
    trainer.fit()

    if fabric.is_global_zero:
        wandb_run.finish()


if __name__ == "__main__":
    main()
