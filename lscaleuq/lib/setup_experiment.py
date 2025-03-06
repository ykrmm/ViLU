import os
from typing import Tuple

import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from wandb.wandb_run import Run as WandbRun
from lscaleuq.lib.logger import logger
from lscaleuq.lib.lightning_fabric import LightningFabric


def setup_experiment(config: DictConfig, hydra_config: dict) -> Tuple[LightningFabric, WandbRun, str]:
    torch.set_printoptions(sci_mode=False)  # Disable scientific notation
    # torch.set_float32_matmul_precision("high")

    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    model = choices["model"]
    backbone = choices["backbone"]
    loss = choices["loss"]
    dataset = choices["dataset"]

    exp_name = f"{config.exp_name}_{model}_{dataset}_{backbone}_{loss}"
    save_dir = os.path.join(config.save_dir, exp_name)
    config_path = os.path.join(save_dir, "config.yaml")
    ckpt_path = os.path.join(save_dir, "last.ckpt")

    fabric: LightningFabric = instantiate(
        config.fabric,
        devices=int(os.getenv("SLURM_GPUS_ON_NODE", torch.cuda.device_count())),
        num_nodes=int(os.getenv("SLURM_NNODES", 1)),
    )
    fabric.seed_everything(config.seed)
    fabric.launch()

    wandb_run = None
    if fabric.is_global_zero:
        if os.path.exists(config_path) and config.resume:
            logger.warning(f"``resume'' is set to ``True''. Loading previous chekpoint: {ckpt_path} and configuration file: {config_path}.")
            prev_cfg = OmegaConf.load(config_path)
            config.wandb_run_id = prev_cfg.wandb_run_id
            assert (
                config.model == prev_cfg.model and config.optimizer == prev_cfg.optimizer and config.loss == prev_cfg.loss
            ), "Trying to resume training with different hyperparameters (config != prev_cfg). Run a new experiment by changing the ``exp_name'' argument."

            wandb_run = wandb.init(
                entity="nueve",
                project="Large_scale_UQ",
                mode=config.wandb_mode if not config.debug else "disabled",
                name=exp_name,
                config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
                reinit=True,
                id=prev_cfg.wandb_run_id,
                resume="allow",
            )

        else:
            if os.path.exists(ckpt_path):
                logger.warning(f"``resume'' is set to ``False'' but found a previous chekpoint: {ckpt_path} that will be overriden.")

            wandb_run = wandb.init(
                entity="nueve",
                project="Large_scale_UQ",
                mode=config.wandb_mode if not config.debug else "disabled",
                name=exp_name,
                config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
                reinit=True,
                id=wandb.util.generate_id(),
            )

            config["wandb_run_id"] = wandb_run.id
            # save config
            os.makedirs(save_dir, exist_ok=True)
            OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

    fabric.barrier()
    if config.debug:
        os.environ["DEBUG"] = "True"
        fabric.info("Launching in debug mode.")

    fabric.info("Model: {}".format(model))
    fabric.info("Backbone Model: {}".format(backbone))
    fabric.info("Loss function: {}".format(loss))
    fabric.info("Dataset: {}".format(dataset))

    return fabric, wandb_run, save_dir
