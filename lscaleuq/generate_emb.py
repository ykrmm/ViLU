import os
import warnings
import logging
import hydra
from hydra.utils import instantiate
from wandb.integration.lightning.fabric import WandbLogger
from omegaconf import OmegaConf, DictConfig

# flake8: noqa
import numpy as np
import torch
from torch.utils.data import DataLoader

from lscaleuq.getter import Getter
from lscaleuq.models.clip import load as load_clip
from lscaleuq.engine import Trainer, GenerateEmbeddings

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("join", lambda *pth: os.path.join(*pth))
OmegaConf.register_new_resolver("mult", lambda *numbers: float(np.prod([float(x) for x in numbers])))
OmegaConf.register_new_resolver("sum", lambda *numbers: sum(map(float, numbers)))
OmegaConf.register_new_resolver("sub", lambda x, y: float(x) - float(y))
OmegaConf.register_new_resolver("div", lambda x, y: float(x) / float(y))
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
OmegaConf.register_new_resolver("eq", lambda a, b: a == b)


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def run(config: DictConfig) -> None:
    getter = Getter(config)
    torch.set_printoptions(sci_mode=False)  # Disable scientific notation

    if config.engine.debug:
        os.environ["DEBUG"] = "True"
        logging.info("Launching in debug mode.")

    wandb_logger = WandbLogger(
        entity=config.wandbconf.entity,
        project=config.wandbconf.project,
        name=config.wandbconf.name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        reinit=True,  # reinit=True is necessary when calling multirun with hydra
    )

    fabric = instantiate(
        config.fabric,
        logger=wandb_logger,
        devices=int(os.getenv("SLURM_GPUS_ON_NODE", torch.cuda.device_count())),
        num_nodes=int(os.getenv("SLURM_NNODES", 1)),
    )

    fabric.launch()

    # Datasets
    dataset_train = instantiate(config.datasets.train)
    dataset_val = instantiate(config.datasets.val)
    if config.engine.eval_zs:
        datasets_zs = instantiate(config.zs_datasets.dst)

    # Dataloaders
    train_loader = instantiate(config.train_loader, dataset_train)
    val_loader = instantiate(config.val_loader, dataset_val)

    if config.engine.eval_zs:
        zs_loaders = {}
        for dts_name, dts in datasets_zs.items():
            zs_loaders[dts_name] = instantiate(config.val_loader, dts) 

    if config.engine.name == "train_confidnet":
        # Load CLIP
        clip_model, _ = load_clip(config.backbone_model.clip_name)

        if config.backbone_model.logit_scale is None:
            logit_scale = clip_model.logit_scale.exp().item()
        else:
            logit_scale = config.backbone_model.logit_scale
        model = instantiate(config.models.model, logit_scale=logit_scale)
        optimizer = instantiate(config.optim.optimizer, params=model.parameters())
        loss_fn = instantiate(config.losses.loss)

        engine = Trainer(
            config=config,
            clip_model=clip_model,
            model=model,
            train_dts=dataset_train,
            val_dts=dataset_val,
            zs_dts=datasets_zs,
            optimizer=optimizer,
            criterion=loss_fn,
        )
    elif config.engine.name == "generate_emb":
        engine = GenerateEmbeddings(
            config=config,
            train_dts=dataset_train,
            val_dts=dataset_val,
            device=device,
        )
    else:
        raise NotImplementedError("Engine not implemented yet")

    engine.run()


if __name__ == "__main__":
    run()
