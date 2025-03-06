from typing import Dict

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

from lscaleuq.models.clip import load as load_clip


class Getter:
    """
    This class allows to create differents object (model,loss,dataset) based on the config file.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def get_wandb_logger(self) -> None:
        pass

    def get_model(self) -> nn.Module:
        """
        Create the model based on the config file.
        """
        # Load CLIP
        clip_model, _ = load_clip(self.config.backbone_model.clip_name)

        if self.config.backbone_model.logit_scale is None:
            logit_scale = clip_model.logit_scale.exp().item()
        else:
            logit_scale = self.config.backbone_model.logit_scale
        model = instantiate(self.config.models.model, logit_scale=logit_scale)
        return clip_model, model

    def get_data_train(self) -> Dataset:
        """
        Create the data based on the config file.
        """
        data = instantiate(self.config.datasets.train)
        return data

    def get_data_val(self) -> Dataset:
        """
        Create the data based on the config file.
        """
        data = instantiate(self.config.datasets.val)
        return data

    def get_data_zs(self) -> Dict[str, Dataset]:
        data = instantiate(self.config.zs_datasets.dst)
        return data

    def get_engine(self) -> DictConfig:
        """
        Create the engine based on the config file.
        """
        engine = instantiate(self.config.engine)
        return engine

    def get_optimizer(self, model: nn.Module) -> Optimizer:
        """
        Create the optimizer based on the config file.
        """

        optimizer = instantiate(self.config.optim.optimizer, params=model.parameters())
        return optimizer

    def get_loss(self) -> nn.Module:
        """
        Create the loss based on the config file.
        """
        loss = instantiate(self.config.losses.loss)
        return loss
