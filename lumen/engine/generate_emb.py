import logging
import os
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset
import lumen.models.clip as clip


class GenerateEmbeddings:
    def __init__(
        self,
        config: DictConfig,
        train_dts: Dataset,
        val_dts: Dataset,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.train_dts = train_dts
        self.val_dts = val_dts
        self.device = device
        self.logger = logger

        # Dataloaders
        self.train_loader = DataLoader(
            train_dts,
            batch_size=self.config.engine.batch_size,
            shuffle=False,
            num_workers=self.config.engine.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False,
        )

        self.val_loader = DataLoader(
            val_dts,
            batch_size=config.engine.batch_size,
            shuffle=False,
            num_workers=config.engine.workers,
            pin_memory=True,
            drop_last=False,
        )

        # Load CLIP
        self.clip_model, _ = clip.load(
            config.backbone_model.clip_name,
            device=self.device,
            download_root=config.backbone_model.download_root,
        )

    def run(self) -> None:
        save_path = self.config.engine.save_visual_features
        dataset_name = self.config.datasets.name
        clip_name = self.config.backbone_model.clip_name
        clip_name = clip_name.replace("/", "-")
        save_dir_train = os.path.join(save_path, dataset_name, clip_name, "train")
        save_dir_val = os.path.join(save_path, dataset_name, clip_name, "val")
        os.makedirs(save_dir_train, exist_ok=True)
        os.makedirs(save_dir_val, exist_ok=True)

        self.clip_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["target"].to(self.device, non_blocking=True)
                visual_features = self.clip_model.encode_image(images)
                visual_features /= visual_features.norm(dim=-1, keepdim=True)

                for j, (feature, label) in enumerate(zip(visual_features, labels)):
                    index = i * self.config.engine.batch_size + j
                    save_path = os.path.join(save_dir_train, f"visual_emb_{index}.pt")
                    torch.save((feature.cpu(), label.cpu()), save_path)
                    self.logger.info(
                        f"Saved visual feature and label for image {index} to {save_path}"
                    )

            for i, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["target"].to(self.device, non_blocking=True)
                visual_features = self.clip_model.encode_image(images)
                visual_features /= visual_features.norm(dim=-1, keepdim=True)

                for j, (feature, label) in enumerate(zip(visual_features, labels)):
                    index = i * self.config.engine.batch_size + j
                    save_path = os.path.join(save_dir_val, f"visual_emb_{index}.pt")
                    torch.save((feature.cpu(), label.cpu()), save_path)
                    self.logger.info(
                        f"Saved visual feature and label for image {index} to {save_path}"
                    )

        self.logger.info("Finished generating visual features.")
