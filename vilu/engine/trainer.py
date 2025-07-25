from typing import Dict, Optional
import os
from collections import defaultdict
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import vilu.lib as lib
from vilu.lib.lightning_fabric import LightningFabric
from vilu.models.clip import tokenize as clip_tokenize
from vilu.dataset_uq.templates import get_templates
from vilu.losses import gather_features


class Trainer:
    def __init__(
        self,
        fabric: LightningFabric,
        clip_model: nn.Module,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        loss_fn: nn.Module,
        n_epochs: int,
        eval_only: bool,
        eval_freq: int,
        print_freq: int,
        max_iter_epoch: int,
        use_gather: bool,
        save_freq: int,
        template_type: str,
        web_dataset: bool,
        caption_dataset: bool,
        save_dir: str,
        resume: bool,
        batchwise_train: bool,
        display_progress: bool,
    ) -> None:
        self.fabric = fabric
        self.clip_model = clip_model
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.n_epochs = n_epochs
        self.eval_only = eval_only
        self.eval_freq = eval_freq
        self.print_freq = print_freq
        self.max_iter_epoch = max_iter_epoch
        self.use_gather = use_gather
        self.save_freq = save_freq
        self.template_type = template_type
        self.web_dataset = web_dataset
        self.caption_dataset = caption_dataset
        self.save_dir = save_dir
        self.resume = resume
        self.batchwise_train = batchwise_train
        self.display_progress = display_progress
        self.template = get_templates(template_type)

        # Load ckpt for model, optim, scheduler, start_epoch
        ckpt_path = os.path.join(self.save_dir, "last.ckpt") if self.resume else None
        self.start_epoch = self.load_ckpt(ckpt_path)

    def fit(self) -> None:
        self.fabric.info("Starting training")
        for epoch in range(self.start_epoch, self.n_epochs):
            if not self.eval_only:
                train_meter = self.train_one_epoch(epoch)
                self.log_metrics_train(train_meter, epoch)
                self.save_ckpt(train_meter, epoch)

            if self.evaluation_time(epoch):
                val_meter = self.evaluate(self.val_loader)
                self.log_metrics_eval(val_meter, epoch)

            if self.eval_only:
                self.fabric.barrier()
                break

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.fabric.info(f"Training epoch {epoch}")
        meter = lib.DictAverage()
        progress = lib.ProgressMeter(len(self.train_loader), meter, prefix=f"Epoch: [{epoch}]")

        # Case Image-classification dataset with fixed set of labels
        if not self.batchwise_train:
            class_names = self.train_loader.dataset.class_names
            class_names = clip_tokenize([self.template[0].format(name) for name in class_names])
            with torch.no_grad(), self.fabric.autocast():
                text_feats = self.clip_model.encode_text(self.fabric.to_device(class_names))
                text_feats /= text_feats.norm(dim=-1, keepdim=True)

        self.model.train()
        self.optimizer.zero_grad()

        track_loader = lib.track(self.train_loader) if self.fabric.is_global_zero else self.train_loader
        for i, batch in enumerate(track_loader):
            images = batch[0] if self.web_dataset else batch["image"]
            if self.batchwise_train:
                captions = batch[1] if self.web_dataset else batch["name"]
                labels = self.fabric.to_device(torch.arange(images.shape[0]))
            else:
                labels = batch["target"]

            with torch.no_grad(), self.fabric.autocast():
                visual_feats = self.clip_model.encode_image(images)
                visual_feats /= visual_feats.norm(dim=-1, keepdim=True)
                # Encoding captions
                if self.batchwise_train:
                    tokenized = self.fabric.to_device(clip_tokenize(captions, truncate=True))
                    text_feats = self.clip_model.encode_text(tokenized)
                    text_feats /= text_feats.norm(dim=-1, keepdim=True)
                
                # Multi-GPU case
                if self.batchwise_train and self.use_gather:
                    visual_feats, text_feats = gather_features(visual_feats, text_feats, self.fabric, gather_with_grad=False)
                    labels = torch.arange(visual_feats.shape[0], device=visual_feats.device, dtype=torch.long)

                probs = torch.softmax(self.model.logit_scale * visual_feats @ text_feats.t(), dim=-1)

            with self.fabric.autocast():
                # Equivalent to predict uncertainty i.e 1 - confid_scores and inverse the binary labels. 
                confid_scores = self.model(visual_feats, text_feats).squeeze(-1)
                loss = self.loss_fn(confid_scores, probs, labels)

            self.fabric.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.batchwise_train:
                missclassified = probs.diag() != probs.max(dim=-1).values
            else:
                missclassified = labels != probs.argmax(dim=-1)

            confidence_success = confid_scores[~missclassified].cpu().detach().numpy()
            confidence_failure = confid_scores[missclassified].cpu().detach().numpy()

            meter.update(
                {
                    "loss": loss,
                    "fpr": lib.get_fpr(confidence_success, confidence_failure),
                    "auc": lib.get_auroc(confidence_success, confidence_failure),
                },
                images.size(0),
            )

            if self.fabric.is_global_zero:
                track_loader.set_postfix({"avg_loss": meter.avg["loss"], "loss": loss.item()})
                if i % self.print_freq == 0 and self.display_progress:
                    progress.display(i)



        self.lr_scheduler.step()

        if self.fabric.is_global_zero:
            progress.display_summary()

        return meter

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> lib.DictAverage:
        dts_name = val_loader.dataset.__class__.__name__
        self.fabric.info(f"Evaluating on {dts_name}")
        metrics = defaultdict(dict)
        if not self.batchwise_train:
            class_names = val_loader.dataset.class_names
            class_names = clip_tokenize([self.template[0].format(name) for name in class_names])

            with self.fabric.autocast():
                text_feats = self.clip_model.encode_text(self.fabric.to_device(class_names))
                text_feats /= text_feats.norm(dim=-1, keepdim=True)

        mode = self.model.training
        self.model.eval()
        self.optimizer.zero_grad()
        confidence_success = []
        confidence_failure = []
        missclassifieds = []
        mcps_success = []
        mcps_failure = []
        shannon_success = []
        shannon_failure = []
        doctor_success = []
        doctor_failure = []

        track_loader = lib.track(val_loader) if self.fabric.is_global_zero else val_loader
        for _, batch in enumerate(track_loader):
            images = batch[0] if self.web_dataset  else batch["image"]
            if self.batchwise_train:
                captions = batch[1] 
                labels = self.fabric.to_device(torch.arange(images.shape[0]))
            else:
                labels = batch["target"]

            with self.fabric.autocast():
                visual_feats = self.clip_model.encode_image(self.fabric.to_device(images))
                visual_feats /= visual_feats.norm(dim=-1, keepdim=True)
                if self.batchwise_train:
                    tokenized = self.fabric.to_device(clip_tokenize(captions, truncate=True))
                    text_feats = self.clip_model.encode_text(tokenized)
                    text_feats /= text_feats.norm(dim=-1, keepdim=True)

            if self.batchwise_train and self.use_gather:
                visual_feats, text_feats = gather_features(visual_feats, text_feats, self.fabric, gather_with_grad=False)
            try:
                probs = torch.softmax(self.model.logit_scale * visual_feats @ text_feats.t(), dim=-1)
            except: # sometimes bug with float16 and float32
                text_feats = text_feats.to(torch.float16)
                probs = torch.softmax(self.model.logit_scale * visual_feats @ text_feats.t(), dim=-1)
            confid_scores = self.model(visual_feats, text_feats).squeeze(-1)

            if self.batchwise_train  and self.use_gather:
                missclassified = probs.diag() != probs.max(dim=-1).values
            else:
                # Gather tensors on same node:
                confid_scores = self.fabric.all_gather(confid_scores).reshape(-1)
                labels = self.fabric.all_gather(labels).reshape(-1)
                probs = self.fabric.all_gather(probs).reshape(-1, probs.shape[-1])
                missclassified = labels != probs.argmax(dim=-1)
            
            mcps = probs.max(dim=-1).values
            with self.fabric.autocast():
                entropy = - (-torch.sum(probs * torch.log(probs + 1e-6), dim=-1))  # less entropy, more uncertain
                entropy /= torch.log(torch.tensor(probs.size(-1)) + 1e-6)
                doctor_uq = torch.sum(torch.tensor(probs)**2, dim=-1)
                doctor_uq = - (1-doctor_uq)/doctor_uq
            confidence_success.append(confid_scores[~missclassified])
            confidence_failure.append(confid_scores[missclassified])
            missclassifieds.append(missclassified)
            shannon_success.append(entropy[~missclassified])
            shannon_failure.append(entropy[missclassified])
            doctor_success.append(doctor_uq[~missclassified])
            doctor_failure.append(doctor_uq[missclassified])
            mcps_success.append(mcps[~missclassified])
            mcps_failure.append(mcps[missclassified])

        missclassifieds = torch.cat(missclassifieds).cpu().numpy()
        confidence_success = torch.cat(confidence_success).cpu().numpy()
        confidence_failure = torch.cat(confidence_failure).cpu().numpy()
        shannon_success = torch.concatenate(shannon_success).cpu().numpy()
        shannon_failure = torch.concatenate(shannon_failure).cpu().numpy()
        doctor_success = torch.concatenate(doctor_success).cpu().numpy()
        doctor_failure = torch.concatenate(doctor_failure).cpu().numpy()
        mcps_success = torch.cat(mcps_success).float().cpu().numpy()
        mcps_failure = torch.cat(mcps_failure).float().cpu().numpy()

        # clip_acc = 1- missclassifieds.sum()/len(missclassifieds)

        metrics["fpr_vilu"] = lib.get_fpr(confidence_success, confidence_failure)
        metrics["auc_vilu"] = lib.get_auroc(confidence_success, confidence_failure)
        metrics["aupr_err_vilu"] = lib.get_aupr_out(-confidence_success, -confidence_failure)

        metrics["fpr_mcp"] = lib.get_fpr(mcps_success, mcps_failure)
        metrics["auc_mcp"] = lib.get_auroc(mcps_success, mcps_failure)
        metrics["aupr_err_mcp"] = lib.get_aupr_out(-mcps_success, -mcps_failure)
        # Shannon entropy
        metrics["fpr_shannon"] = lib.get_fpr(shannon_success, shannon_failure)
        metrics["auc_shannon"] = lib.get_auroc(shannon_success, shannon_failure)
        metrics["aupr_err_shannon"] = lib.get_aupr_out(-shannon_success, -shannon_failure)
        
        metrics["fpr_doctor"] = lib.get_fpr(doctor_success, doctor_failure)
        metrics["auc_doctor"] = lib.get_auroc(doctor_success, doctor_failure)
        metrics["aupr_err_doctor"] = lib.get_aupr_out(-doctor_success, -doctor_failure)

        self.model.train(mode)
        return metrics

    def evaluation_time(self, epoch: int) -> bool:
        last_epoch = epoch + 1 == self.n_epochs
        eval_freq_time = epoch % self.eval_freq == 0
        return eval_freq_time or last_epoch or self.eval_only



    def log_metrics_train(self, train_meter: lib.DictAverage, epoch: int) -> None:
        if self.fabric.is_global_zero:
            for metric, value in train_meter.avg.items():
                self.fabric.info(f" * train_{metric} = {value:.3f}")

    def log_metrics_eval(self, val_meter: lib.DictAverage, epoch: int) -> None:
        for m in ["_vilu","_doctor","_shannon","_mcp"]:
            self.fabric.info(f" * auc{m} = {val_meter[f'auc{m.strip()}']:.3f}, fpr{m} = {val_meter[f'fpr{m.strip()}']:.3f}, aupr_err{m} = {val_meter[f'aupr_err{m.strip()}']:.3f}")

    def save_ckpt(
        self,
        train_meter: lib.DictAverage,
        epoch: int,
    ) -> None:
        state = {
            "epoch": epoch,
            "train_meter": {k: v.avg for k, v in train_meter.items()},
            "state_dict": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.lr_scheduler.state_dict(),
        }
        try:
            self.fabric.save(os.path.join(self.save_dir, "last.ckpt"), state)
        except Exception as e:
            self.fabric.warning(f"Unable to save the checkpoint at epoch {epoch} (see error below). Training will proceed. \n {e}")

    def load_ckpt(self, ckpt_path: Optional[str] = None) -> int:
        if ckpt_path is not None:
            state = {
                "epoch": -1,
                "train_meter": {},
                "state_dict": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.lr_scheduler,
            }
            if os.path.exists(ckpt_path):
                self.fabric.load(ckpt_path, state)
                self.fabric.info(f"Resuming training from epoch {state['epoch']}")
                return state["epoch"]
            else:
                self.fabric.info(f"Chekpoint file ``{ckpt_path}'' not found. Starting training from scratch")
            return 0

        else:
            self.fabric.info("Starting training from scratch")
            return 0
