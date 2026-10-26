"""Evaluate a pretrained ViLU or LVU checkpoint on ImageNet-1k val (failure detection for CLIP zero-shot).

    python scripts/eval_imagenet.py --ckpt weights/vilu_imagenet.ckpt --model vilu --data-root data/ImageNet
    python scripts/eval_imagenet.py --ckpt weights/lvu_imagenet.ckpt  --model lvu  --data-root data/ImageNet

Prints AUC, FPR at 95% TPR and AUPR-error for the model and for the MCM baseline (maximum softmax
probability of CLIP), and CLIP's zero-shot accuracy, as in the evaluation of `vilu/engine/trainer.py`.
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from vilu import lib
from vilu.dataset_uq import ConvertImageToRGB, ImageNetDataset, get_templates
from vilu.models import ViLUAttention
from vilu.models.clip import load as load_clip
from vilu.models.clip import tokenize


def failure_metrics(scores: np.ndarray, correct: np.ndarray) -> dict:
    success, failure = scores[correct], scores[~correct]
    return {
        "auc": lib.get_auroc(success, failure),
        "fpr95": lib.get_fpr(success, failure),
        "aupr_err": lib.get_aupr_out(-success, -failure),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", choices=["vilu", "lvu"], required=True)
    parser.add_argument("--data-root", default="data/ImageNet", help="ImageNet in torchvision layout")
    parser.add_argument("--backbone", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    device = "cuda"
    clip_model, _ = load_clip(args.backbone, device=device)
    clip_model.eval()

    lvu = args.model == "lvu"
    model = ViLUAttention(concat=True, identity_init=True, use_predicted_caption=not lvu, use_attention=not lvu,
                          use_sigmoid=lvu, logit_scale=clip_model.logit_scale.exp().item())
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        ConvertImageToRGB(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = ImageNetDataset(args.data_root, split="val", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    template = get_templates("base")[0]  # "a photo of a {}.", as in training
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast:
        text_feats = clip_model.encode_text(tokenize([template.format(c) for c in dataset.class_names]).to(device))
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    scores, mcm, correct = [], [], []
    for batch in loader:
        with autocast:
            img_feats = clip_model.encode_image(batch["image"].to(device))
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            scores.append(model(img_feats, text_feats).squeeze(-1).float().cpu())
        probs = torch.softmax(model.logit_scale * img_feats.float() @ text_feats.float().t(), dim=-1)
        mcm.append(probs.max(dim=-1).values.cpu())
        correct.append(probs.argmax(dim=-1).cpu().eq(batch["target"]))

    scores, mcm, correct = torch.cat(scores).numpy(), torch.cat(mcm).numpy(), torch.cat(correct).numpy()
    print(json.dumps({
        "ckpt": args.ckpt,
        "clip_zero_shot_acc": float(correct.mean()),
        args.model: failure_metrics(scores, correct),
        "mcm": failure_metrics(mcm, correct),
    }, indent=2))


if __name__ == "__main__":
    main()
