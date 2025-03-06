import os
from os.path import dirname, abspath
import json
from typing import Dict, Union
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ImageNetEmbDataset(Dataset):
    def __init__(
        self,
        root: str,
        clip_backbone: str,
        split: str = "train",
    ) -> None:
        """
        ImageNet Embedding dataset

        Parameters:
        root (str): Path to embedding,label directory.
        clip_backbone (str): CLIP backbone model name.
        json_classes_path (str): Path to the json containing ImageNet classes.
        split (str):  ('train' or 'val').
        """
        self.split = split
        clip_backbone = clip_backbone.replace("/", "-")
        self.embeddings_dir = os.path.join(root, clip_backbone, split)
        path_to_datasets = dirname(abspath(__file__))
        json_classes = os.path.join(path_to_datasets, "files", "imagenet_class_name_clip.json")
        with open(json_classes, "r") as f:
            self.class_names_json = json.load(f)
        self.embeddings_files = sorted([os.path.join(self.embeddings_dir, f) for f in os.listdir(self.embeddings_dir) if f.endswith(".pt")])
        self.class_names = [self.class_names_json[str(i)] for i in range(len(self.class_names_json))]

    def __len__(self) -> int:
        return len(self.embeddings_files)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        embeddings_path = self.embeddings_files[index]
        visual_feature, target = torch.load(embeddings_path)
        return {
            "image": visual_feature,
            "target": target,
            "path": "",
            "index": index,
            "name": self.class_names[target],
        }
