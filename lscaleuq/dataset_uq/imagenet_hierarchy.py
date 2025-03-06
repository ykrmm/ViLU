# flake8: noqa

from typing import Dict, Union
import json
import os
from os.path import dirname, abspath
from torch import Tensor
from torchvision.datasets import ImageNet


class ImageNetSimpleHierarchyDataset(ImageNet):
    def __init__(
        self,
        root,
        split="train",
        hierarchical_level=1,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
        )
        assert hierarchical_level in [0, 1, 2, 3], f"the hierachical level provided ({hierarchical_level}) is not valid. Acceptable values are 0, 1, 2, 3."
            
        path_to_datasets = dirname(abspath(__file__))
        json_classes = os.path.join(
            path_to_datasets, "files", "imagenet_class_name_clip_hierarchical.json"
        )
        with open(json_classes, "r") as f:
            self.class_names_json = json.load(f)
        self.class_names = [
            self.class_names_json[str(i)][hierarchical_level] for i in range(len(self.class_names_json))
        ]

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)

        return {
            "image": img,
            "target": target,
            "path": "",
            "index": index,
            "name": self.class_names[target],
        }
