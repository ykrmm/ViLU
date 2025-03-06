# flake8: noqa

from typing import Callable, Any, Dict, Union
from pathlib import Path

from torch import Tensor
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(CIFAR10):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Union[Callable[..., Any], None] = None,
        target_transform: Union[Callable[..., Any], None] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.class_names = self.classes
        self.n_classes = len(self.class_names)
        self.idx_to_class = {idx: c for (c, idx) in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)

        return {
            "image": img,
            "target": target,
            "path": "",
            "index": index,
            "name": self.class_names[target],
        }
