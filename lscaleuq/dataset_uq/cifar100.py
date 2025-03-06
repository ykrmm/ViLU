# flake8: noqa

from typing import Callable, Any, Dict, Union
from pathlib import Path

from torch import Tensor
from torchvision.datasets import CIFAR100


class CIFAR100Dataset(CIFAR100):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Union[Callable[..., Any], None] = None,
        target_transform: Union[Callable[..., Any], None] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.shift_type = "original"
        self.class_names = self.classes
        self.n_classes = len(self.class_names)
        self.labels = self.targets
        self.idx_to_class = {idx: c for (c, idx) in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)

        return {
            "image": img,
            "target": target,
            "path": "",
            "index": index,
            "name": self.idx_to_class[target],
        }


if __name__ == "__main__":
    dts = CIFAR100Dataset("/share/DEEPLEARNING/datasets/CIFAR-100")
    import ipdb

    ipdb.set_trace()
