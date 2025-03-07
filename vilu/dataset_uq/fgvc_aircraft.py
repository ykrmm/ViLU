from typing import Dict, List, Any, Union

from torch import Tensor
from torchvision.datasets import FGVCAircraft

ArgsType = List[Any]
KwargsType = Dict[str, Any]


class FGVCAircraftDataset(FGVCAircraft):
    def __init__(self, root: str, *args: ArgsType, **kwargs: KwargsType) -> None:
        root = root
        super().__init__(root, *args, **kwargs)
        self.labels = self.targets = self._labels
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_names = self.classes

        self.shift_type = "original"
        self.n_classes = len(self.class_names)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
            "path": self._image_files[index],
            "index": index,
            "name": self.idx_to_class[target],
        }
