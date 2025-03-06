from typing import Optional, Callable, Dict, Union
import os
from os.path import dirname, abspath

import PIL
from torch import Tensor
from torch.utils.data import Dataset

import lscaleuq.lib as lib


class DTDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable[[PIL.Image.Image], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        assert split in ["train", "test"]
        db = lib.load_json(os.path.join(dirname(abspath(__file__)), "files", "data_suite_splits", "split_zhou_DescribableTextures.json"))
        db = db[split]

        self.paths = []
        self.targets = []
        self.names = []
        for x in db:
            self.paths.append(os.path.join(self.root, "dtd", "images", x[0]))
            self.targets.append(x[1])
            self.names.append(x[2])

        # unique names
        self.class_names = []
        for nm in self.names:
            if nm not in self.class_names:
                self.class_names.append(nm)

        self.class_to_idx = {nm: i for i, nm in enumerate(self.class_names)}
        self.labels = self.targets = [self.class_to_idx[nm] for nm in self.names]

        self.n_classes = len(self.class_names)
        self.idx_to_class = {idx: c for (c, idx) in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        path = self.paths[index]
        target = self.targets[index]
        img = PIL.Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "target": target,
            "path": path,
            "index": index,
            "name": self.idx_to_class[target],
        }
