from os.path import join
from typing import Any, Dict, Union
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch


class EmbDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
    ) -> None:
        self.images_path = Path(join(root, split, "images"))
        self.captions_path = Path(join(root, split, "texts"))

        self.npy_images = sorted([str(f) for f in self.images_path.glob("*.npy")])
        self.npy_captions = sorted([str(f) for f in self.captions_path.glob("*.npy")])

        if not self.npy_images or not self.npy_captions:
            raise ValueError(f"No .npy files found in {self.images_path} or {self.captions_path}")

        if len(self.npy_images) != len(self.npy_captions):
            raise ValueError("The number of image and caption embedding files must match!")

        # Preload metadata to index embeddings within files
        self.image_lengths = [np.load(f, mmap_mode="r").shape[0] for f in self.npy_images]
        self.caption_lengths = [np.load(f, mmap_mode="r").shape[0] for f in self.npy_captions]

        if self.image_lengths != self.caption_lengths:
            raise ValueError("Image and caption file lengths do not match!")

        self.total_length = sum(self.image_lengths)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> Dict[str, Union[Any, str, int]]:
        # Locate the file and row corresponding to the index
        file_index = 0
        while index >= self.image_lengths[file_index]:
            index -= self.image_lengths[file_index]
            file_index += 1

        # Load the required row only
        img_file = self.npy_images[file_index]
        caption_file = self.npy_captions[file_index]

        img_emb = torch.tensor(np.load(img_file, mmap_mode="r")[index], dtype=torch.float16)
        caption_emb = torch.tensor(np.load(caption_file, mmap_mode="r")[index], dtype=torch.float16)

        return {
            "image": img_emb,
            "target": -1,  # Placeholder for labels, if needed
            "path": "",  # Optionally store path information if needed
            "index": index,
            "name": caption_emb,
        }
