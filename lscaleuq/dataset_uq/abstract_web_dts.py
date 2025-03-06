import glob
from typing import Callable, Optional

import PIL
from torch import Tensor
import webdataset as wds
from os.path import join


def get_webdataset(
    root: str,
    split: str,
    transform: Optional[Callable[[PIL.Image.Image], Tensor]] = None,
    length: int = None,
    batch_size: int = 1,
) -> wds.WebDataset:
    path = join(root, split)
    tar_files = glob.glob(join(path, "*.tar"))
    if not tar_files:
        raise ValueError(f"No .tar files found in {path}")
    dataset = wds.WebDataset(tar_files, empty_check=False).shuffle(100).decode("pil").to_tuple("jpg", "txt").map_tuple(transform, lambda x: x).batched(batch_size).with_length(length // batch_size)
    return dataset
