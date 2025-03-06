from typing import Optional, Callable, Dict, Union
import os
import csv
from torch import Tensor
from torch.utils.data import Dataset
from transformers import CLIPTokenizer  # type: ignore
import PIL
from PIL import Image
import random
import re

Image.MAX_IMAGE_PIXELS = None


# Define the custom dataset
class PixMoDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[PIL.Image.Image], Tensor]] = None,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        split: str = "train",
    ) -> None:
        """
        Args:
            image_dir (str): Directory with all the images.
            captions_file (str): Path to the CSV file with image filenames and captions.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root = root
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.data = []

        captions_file = os.path.join(root, "pixmo_cap.csv")

        # Load the image-caption pairs from the CSV file
        with open(captions_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({"image_filename": row["image_filename"], "caption": row["caption"]})

        if split == "val":
            self.data = self.data[:10000]
        else:
            self.data = self.data[10000:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str, int]]:
        item = self.data[idx]
        image_path = os.path.join(self.root, item["image_filename"])
        caption = item["caption"]

        sub_caption = self.subcaption(caption)

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {"image": image, "target": sub_caption, "path": "", "index": idx, "name": sub_caption}

    def subcaption(self, caption: str) -> str:
        sub_caption = []

        # Step 1: Split caption into sentences.
        delimiter = r"\. "
        split_caption = [sentence + "." for sentence in re.split(delimiter, caption.strip()) if sentence]
        n = len(split_caption)

        # Step 2: Tokenize sentences. Truncate the sentence if the number of tokens is > max_length
        sentences_tokens = self.tokenizer(split_caption, truncation=True, max_length=77)["input_ids"]

        # Step 3: Sample a starting index
        start_idx = random.choice(range(n))

        while start_idx < n:
            if len(sub_caption) + len(sentences_tokens[start_idx]) > self.max_length:
                break
            sub_caption.extend(sentences_tokens[start_idx])
            start_idx += 1

        return self.tokenizer.decode(sub_caption, skip_special_tokens=True)
