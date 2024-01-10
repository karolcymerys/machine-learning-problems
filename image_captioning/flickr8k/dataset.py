import csv
import os.path
from typing import Callable, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from vocabulary import Vocabulary


class Flickr8kImageCaptioningTrainDataset(Dataset):
    def __init__(self,
                 vocabulary: Vocabulary,
                 root: str,
                 images_path: str = 'Images',
                 img_transform: Callable = lambda x: x) -> None:
        self.vocabulary = vocabulary
        self.images_path = os.path.join(root, images_path)
        self.img_transform = img_transform
        with open(os.path.join(root, 'captions.txt')) as file:
            csv_reader = csv.DictReader(file)
            self.mappings = dict(enumerate([row for row in csv_reader], 0))

    def __len__(self) -> int:
        return len(self.mappings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ann = self.mappings[idx]

        return (
            self.img_transform(Image.open(os.path.join(self.images_path, ann['image'])).convert('RGB')),
            torch.Tensor(self.vocabulary.tokenize_caption(ann['caption'])).long()
        )


class Flickr8kImageCaptioningTestDataset(Dataset):
    def __init__(self,
                 root: str,
                 images_path: str,
                 img_transform: Callable = lambda x: x) -> None:
        self.images_path = os.path.join(root, images_path)
        self.img_transform = img_transform
        with open(os.path.join(root, 'captions.txt')) as file:
            csv_reader = csv.DictReader(file)
            self.mappings = dict(enumerate([row for row in csv_reader], 0))

    def __len__(self) -> int:
        return len(self.mappings)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, torch.Tensor]:
        ann = self.mappings[idx]
        img = Image.open(os.path.join(self.images_path, ann['image'])).convert('RGB')

        return (
            img,
            self.img_transform(img)
        )
