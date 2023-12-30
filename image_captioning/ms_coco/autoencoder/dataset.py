import json
import os.path
from typing import Callable, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from vocabulary import Vocabulary


class COCOImageCaptioningTrainDataset(Dataset):
    def __init__(self,
                 vocabulary: Vocabulary,
                 annotations_filepath: str,
                 images_path: str,
                 img_transform: Callable = lambda x: x) -> None:
        self.coco = COCO(annotations_filepath)
        self.vocabulary = vocabulary
        self.images_path = images_path
        self.img_transform = img_transform
        self.mappings = dict(enumerate(list(self.coco.anns.keys()), 0))

    def __len__(self) -> int:
        return len(self.mappings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.coco.anns[self.mappings[idx]]
        img_path = self.coco.loadImgs(img['image_id'])[0]['file_name']

        return (
            self.img_transform(Image.open(os.path.join(self.images_path, img_path)).convert('RGB')),
            torch.Tensor(self.vocabulary.tokenize_caption(img['caption'])).long()
        )


class COCOImageCaptioningTestDataset(Dataset):
    def __init__(self,
                 annotations_filepath: str,
                 images_path: str,
                 img_transform: Callable = lambda x: x) -> None:
        self.images_path = images_path
        self.img_transform = img_transform
        self.mappings = dict(enumerate([item['file_name'] for item in json.loads(open(annotations_filepath).read())['images']], 0))

    def __len__(self) -> int:
        return len(self.mappings)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, torch.Tensor]:
        img_path = self.mappings[idx]
        img = Image.open(os.path.join(self.images_path, img_path)).convert('RGB')

        return (
            img,
            self.img_transform(img)
        )
