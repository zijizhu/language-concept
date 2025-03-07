import re
from pathlib import Path
import pickle as pkl
from typing import Optional, Callable

import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


class SUNDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str):
        super().__init__()
        split_mat = loadmat((Path("data") / "SUN" / "splits.mat").as_posix(), squeeze_me=True)

        self.split_indices = {
            'train': split_mat['train_loc'] - 1,
            'val': split_mat['val_loc'] - 1,
            'test': split_mat['test_seen_loc'] - 1
        }
        self.split = split

        self.images = loadmat((Path(data_dir) / "SUNAttributeDB" / "images.mat").as_posix(), squeeze_me=True)['images'].tolist()
        with open("data/SUN/sun_classes.txt") as fp:
            classes = fp.read().splitlines()
        self.classes = {c: i for i, c in enumerate(classes)}

        self.transforms = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.split_indices[self.split])

    def __getitem__(self, index: int):
        split_idx = self.split_indices[self.split][index]
        path = self.images[split_idx]

        class_name = path.split('/', 1)[-1].rsplit('/', 1)[0]
        class_name = ' '.join(re.split("[_/]", class_name))
        img = Image.open(Path(self.data_dir) / "images" / path).convert('RGB')
        class_idx = self.classes[class_name]

        return self.transforms(img), class_idx


attribute_indices = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91,
                     93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181,
                     183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253,
                     254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

class CUBConceptDataset(ImageFolder):
    def __init__(self,
                 image_root: str | Path,
                 return_attributes: bool = False,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(
            root=image_root,
            transform=transforms,
            target_transform=target_transform
        )

        with open(Path('data') / 'CUB' / 'class_attr_data_10' / "train.pkl", "rb") as fp:
            train_attribute_anns = pkl.load(fp)

        label2attr = dict()
        for ann in train_attribute_anns:
            label, attribute_vector = ann["class_label"], ann["attribute_label"]
            if label not in label2attr:
                label2attr[label] = attribute_vector

        self.attributes = torch.tensor([label2attr[i] for i in range(len(label2attr))], dtype=torch.long)

        with open(Path('data') / 'CUB' / 'cub_attributes_cleaned.txt', "r") as fp:
            all_attribute_texts = fp.read().splitlines()
        self.attribute_texts = [all_attribute_texts[i] for i in attribute_indices]

        self.return_attributes = return_attributes

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im_pt = self.transform(Image.open(im_path).convert("RGB"))
        attr = self.attributes[label]
        if self.return_attributes:
            return im_pt, label, attr
        return im_pt, label

