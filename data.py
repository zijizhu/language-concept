from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import re

class SUNDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        super().__init__()
        split_mat = loadmat(Path("data") / "SUN" / "splits.mat", squeeze_me=True)

        self.split_indices = {
            'train': split_mat['train_loc'] - 1,
            'val': split_mat['val_loc'] - 1,
            'test': split_mat['test_seen_loc'] - 1
        }
        self.split = split

        self.images = loadmat(Path(data_dir) / "SUNAttributeDB" / "images.mat", squeeze_me=True)['images'].tolist()
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
        class_name = ' '.join(re.split("_|/", class_name))
        img = Image.open(Path(self.data_dir) / "images" / path).convert('RGB')
        class_idx = self.classes[class_name]

        return self.transforms(img), class_idx
