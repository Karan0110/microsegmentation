import os
import re
from pathlib import Path
from typing import Tuple, Callable, Union

import numpy as np

from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Expected directory structure:
# Base dir
# - Control
# -- image-{a}.png
# ...
# -- image-{b}.png
# - Depoly
# -- image-{a}.png
# ...
# -- image-{b}.png

# We use labels:
# (1,0) = Control 
# (0,1) = Depoly 

class ProcessedTubulatonDataset(Dataset):
    train_test_split=0.8
    
    def __init__(self, 
                 base_dir : Path, 
                 transform : Union[Callable, None] = None, 
                 for_training : bool = True) -> None:
        super().__init__()
        
        self.base_dir : Path = base_dir
        self.transform = transform
        
        filenames = os.listdir(self.base_dir / 'Control')
        indexes = []
        for xs in filenames:
            if not xs.endswith('.png'):
                continue

            match = re.search(r'image-(\d+)\.png', xs)
            if match is None:
                raise FileNotFoundError(f"Found a .png file in Control named {xs}. It should be of the form image-[number].png")
            else:
                index = int(match.group(1))
                indexes.append(index)
        
        self.start_index = min(indexes)
        self.end_index = max(indexes)+1
        
        self.train_cutoff = int(self.start_index + ProcessedTubulatonDataset.train_test_split * (self.end_index - self.start_index))
        
        self.for_training = for_training
        
        if self.for_training:
            self.end_index = self.train_cutoff
        else:
            self.start_index = self.train_cutoff

    def __len__(self) -> int:
        return 2 * (self.end_index - self.start_index)

    def __getitem__(self, raw_idx):
        if raw_idx >= len(self) // 2:
            idx = raw_idx - (len(self)//2) + self.start_index
            label = np.array([0., 1.]).astype(np.float32)
            image_path = self.base_dir / f'Depoly/image-{idx}.png'
        else:
            idx = raw_idx + self.start_index
            label = np.array([1., 0.]).astype(np.float32)
            image_path = self.base_dir / f'Control/image-{idx}.png'

        image = Image.open(image_path)
        
        image = (np.array(image) / 255.).astype(np.float32)
        assert len(image.shape) == 2
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(base_dir : Path, 
                     patch_size : int) -> Tuple[DataLoader, DataLoader]:
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(patch_size, pad_if_needed=True, padding_mode='symmetric'),
    ])

    train_set = ProcessedTubulatonDataset(base_dir=base_dir, for_training=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=0)

    test_set = ProcessedTubulatonDataset(base_dir=base_dir, for_training=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=True, num_workers=0)

    return train_loader, test_loader
