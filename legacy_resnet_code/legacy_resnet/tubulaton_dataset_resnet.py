import os
import re
from pathlib import Path
from typing import Tuple, Callable, Union

import numpy as np

from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from elastic_transform import ElasticTransform

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
        
        file_names = os.listdir(self.base_dir / 'Control')
        indexes = []
        for file_name in file_names:
            if not file_name.endswith('.png'):
                continue

            match = re.search(r'image-(\d+)\.png', file_name)
            if match is None:
                raise FileNotFoundError(f"Found a .png file in Control named {file_name}. It should be of the form image-[number].png")
            else:
                index = int(match.group(1))

                expected_file_name = f"image-{index}.png"
                if expected_file_name != file_name:
                    raise FileNotFoundError(f"Found a .png file in Control named {file_name}. It should be of the form image-[number].png")
                
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
                     patch_size : int,
                     transform_config : list) -> Tuple[DataLoader, DataLoader]:
    # Create transformation pipeline
    transform_list = []

    for item in transform_config:
        transform_type = item.pop('type')

        if transform_type == 'ElasticTransform':
            transform_class = ElasticTransform
        else:
            transform_class = getattr(transforms, transform_type)

        transform = transform_class(**item)
        transform_list.append(transform)
    transform_list.append(transforms.RandomResizedCrop(patch_size))
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms=transform_list)

    # Create DataLoaders
    train_set = ProcessedTubulatonDataset(base_dir=base_dir, for_training=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=0)

    test_set = ProcessedTubulatonDataset(base_dir=base_dir, for_training=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=50, shuffle=True, num_workers=0)

    return train_loader, test_loader
