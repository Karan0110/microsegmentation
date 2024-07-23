import os
import re
from pathlib import Path
from typing import Tuple, Callable, Union
from enum import Enum

import numpy as np

from PIL import Image

import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Expected directory structure:
# Base dir
# - Images
# -- image-{a}.png
# ...
# -- image-{b}.png
# - Labels
# -- label-{a}.png
# ...
# -- label-{b}.png

class Labels(Enum):
    POLYMERIZED = 0
    DEPOLYMERIZED = 1

class SyntheticDataset(Dataset):
    train_test_split=0.8
    
    def __init__(self, 
                 base_dir : Path, 
                 color_to_label : dict,
                 transform : Union[Callable, None] = None) -> None:
        super().__init__()

        self.base_dir : Path = base_dir

        self.color_to_label = color_to_label
        self.color_to_label_index = {color : Labels[self.color_to_label[color]].value for color in self.color_to_label}

        self.transform = transform
        
        images_index_range = self._get_index_range(dir_path=base_dir / 'Images',
                                                   file_name_stem="image")
        labels_index_range = self._get_index_range(dir_path=base_dir / 'Labels',
                                                   file_name_stem="label")
        if images_index_range != labels_index_range:
            raise ValueError(f"""Invalid file structure for data. Different index ranges for Images and Labels:
                                 Images: {images_index_range}
                                 Labels: {labels_index_range}
                             """)
        self.index_range = images_index_range

    def _get_index_range(self, 
                         dir_path : Path,
                         file_name_stem : str,
                         check_range : bool = True) -> range:
        file_names = os.listdir(dir_path)
        indices = []
        for file_name in file_names:
            if not file_name.endswith('.png'):
                continue

            match = re.search(file_name_stem + r'-(\d+)\.png', file_name)
            if match is None:
                raise ValueError(f"Found a .png file in {dir_path} named {file_name}. It should be of the form {file_name_stem}-[number].png")
            else:
                index = int(match.group(1))

                expected_file_name = f"{file_name_stem}-{index}.png"
                if expected_file_name != file_name:
                    raise ValueError(f"Found a .png file in {dir_path} named {file_name}. It should be of the form {file_name_stem}-[number].png")
                
                indices.append(index)

        start_index = min(indices)
        end_index = max(indices)+1

        index_range = range(start_index, end_index)

        if check_range:
            for index in index_range:
                expected_file_name = f"{file_name_stem}-{index}.png"
                expected_file_path = dir_path / expected_file_name

                if not expected_file_path.exists():
                    raise FileNotFoundError(f"Could not find file {expected_file_path}\n(index range: {index_range})")

        return index_range

    def __len__(self) -> int:
        return len(self.index_range)

    def __getitem__(self, raw_index : int):
        index = raw_index + min(self.index_range)

        image_path = self.base_dir / f'Images/image-{index}.png'
        image = Image.open(image_path)
        image = (np.array(image) / 255.).astype(np.float32)
        assert len(image.shape) == 2
        
        label_path = self.base_dir / f'Labels/label-{index}.png'
        label = Image.open(label_path)
        label = np.array(label)

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask'].to(torch.int64)

        #Convert masks to format that works with pytorch CrossEntropy loss
        mapping_array = [self.color_to_label_index.get(i, -1) for i in range(256)]  # Create a mapping list with default -1 for unmapped values
        mapping_tensor = torch.tensor(mapping_array)

        label = mapping_tensor[label]

        return image, label

def get_data_loaders(base_dir : Path, 
                     patch_size : int,
                     transform_config : list,
                     color_to_label : dict,
                     train_test_split : float,
                     batch_size : int) -> Tuple[DataLoader, DataLoader]:
    # Create transformation pipeline
    transform_list = []

    for item in transform_config:
        transform_type = item['name']
        transform_class = getattr(A, transform_type)
        transform = transform_class(**item['params'])
        transform_list.append(transform)

    transform_list.append(A.RandomCrop(height=patch_size, width=patch_size))
    transform_list.append(ToTensorV2())

    transform = A.Compose(transforms=transform_list)

    # Create the dataset
    dataset = SyntheticDataset(base_dir=base_dir, 
                               transform=transform,
                               color_to_label=color_to_label)

    # Define the lengths for train and test splits
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset into train and test sets
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader

#TODO - ask ChatGPT to list these again with all the tunable params listed
#     transform = A.Compose([
#     A.RandomRotate90(p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.Transpose(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
#     A.GridDistortion(p=0.5),
#     A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),
#     A.ElasticTransform(p=0.5),
#     A.GaussNoise(p=0.2),
#     A.MotionBlur(p=0.2),
#     A.MedianBlur(blur_limit=3, p=0.1),
#     A.Blur(blur_limit=3, p=0.1),
#     A.RandomBrightnessContrast(p=0.2),
#     A.HueSaturationValue(p=0.2),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])