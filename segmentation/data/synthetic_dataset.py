import os
import multiprocessing
import re
from pathlib import Path
from typing import Tuple, Callable, Union, Optional

import numpy as np

from PIL import Image

import torch

from torch.utils.data import DataLoader, Dataset, random_split

# Prevent auto-updating of albumentation (so program can run properly without internet)
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .labels import Labels

# Expected directory structure:
# Base dir
# - Images
# -- image-{a}.png
# ...
# -- image-{b}.png
# - Masks
# -- mask-{a}.png
# ...
# -- mask-{b}.png

class SyntheticDataset(Dataset):
    def __init__(self, 
                 base_dir : Path, 
                 color_to_label : dict,
                 transform : Union[Callable, None] = None,
                 max_batches : Union[None, int] = None) -> None:
        super().__init__()

        self.base_dir : Path = base_dir

        self.color_to_label = color_to_label
        self.color_to_label_index = {color : Labels[self.color_to_label[color]].value for color in self.color_to_label}

        self.transform = transform

        images_index_range = self._get_index_range(dir_path=base_dir / 'Images',
                                                   file_name_stem="image")
        masks_index_range = self._get_index_range(dir_path=base_dir / 'Masks',
                                                   file_name_stem="mask")
        if images_index_range != masks_index_range:
            raise ValueError(f"""Invalid file structure for data. Different index ranges for Images and masks:
                                 Images: {images_index_range}
                                 Masks: {masks_index_range}
                             """)
        self.index_range = images_index_range

        if max_batches is not None:
            new_max = min(max(self.index_range), min(self.index_range) + max_batches - 1)

            self.index_range = range(min(self.index_range), new_max+1)

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

    def __getitem__(self, raw_index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = raw_index + min(self.index_range)

        image_path = self.base_dir / f'Images/image-{index}.png'
        image = Image.open(image_path)
        image = (np.array(image) / 255.).astype(np.float32)
        assert len(image.shape) == 2
        
        mask_path = self.base_dir / f'Masks/mask-{index}.png'
        mask = Image.open(mask_path)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].to(torch.int64)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Augmentation pipeline failed to output a tensor! Instead gave {image}")

        #Convert masks to format that works with pytorch loss
        mapping_array = [self.color_to_label_index.get(i, -1) for i in range(256)]  # Create a mapping list with default -1 for unmapped values
        mapping_tensor = torch.tensor(mapping_array)

        mask = mapping_tensor[mask]

        return image, mask

def get_data_loaders(base_dir : Path, 
                     patch_size : int,
                     augmentation_config : dict,
                     color_to_label : dict,
                     dataset_name : str,
                     train_test_split : float,
                     max_batches_per_train_epoch : Union[int, None],
                     max_batches_per_test : Union[int, None],
                     batch_size : int,
                     num_workers : Optional[int] = None,
                     verbose : bool = False) -> Tuple[DataLoader, DataLoader]:
    # Preprocess color_to_label to appropriate form
    color_to_label = {int(key) : value for key, value in color_to_label.items()}

    # Create transformation pipeline
    transform_list = []

    for item in augmentation_config['transforms']:
        transform_type = item['name']
        transform_class = getattr(A, transform_type)
        transform = transform_class(**item['params'])
        transform_list.append(transform)

    # transform_list.append(A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0))
    transform_list.append(A.RandomCrop(height=patch_size, width=patch_size))
    transform_list.append(ToTensorV2())

    transform = A.Compose(transforms=transform_list)

    # Create the dataset
    if max_batches_per_test is not None and max_batches_per_train_epoch is not None:
        max_batches = max_batches_per_train_epoch + max_batches_per_test
    else:
        max_batches = None

    dataset_dir = base_dir / dataset_name
    if verbose:
        print(f"\nTraining data path: {dataset_dir}")
    dataset = SyntheticDataset(base_dir=dataset_dir, 
                               transform=transform,
                               color_to_label=color_to_label,
                               max_batches=max_batches)

    # Define the lengths for train and test splits
    # We don't just use proportions in the lengths argument since it breaks when the sizes are small 
    # (i.e. when running tests)
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset into train and test sets
    train_set, test_set = random_split(dataset=dataset, 
                                       lengths=(train_size, test_size))


    # Create DataLoaders
    if num_workers is not None:
        if verbose:
            print(f"\nUsing num_workers={num_workers}")
    elif 'NUM_WORKERS' in os.environ:
        num_workers = int(os.environ['NUM_WORKERS'])
        if verbose:
            print(f"\nEnvironment variable NUM_WORKERS found. \nUsing num_workers={num_workers}")
    else:
        num_workers = multiprocessing.cpu_count()
        if verbose:
            print(f"\nEnvironment variable NUM_WORKERS not set. \nUsing num_workers={num_workers}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader
