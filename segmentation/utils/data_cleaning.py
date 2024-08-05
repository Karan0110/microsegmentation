# This shouldn't be a thing. Just a stop-gap until the big change on generate.py

from pathlib import Path
from typing import Tuple, List
import sys
import os
import re
import shutil

from PIL import Image

def get_image_label_paths(data_dir : Path,
                          file_id : int) -> Tuple[Path, Path]:
    image_file_path = Path(data_dir / "Images" / f"image-{file_id}.png")
    label_file_path = Path(data_dir / "Labels" / f"label-{file_id}.png")

    return image_file_path, label_file_path

def _get_file_ids(data_dir : Path,
                  file_name_stem : str) -> List[int]:
    file_names = os.listdir(data_dir)

    file_ids = []
    for file_name in file_names:
        if not file_name.endswith('.png'):
            continue

        match = re.search(file_name_stem + r'-(\d+)\.png', file_name)
        if match is None:
            raise ValueError(f"Found a .png file in {data_dir} named {file_name}. It should be of the form {file_name_stem}-[number].png")
        else:
            file_id = int(match.group(1))

            expected_file_name = f"{file_name_stem}-{file_id}.png"
            if expected_file_name != file_name:
                raise ValueError(f"Found a .png file in {data_dir} named {file_name}. It should be of the form {file_name_stem}-[number].png")
            
            file_ids.append(file_id)
    
    return sorted(file_ids)

def get_file_ids(data_dir : Path) -> List[int]:
    image_file_ids = _get_file_ids(data_dir=data_dir / "Images/",
                                   file_name_stem='image')
    label_file_ids = _get_file_ids(data_dir=data_dir / "Labels/",
                                   file_name_stem='label')

    if tuple(image_file_ids) != tuple(label_file_ids):
        raise FileNotFoundError(f"Image and Label files are misaligned!")

    return image_file_ids

def get_image_dimensions(data_dir : Path,
                         file_id : int) -> Tuple[int, int]:
    image_path, _ = get_image_label_paths(file_id=file_id, data_dir=data_dir)
    with Image.open(image_path) as img:
        width, height = img.size
    return height, width

def save_to_clean(data_dir : Path,
                  clean_dir : Path,
                  file_ids : List[int]) -> None:

    os.makedirs(clean_dir / 'Images', exist_ok=True)
    os.makedirs(clean_dir / 'Labels', exist_ok=True)

    for new_file_id, file_id in enumerate(file_ids, start=1):
        source_image_file_path, source_label_file_path = get_image_label_paths(data_dir=data_dir,
                                                                               file_id=file_id)
        target_image_file_path, target_label_file_path = get_image_label_paths(data_dir=clean_dir,
                                                                               file_id=new_file_id)

        shutil.copy(source_image_file_path, target_image_file_path)        
        shutil.copy(source_label_file_path, target_label_file_path)        

if __name__ == '__main__':
    data_dir = Path(sys.argv[1])
    clean_dir = Path(sys.argv[2])

    file_ids = get_file_ids(data_dir=data_dir)
    new_file_ids = []

    deleted_count = 0

    for file_id in file_ids:
        height, width = get_image_dimensions(data_dir=data_dir, file_id=file_id)

        if height < 256 or width < 256:
            deleted_count += 1
        else:
            new_file_ids.append(file_id)
    
    print(f"{deleted_count} files to discard in move")

    save_to_clean(data_dir=data_dir,
                  clean_dir=clean_dir,
                  file_ids=new_file_ids)

    print(f"Cleaned data and saved to {clean_dir}")
    print(f"{len(new_file_ids)} samples remaining in cleaned data.")

# python3 data_cleaning.py /Users/karan/MTData/Synthetic /Users/karan/MTData/Synthetic_CLEAN
