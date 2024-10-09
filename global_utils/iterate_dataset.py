from pathlib import Path
from typing import Generator, Tuple

def iterate_image_mask_pairs(data_dir: Path) -> Generator[Tuple[Path, Path], None, None]:
    images_dir = data_dir / 'Images'
    masks_dir = data_dir / 'Masks'
    
    # Ensure both 'Images' and 'Masks' directories exist
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Both 'Images' and 'Masks' directories must exist in the dataset folder.")

    # Iterate over image files in the 'Images' folder
    for image_file in images_dir.iterdir():
        if image_file.is_file():
            # Find the corresponding mask file in the 'Masks' folder
            mask_file = masks_dir / image_file.name

            # Ensure the mask file exists
            if mask_file.exists() and mask_file.is_file():
                # Yield the image and mask file paths as a pair
                yield image_file, mask_file
            else:
                raise FileNotFoundError(f"No corresponding mask found for: {image_file}")
