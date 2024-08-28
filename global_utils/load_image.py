from pathlib import Path

from PIL import Image

import numpy as np

def load_grayscale_image(image_file_path : Path) -> np.ndarray:
    image = np.array(Image.open(image_file_path).convert('L'))
    image = image.astype(np.float32) / 255.

    return image
