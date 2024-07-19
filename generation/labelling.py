from pathlib import Path
from typing import Tuple

import numpy as np

def get_label(mt_points : np.ndarray,
              z_slice : float,
              config : dict,
              bounding_box : Tuple[int, int, int, int],
              verbose : bool = True) -> np.ndarray:
    z_slice_radius_in_tubulaton = config['microscope']['z_slice_range'] / config['tubulaton_unit_in_meters'] / 2
    mt_label_point_radius = config['mt_label_point_radius']

    x_min, y_min, x_max, y_max = bounding_box

    image = np.zeros((int(y_max - y_min) + 1, int(x_max - x_min) + 1), dtype=np.uint8)
    
    hits = 0
    for mt_point in mt_points:
        x, y, z = mt_point
        if abs(z - z_slice) < z_slice_radius_in_tubulaton:
            hits += 1
            new_x = int(x - x_min)
            new_y = int(y - y_min)
            
            box_y_min = max(new_y-mt_label_point_radius, 0)
            box_x_min = max(new_x-mt_label_point_radius, 0)

            box_y_max = min(new_y+mt_label_point_radius, int(y_max - y_min))
            box_x_max = min(new_x+mt_label_point_radius, int(x_max - x_min))

            image[box_y_min:box_y_max, box_x_min:box_x_max] = 1
    if verbose:
        print(f"{hits} tubulin found in focal plane.")

    return image
