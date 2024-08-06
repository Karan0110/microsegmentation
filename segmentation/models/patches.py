from typing import Tuple

import numpy as np

from torch import Tensor
import torch.nn.functional as F

#Returns tensor of shape (num_patches, channels, height, width)
#The patches are ordered lexicographically, e.g.
# 0 1 2
# 3 4 5
def get_image_patches(image : Tensor, 
                      patch_size : Tuple[int,int],
                      padding_mode : str = 'zeros') -> Tensor:
    # batch, channels, height, width
    b, c, h, w = image.shape
    patch_h, patch_w = patch_size
    
    # Apply padding
    pad_h = (patch_h - (h % patch_h)) % patch_h
    pad_w = (patch_w - (w % patch_w)) % patch_w

    padded_image = None
    if pad_h > 0 or pad_w > 0:
        if padding_mode == 'zeros':
            padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode=padding_mode)
    else:
        padded_image = image

    # Unfold to extract patches
    patches = padded_image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    
    # Reshape tensor
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, c, patch_h, patch_w)
    
    return patches

def stitch_image_patches(patches: Tensor, 
                         original_shape: Tuple[int, int, int, int], 
                         patch_size: int) -> Tensor:
    b, c, h, w = original_shape
    patches_per_col = (h + patch_size-1) // patch_size
    patches_per_row = (w + patch_size-1) // patch_size

    padded_h = patch_size * patches_per_col
    padded_w = patch_size * patches_per_row
    
    # Reshape patches to the original dimensions
    patches = patches.view(b, patches_per_col, patches_per_row, c, patch_size, patch_size)
    
    # Permute to match the original image dimensions
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    
    # Combine patches
    image = patches.view(b, c, padded_h, padded_w)
    
    return image