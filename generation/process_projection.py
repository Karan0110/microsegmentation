import os
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter

RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2

def pad_image(image : np.array, 
              padding : int) -> np.array:

    height, width = image.shape
    padded_image = np.zeros((height + 2 * padding, width + 2 * padding))

    padded_image[padding:height+padding, padding:width+padding] = image

    return padded_image

#Rescale image intensities to use upper end more (lossy)
def rescale_image_intensities(image : np.array, 
                  percentile : float) -> np.array:
    image = image.astype(np.float32)

    perc_value = np.percentile(image, percentile)

    rescaled_image = np.clip(image / perc_value, 0, 1)
    
    return rescaled_image

#Discretize array by breaking into a grid of quant_size x quant_size and assiging to each tile the average value
def discretize_array(array : np.array, 
                     quant_size : int) -> np.array:
    h, w = array.shape
    
    h_tiles = h // quant_size
    w_tiles = w // quant_size
    
    discretized_array = np.zeros_like(array)
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            tile = array[i*quant_size:(i+1)*quant_size, j*quant_size:(j+1)*quant_size]
            
            avg_value = np.mean(tile)
            
            discretized_array[i*quant_size:(i+1)*quant_size, j*quant_size:(j+1)*quant_size] = avg_value
    
    return discretized_array

#Gets bounding box of MT network
def get_bounding_box(image : np.array) -> Tuple[int, int, int, int]:
    non_zero_indices = np.argwhere(image)
    min_i, min_j = non_zero_indices.min(axis=0)
    max_i, max_j = non_zero_indices.max(axis=0)

    return (min_i, min_j, max_i, max_j)

#Simulate depolymerisation (run this before adding noise)
def simulate_depoly(image : np.array, 
                    blackout_intensity : int = 0.05,
                    blackout_size: int = 4,) -> np.array:
    depoly_image = image.copy()
    
    height, width = image.shape

    rate = blackout_intensity * width * height
    num_blackouts = np.random.poisson(lam=rate)

    for _ in range(num_blackouts):
        i,j = (np.random.randint(height) - blackout_size + 1, np.random.randint(width) - blackout_size + 1)
        depoly_image[i:i+blackout_size, j:j+blackout_size] = 0.

    return depoly_image
    
# Add noise to an image (must already be cropped) 
def add_noise(image : np.array, 
              min_i : int, 
              min_j : int, 
              max_i : int, 
              max_j : int, 
              apply_gaussian_blur : bool = True, 
              grain_size : int = 6, 
              noise_level : float = 0.05, 
              exp_scale : float = 0.85, 
              intensity_quantile : float = .98) -> np.array:

    noised_image = image.copy()
    height, width = noised_image.shape

    if not (0 <= min_i <= max_i <= height and 0 <= min_j <= max_j <= width):
        raise ValueError(f"Invalid min/max i,j choices: \nmin_i = {min_i}, max_i = {max_i}, min_j = {min_j}, max_j = {max_j}")
    
    #(Optional) Gaussian blur
    if apply_gaussian_blur:
        noised_image = cv2.GaussianBlur(noised_image, ksize=(7,7), sigmaX=0)

    #Quantise image
    noised_image = discretize_array(noised_image, quant_size=grain_size)

    #Add grains
    
    grains = (np.tile(np.random.random((height // grain_size + 1, width // grain_size + 1)), reps=(grain_size,grain_size)) < noise_level).astype(float)
    grains = grains[:height, :width]

    mask = np.zeros((height, width))
    mask[min_i:max_i, min_j:max_j] = np.random.exponential(scale=exp_scale, size=mask[min_i:max_i, min_j:max_j].shape)

    # Spreads out the mask a little
    mask = gaussian_filter(mask, sigma=height/6)
    mask = discretize_array(mask, quant_size=grain_size)
    
    noised_image += grains * mask

    #Maximise contrast
    noised_image = rescale_image_intensities(noised_image, 100 * intensity_quantile)
    
    return noised_image

def process_projection(control_image : np.array, 
                       padding : int = 50,
                       verbose : bool = False) -> Tuple[np.array, np.array]:
    if verbose:
        print("Processing image...")

    height, width = control_image.shape    

    # Add padding to image
    min_i, min_j, max_i, max_j = padding, padding, height + padding, width + padding
    control_image = pad_image(control_image, padding=padding)

    #Get depoly image
    depoly_image = simulate_depoly(control_image)

    #Get noised outputs
    noised_control_image = add_noise(control_image, min_i=min_i, min_j=min_j, max_i=max_i, max_j=max_j)
    noised_depoly_image = add_noise(depoly_image, min_i=min_i, min_j=min_j, max_i=max_i, max_j=max_j)

    return (noised_control_image, noised_depoly_image)
