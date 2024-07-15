import os
import random
from typing import Tuple, Union
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import j1
from scipy.signal import convolve2d

from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkCommonDataModel import vtkPolyData

from visualize import visualize 

def get_mt_points(file_path : Path,
                  config : dict) -> Tuple[np.ndarray, np.ndarray]:
    reader = vtkPolyDataReader()
    reader.SetFileName(str(file_path))
    reader.Update()

    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    points_array = np.empty((num_points, 3))

    for i in range(num_points):
        x, y, z = points.GetPoint(i)
        points_array[i] = [float(x), float(y), float(z)]

    mt_ids_array = polydata.GetPointData().GetArray("identite")
    mt_ids = np.array([mt_ids_array.GetValue(i) for i in range(mt_ids_array.GetNumberOfTuples())])
    
    return points_array, mt_ids

def get_fluorophore_points(mt_points : np.ndarray,
                           config : dict) -> np.ndarray:
    fluorophore_concentration : float = config['fluorophore_concentration']
    tubulaton_unit_in_meters: float = config['tubulaton_unit_in_meters']
    num_fluorophores : int = config['num_fluorophores']
    fluorophore_displacement_std : float = config['fluorophore_displacement_std']

    #How many fluorophores are there?
    # TODO - actually use this!!!
    x_dim = mt_points[:, 0].max() - mt_points[:, 0].min()
    y_dim = mt_points[:, 1].max() - mt_points[:, 1].min()
    z_dim = mt_points[:, 2].max() - mt_points[:, 2].min()

    volume = x_dim * y_dim * z_dim * (tubulaton_unit_in_meters ** 3)
    fluorophores_in_moles = volume * (fluorophore_concentration * (10**3))

    # TODO - we haven't actually used calculated number of FPs (too many, not sure how we should simulate this properly)
    # Just used a preset config value
    num_fluorophores = int(config['num_fluorophores'])

    # TODO - can more than one YFP bind to a MT?
    # Sample or choice? (currently using choice)

    # Using sample
    if num_fluorophores >= mt_points.shape[0]:
        fluorophore_indices = list(range(mt_points.shape[0]))
    else:
        fluorophore_indices = random.sample(range(mt_points.shape[0]), k=num_fluorophores)

    # Using choice
    # fluorophore_indices = random.choices(range(mt_points.shape[0]), k=num_fluorophores)

    fluorophore_points = mt_points[fluorophore_indices]

    displacement = np.random.normal(loc=0., 
                                    scale=fluorophore_displacement_std, 
                                    size=fluorophore_points.shape)
    fluorophore_points += displacement

    return fluorophore_points

def get_intensities(fluorophore_points : np.ndarray,
                    z_slice : float,
                    config : dict) -> np.ndarray:
    z_slice_radius_in_tubulaton = config['z_slice_range'] / config['tubulaton_unit_in_meters'] / 2

    x_min = fluorophore_points[:, 0].min()
    x_max = fluorophore_points[:, 0].max()
    y_min = fluorophore_points[:, 1].min()
    y_max = fluorophore_points[:, 1].max()

    original_image = np.zeros((int(y_max - y_min) + 10, int(x_max - x_min) + 10))
    
    hits = 0
    for fluorophore_point in fluorophore_points:
        x, y, z = fluorophore_point
        if abs(z - z_slice) < z_slice_radius_in_tubulaton:
            hits += 1
            new_x = int(x - x_min)
            new_y = int(y - y_min)
            
            original_image[new_y:new_y+3, new_x:new_x+3] = 1.
    print(f"{hits} Hits")

    return original_image

def apply_psf(original_image : np.ndarray,
              config : dict) -> np.ndarray:
    psf_kernel_size = config['psf_kernel_size']
    tubulaton_unit_in_meters: float = config['tubulaton_unit_in_meters']
    numerical_aperature = config['numerical_aperature']
    emission_wavelength = config['emission_wavelength']
    peak_intensity = config['peak_intensity']

    # TODO - at current size the kernel doesn't do anything
    # Either remove it, or perform more costly convolution
    psf_kernel = np.empty((psf_kernel_size, psf_kernel_size))
    for i in range(psf_kernel_size):
        for j in range(psf_kernel_size):
            y = ((i + 0.5) - psf_kernel_size / 2) * tubulaton_unit_in_meters
            x = ((j + 0.5) - psf_kernel_size / 2) * tubulaton_unit_in_meters

            r = np.sqrt(x**2 + y**2)
            rho = 2 * np.pi * numerical_aperature * r / emission_wavelength

            if 2*i+1 == psf_kernel_size and 2*j+1 == psf_kernel_size:
                psf_value = peak_intensity
            else:
                psf_value = peak_intensity * (2 * j1(rho) / rho) ** 2

            psf_kernel[i,j] = psf_value
    
    blurred_image = convolve2d(original_image, psf_kernel, mode='same')

    return blurred_image

def apply_shot_noise(image : np.ndarray,
                     config : dict) -> np.ndarray:
    dark_electron_emission = config['dark_electron_emission']
    electron_emission_multiplier= config['electron_emission_multiplier']
    digital_gain = config['digital_gain']

    means = electron_emission_multiplier * image + dark_electron_emission

    intensity = np.random.poisson(lam=means) * digital_gain

    return intensity 

def process_tubulaton(file_path : Path,
                      config_file_path : Path,
                      z_slice : float) -> np.ndarray:
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    mt_points, mt_ids = get_mt_points(file_path=file_path,
                                    config=config)

    fluorophore_points = get_fluorophore_points(mt_points=mt_points,
                                                config=config)

    z_min = fluorophore_points[:, 2].min()
    z_max = fluorophore_points[:, 2].max()


    image = get_intensities(fluorophore_points=fluorophore_points,
                                    # z_slice=(z_min + z_max)/2,
                                    z_slice=z_min*0.8 + z_max*0.2,
                                    config=config)

    image = apply_psf(original_image=image,
                      config=config)

    image = apply_shot_noise(image=image,
                             config=config)

    return image

if __name__ == '__main__':
    file_path : Path = Path('/Users/karan/MTData/SimulatedData_OLD/tubulaton-run/tubulaton-55197725.vtk')
    config_file_path : Path = Path('/Users/karan/Microtubules/generation/config.json')

    image = process_tubulaton(file_path=file_path,
                              config_file_path=config_file_path,
                              z_slice=800.)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


# #DEMO: get_mt_points
# # Just sample some points
# def demo_get_mt_points(file_path : Path = Path('/Users/karan/MTData/SimulatedData_OLD/tubulaton-run/tubulaton-55197725.vtk'),
#                        config_file_path : Path = Path('/Users/karan/Microtubules/generation/config.json')) -> None:
#     with open(config_file_path, 'r') as file:
#         config = json.load(file)

#     mt_points, mt_ids = get_mt_points(file_path=file_path,
#                                     config=config)

#     indices = random.sample(range(mt_points.shape[0]), k=2*10**4)
#     points = mt_points[indices]
#     mt_ids = mt_ids[indices]

#     unique_ids = np.unique(mt_ids)
#     id_to_color_map = {}
#     for id in unique_ids:
#         color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
#         id_to_color_map[id] = color
#     colors = list(map(lambda id: id_to_color_map[id], mt_ids))

#     visualize(points=points,
#             show_axes=False,
#             colors=colors)

# def demo_get_fluorophore_points(file_path : Path = Path('/Users/karan/MTData/SimulatedData_OLD/tubulaton-run/tubulaton-55197725.vtk'),
#                                 config_file_path : Path = Path('/Users/karan/Microtubules/generation/config.json')) -> None:
#     with open(config_file_path, 'r') as file:
#         config = json.load(file)

#     mt_points, mt_ids = get_mt_points(file_path=file_path,
#                                     config=config)

#     fluorophore_points = get_fluorophore_points(mt_points=mt_points,
#                                                 config=config)

#     visualize(points=fluorophore_points)

# def demo_get_intensities(file_path : Path = Path('/Users/karan/MTData/SimulatedData_OLD/tubulaton-run/tubulaton-55197725.vtk'),
#                          config_file_path : Path = Path('/Users/karan/Microtubules/generation/config.json')) -> None:
#     with open(config_file_path, 'r') as file:
#         config = json.load(file)

#     mt_points, mt_ids = get_mt_points(file_path=file_path,
#                                     config=config)

#     fluorophore_points = get_fluorophore_points(mt_points=mt_points,
#                                                 config=config)

#     z_min = fluorophore_points[:, 2].min()
#     z_max = fluorophore_points[:, 2].max()

#     print("ATTENTION:")
#     print(z_min, z_max)

#     blurred_image = get_intensities(fluorophore_points=fluorophore_points,
#                                     # z_slice=(z_min + z_max)/2,
#                                     z_slice=z_min*0.8 + z_max*0.2,
#                                     config=config)

#     plt.imshow(blurred_image, cmap='gray')
#     plt.axis('off')
#     plt.show()