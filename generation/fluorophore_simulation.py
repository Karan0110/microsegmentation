import random
from pathlib import Path
from typing import Tuple

import numpy as np

from scipy.constants import Planck, speed_of_light

from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkCommonDataModel import vtkPolyData

def rotate_x(points: np.ndarray, angle: float) -> np.ndarray:
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])
    
    # Apply the rotation matrix to the points
    rotated_points = points @ rotation_matrix.T
    return rotated_points

def get_mt_points(file_path : Path,
                  config : dict,
                  verbose : bool) -> Tuple[np.ndarray, np.ndarray]:
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

    if mt_ids_array is None:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)

    mt_ids = np.array([mt_ids_array.GetValue(i) for i in range(mt_ids_array.GetNumberOfTuples())])
    
    if config['apply_random_rotation']:
        # theta = np.pi
        # print("WARNING: Using non-random theta value for rotation!")
        theta = np.random.random() * (2*np.pi)
        if verbose:
            print(f"Rotating .vtk file by {theta} radians")
        points_array = rotate_x(points_array, theta)

    # Discard all NaN points
    nan_mask = np.isnan(points_array).any(axis=1)
    points_array = points_array[~nan_mask]
    mt_ids = mt_ids[~nan_mask]
    
    return points_array, mt_ids

def calculate_fluorophore_emission_per_second(config : dict) -> float:
    microscope_config = config['microscope']
    fluorophore_config = config['fluorophore']

    laser_power = microscope_config['laser_max_power'] * microscope_config['laser_proportional_power']
    laser_spot_area = microscope_config['laser_spot_area']
    absorption_cross_section = fluorophore_config['absorption_cross_section']
    quantum_yield = fluorophore_config['quantum_yield']
    excitation_wavelength = fluorophore_config['excitation_wavelength']

    laser_intensity = laser_power / laser_spot_area
    photon_energy = Planck * speed_of_light / excitation_wavelength

    intensity = quantum_yield * absorption_cross_section * laser_intensity / photon_energy

    return intensity

def get_fluorophore_points(tubulin_points : np.ndarray,
                           config : dict) -> np.ndarray:
    tubulaton_unit_in_meters: float = config['tubulaton_unit_in_meters']
    # num_fluorophores : int = config['num_fluorophores']

    fluorophore_config = config['fluorophore']
    fluorophore_displacement_std : float = fluorophore_config['displacement_std']
    fluorophore_concentration : float = fluorophore_config['concentration']

    #How many fluorophores are there?
    x_dim = tubulin_points[:, 0].max() - tubulin_points[:, 0].min()
    y_dim = tubulin_points[:, 1].max() - tubulin_points[:, 1].min()
    z_dim = tubulin_points[:, 2].max() - tubulin_points[:, 2].min()

    volume = x_dim * y_dim * z_dim * (tubulaton_unit_in_meters ** 3)
    fluorophores_in_moles = volume * (fluorophore_concentration * (10**3))

    fluorophore_batch_size = config['fluorophore_batch_size']
    num_fluorophores = int(fluorophores_in_moles * 6.0e23 / fluorophore_batch_size)

    # TODO - ask biologist if this is realistic

    # TODO - can more than one YFP bind to a MT?
    # Sample or choice? (currently using choice)

    # Using sample
    if num_fluorophores >= tubulin_points.shape[0]:
        fluorophore_indices = list(range(tubulin_points.shape[0]))
    else:
        fluorophore_indices = random.sample(range(tubulin_points.shape[0]), k=num_fluorophores)

    # Using choice
    # fluorophore_indices = random.choices(range(mt_points.shape[0]), k=num_fluorophores)

    fluorophore_points = tubulin_points[fluorophore_indices].copy()
    displacement = np.random.normal(loc=0., 
                                    scale=fluorophore_displacement_std/np.sqrt(3), 
                                    size=fluorophore_points.shape)
    fluorophore_points += displacement

    return fluorophore_points

def get_fluorophore_image(fluorophore_points : np.ndarray,
                          z_slice : float,
                          config : dict,
                          bounding_box : Tuple[int, int, int, int],
                          verbose : bool = True) -> np.ndarray:

    # TODO - we destroy z-axis information here, without accounting for the different PSF

    z_slice_radius_in_tubulaton = config['microscope']['z_slice_range'] / config['tubulaton_unit_in_meters'] / 2
    fluorophore_batch_size = config['fluorophore_batch_size']
    fluorophore_batch_spread = config['fluorophore_batch_spread']

    x_min, y_min, x_max, y_max = bounding_box

    image = np.zeros((int(y_max - y_min) + 1, int(x_max - x_min) + 1))
    
    hits = 0
    for fluorophore_point in fluorophore_points:
        x, y, z = fluorophore_point
        if abs(z - z_slice) < z_slice_radius_in_tubulaton:
            hits += 1
            new_x = int(x - x_min)
            new_y = int(y - y_min)
            
            temp_y = min(new_y+fluorophore_batch_spread, int(y_max - y_min))
            temp_x = min(new_x+fluorophore_batch_spread, int(x_max - x_min))
            image[new_y:temp_y, new_x:temp_x] = float(fluorophore_batch_size) / (fluorophore_batch_spread ** 2)
    # if verbose:
    #     print(f"{hits} fluorophores found in the focal plane.")

    return image
