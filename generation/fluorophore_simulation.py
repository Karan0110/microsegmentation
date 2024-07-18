import random
from pathlib import Path
from typing import Tuple

import numpy as np

from scipy.constants import Planck, speed_of_light

from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkCommonDataModel import vtkPolyData

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

    # TODO - this uses the "identite" attribute, which will be deprecated soon
    # This could cause errors if we run tubulaton on a fixed version
    mt_ids_array = polydata.GetPointData().GetArray("identite")
    mt_ids = np.array([mt_ids_array.GetValue(i) for i in range(mt_ids_array.GetNumberOfTuples())])
    
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

def get_fluorophore_points(mt_points : np.ndarray,
                           config : dict) -> np.ndarray:
    tubulaton_unit_in_meters: float = config['tubulaton_unit_in_meters']
    # num_fluorophores : int = config['num_fluorophores']

    fluorophore_config = config['fluorophore']
    fluorophore_displacement_std : float = fluorophore_config['displacement_std']
    fluorophore_concentration : float = fluorophore_config['concentration']

    #How many fluorophores are there?
    # TODO - actually use this!!!
    x_dim = mt_points[:, 0].max() - mt_points[:, 0].min()
    y_dim = mt_points[:, 1].max() - mt_points[:, 1].min()
    z_dim = mt_points[:, 2].max() - mt_points[:, 2].min()

    volume = x_dim * y_dim * z_dim * (tubulaton_unit_in_meters ** 3)
    fluorophores_in_moles = volume * (fluorophore_concentration * (10**3))
    print(f"Theoretical number of FPs: {fluorophores_in_moles * 6.0e23}")

    # TODO - we haven't actually used calculated number of FPs (It could be tractable though - I initially thought it was on order 10^20, actually more like 10^8)
    # Just used a preset config value

    # num_fluorophores = int(config['num_fluorophores'])
    fluorophore_batch_size = config['fluorophore_batch_size']
    num_fluorophores = int(fluorophores_in_moles * 6.0e23 / fluorophore_batch_size)

    # TODO - ask biologist if this is realistic

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
                                    scale=fluorophore_displacement_std/np.sqrt(3), 
                                    size=fluorophore_points.shape)
    fluorophore_points += displacement

    return fluorophore_points

def get_fluorophore_image(fluorophore_points : np.ndarray,
                          z_slice : float,
                          config : dict,
                          verbose : bool = True) -> np.ndarray:

    # TODO - we destroy z-axis information here, without accounting for the different PSF

    z_slice_radius_in_tubulaton = config['microscope']['z_slice_range'] / config['tubulaton_unit_in_meters'] / 2
    fluorophore_batch_size = config['fluorophore_batch_size']
    fluorophore_batch_spread = config['fluorophore_batch_spread']

    x_min = fluorophore_points[:, 0].min()
    x_max = fluorophore_points[:, 0].max()
    y_min = fluorophore_points[:, 1].min()
    y_max = fluorophore_points[:, 1].max()

    original_image = np.zeros((int(y_max - y_min) + 1, int(x_max - x_min) + 1))
    
    hits = 0
    for fluorophore_point in fluorophore_points:
        x, y, z = fluorophore_point
        if abs(z - z_slice) < z_slice_radius_in_tubulaton:
            hits += 1
            new_x = int(x - x_min)
            new_y = int(y - y_min)
            
            original_image[new_y:new_y+fluorophore_batch_spread, new_x:new_x+fluorophore_batch_spread] = float(fluorophore_batch_size) / (fluorophore_batch_spread ** 2)
    if verbose:
        print(f"{hits} fluorophores found in the focal plane.")

    return original_image
