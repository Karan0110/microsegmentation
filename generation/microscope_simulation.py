import numpy as np

from scipy.special import j1
from scipy.constants import Planck, speed_of_light

def get_psf_kernel(config : dict) -> np.ndarray:
    microscope_config = config['microscope']
    fluorophore_config = config['fluorophore']

    tubulaton_unit_in_meters : float = config['tubulaton_unit_in_meters']

    psf_kernel_size = microscope_config['psf_kernel_size']
    numerical_aperature = microscope_config['numerical_aperature']
    peak_intensity = microscope_config['peak_intensity']

    emission_wavelength = fluorophore_config['emission_wavelength']

    # TODO - Are we using the right numerical aperature? (we are currently using NA of objective lens)
    psf_kernel = np.empty((psf_kernel_size, psf_kernel_size))
    for i in range(psf_kernel_size):
        for j in range(psf_kernel_size):
            if 2*i+1 == psf_kernel_size and 2*j+1 == psf_kernel_size:
                psf_value = peak_intensity
            else:
                y = ((i + 0.5) - psf_kernel_size / 2) * tubulaton_unit_in_meters
                x = ((j + 0.5) - psf_kernel_size / 2) * tubulaton_unit_in_meters

                r = np.sqrt(x**2 + y**2)
                rho = 2 * np.pi * numerical_aperature * r / emission_wavelength

                psf_value = peak_intensity * (2 * j1(rho) / rho) ** 2

            psf_kernel[i,j] = psf_value
        
    return psf_kernel

def get_digital_signal(intensities : np.ndarray,
                       config : dict) -> np.ndarray:
    microscope_config = config['microscope']

    dark_count_rate = microscope_config['dark_count_rate']
    quantum_efficiency = microscope_config['quantum_efficiency']

    digital_gain = microscope_config['digital_gain']

    # i.e. exposure time on HyD for a single pixel
    pixel_dwell_time = microscope_config['pixel_dwell_time']

    means = (quantum_efficiency * intensities + dark_count_rate) * pixel_dwell_time

    image = np.random.poisson(lam=means) * digital_gain 

    return image 

def quantize_intensities(image : np.ndarray,
                         config : dict) -> np.ndarray:
    bit_depth = config['microscope']['bit_depth']
    max_intensity = 2 ** bit_depth - 1

    normalized_image = (image - image.min()) / (image.max() - image.min()) * max_intensity
    quantized_image = np.round(normalized_image).astype(int)

    return quantized_image
