from pathlib import Path
from typing import Tuple, Union, List, Optional
import argparse

import numpy as np
import matplotlib.pyplot as plt

import tifffile

from .models.inference import get_segmentation, get_hard_segmentation
from .utils.serialization import load_model_from_file
from .utils import get_device

from global_utils import load_json5

def get_otsu_threshold(image: np.ndarray, 
                       num_bins: int = 30) -> float:
    # Calculate histogram with the desired number of bins
    hist, bin_edges = np.histogram(image, bins=num_bins)
    
    # Normalize the histogram
    hist = hist / hist.sum()
    
    # Compute cumulative sum and cumulative mean
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(num_bins))
    
    # Compute global mean
    global_mean = cumulative_mean[-1]
    
    # Compute between-class variance for each threshold
    between_class_variance = ((global_mean * cumulative_sum - cumulative_mean) ** 2) / (cumulative_sum * (1 - cumulative_sum))
    
    # Find the threshold that maximizes the between-class variance
    optimal_threshold_index = np.nanargmax(between_class_variance)
    optimal_threshold = bin_edges[optimal_threshold_index]
    
    return optimal_threshold

def load_time_series(time_series_file_path : Path,
                     axis_format : str = 'tzcyx',
                     channel : int = 0,
                     z_slice : int = 0) -> np.ndarray:
    with tifffile.TiffFile(time_series_file_path) as time_series_file:
        time_series = time_series_file.asarray()

    # Convert image to tzcyx format - inserting z and channel dimensions if needed
    time_axis = axis_format.find('t')
    z_axis = axis_format.find('z')
    channel_axis = axis_format.find('c')
    y_axis = axis_format.find('y')
    x_axis = axis_format.find('x')

    if z_axis == -1:
        z_axis = len(time_series.shape)
        time_series = time_series[..., np.newaxis]
    if channel_axis == -1:
        channel_axis = len(time_series.shape)
        time_series = time_series[..., np.newaxis]

    time_series = np.transpose(time_series, axes=(time_axis, z_axis, channel_axis, y_axis, x_axis))

    time_series = time_series[:, z_slice, channel, :, :]

    return time_series

def get_colored_image(image : np.ndarray,
                      color : Tuple[float, float, float] = (1., 0., 0.)) -> np.ndarray:
    np_color = np.array(color)

    colored_image = np.empty((*image.shape, 4))
    colored_image[:, :, :3] = np_color
    colored_image[:, :, 3] = image

    return colored_image

# Time series : (T, H, W)
def plot_time_series(time_series : np.ndarray,
                     axs : np.ndarray,
                     column : int,
                     title : Optional[str] = None,
                     vmax : int = 256) -> None:
    if title is not None:
        axs[0,column].set_title(title)

    for time in range(axs.shape[0]):
        image = time_series[time]
        axs[time, column].imshow(image, cmap='gray', vmin=0, vmax=vmax)
        axs[time, column].axis('off')

def get_command_line_arguments() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Segment MTs in a time series"
    )

    # Positional argument (mandatory)
    parser.add_argument('-ts', '--timeseries', 
                        type=Path, 
                        required=True,
                        help='Time Series File Path (.lsm)')

    parser.add_argument('-z', '--zslice', 
                        type=int, 
                        required=True,
                        help='z-slice for time series')

    parser.add_argument('-mf', '--model',
                        type=Path,
                        required=True,
                        help="Model Dir Path")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    parser.add_argument('-sf', '--savefile',
                        type=str,
                        default=None,
                        help="Savefile path (Leave blank to use model parent directory")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_command_line_arguments()

    verbose = args.verbose

    # Load model
    model_dir = args.model
    device = get_device(verbose=verbose)

    model, config = load_model_from_file(model_dir=model_dir,
                                         device=device)

    patch_size = config['model']['patch_size']

    # Load time series
    time_series_file_path = args.timeseries
    z_slice = args.zslice
    
    time_series = load_time_series(time_series_file_path=time_series_file_path,
                                   z_slice=z_slice)
    time_steps = time_series.shape[0]

    hard_segmentations = np.empty_like(time_series)
    threshold = None

    for time in range(time_steps):
        input_image = time_series[time].astype(np.float32) / 255.
        segmentation = get_segmentation(image=input_image,
                                        model=model,
                                        device=device,
                                        patch_size=patch_size,
                                        batch_size=1).to('cpu').numpy()

        if threshold is None:
            # threshold = get_otsu_threshold(segmentation)
            threshold = np.quantile(segmentation.flatten(), 0.97)

            # plt.hist(segmentation.flatten(), bins=30)
            # plt.show()

            print(f"Threshold : {threshold}")

        hard_segmentation = get_hard_segmentation(segmentation=segmentation,
                                                  segmentation_threshold=threshold)
        hard_segmentation = (hard_segmentation * 255.).astype(int)

        hard_segmentations[time, :, :] = hard_segmentation
    
    nrows = time_steps
    ncols = 2

    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.reshape((nrows, ncols))

    plot_time_series(time_series=time_series,
                     axs=axs,
                     column=0,
                     vmax=int(np.percentile(time_series.flatten(), 99.8)),
                     title='Experimental Data')

    plot_time_series(time_series=hard_segmentations,
                     axs=axs,
                     column=1,
                     vmax=256,
                     title='Hard Segmentation')

    plt.show()
        