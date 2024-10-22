from pathlib import Path
import os
import numpy as np
from typing import Tuple, Union, List, Optional, Dict, Any, Callable
from dotenv import load_dotenv
import argparse
from random import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from skimage.filters import threshold_otsu

from scipy.signal import argrelextrema

from segmentation.utils import get_device
from global_utils import load_json5

from segmentation.models.inference import query_inference
from segmentation.utils.serialization import load_model

from global_utils.arguments import get_path_argument
from global_utils import load_grayscale_image
from global_utils.iterate_dataset import iterate_image_mask_pairs

def calculate_metrics(probs_flat: np.ndarray, targets_flat: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics for model predictions.

    Args:
        probs (np.ndarray): The predicted probabilities (soft outputs) from the model.
        targets (np.ndarray): The ground truth binary labels (0 or 1).

    Returns:
        Dict[str, Any]: A dictionary of calculated metrics and data for further processing.
    """

    # Initialize metrics dictionary
    metrics: Dict[str, Any] = {}

    # 2. Brier Score (same formula as MSE for probabilities)
    brier_score: float = float(np.mean((probs_flat - targets_flat)**2))
    metrics['brier'] = brier_score

    # 3. Precision-Recall Curve and AUC-PR
    precision, recall, _ = precision_recall_curve(targets_flat, probs_flat)
    auc_pr: float = float(auc(recall, precision))
    metrics['auc_pr'] = auc_pr
    
    # 4. ROC-AUC
    roc_auc: float = float(roc_auc_score(targets_flat, probs_flat))
    metrics['auc_roc'] = roc_auc

    # 5. Soft Dice Coefficient
    intersection: float = float(np.sum(probs_flat * targets_flat))
    dice_soft: float = float((2. * intersection) / (np.sum(probs_flat) + np.sum(targets_flat) + 1e-6))
    metrics['soft_dice'] = dice_soft

    # 6. Hard Dice Coefficient (for some special threshold choices)
    local_minima_values = calculate_local_minima(probs_flat)
    thresholds = {
        "otsu": threshold_otsu(probs_flat),
        "llm": local_minima_values.min() if local_minima_values.shape[0] > 0 else None,
        "hlm": local_minima_values.max() if local_minima_values.shape[0] > 0 else None,
    }

    for threshold_name in thresholds:
        threshold = thresholds[threshold_name]

        if threshold is not None:
            dice_score = get_hard_dice_score(probs_flat, targets_flat, threshold)
        else:
            dice_score = np.nan

        metrics[f"dice_{threshold_name}"] = dice_score

    # 7. BCE (with 1:1 and 9:1 weightings)
    weights = [0.5, 0.9] 
    for alpha in weights:
        label = f"Weighted BCE ({int(1/(1-alpha))} : 1)"
        bce = -(alpha * mask * np.log(segmentation) + (1. - alpha) * (1.-mask) * np.log(1.-segmentation)).mean()

        metrics[label] = bce

    return metrics

def get_hard_dice_score(probs_flat : np.ndarray, targets_flat : np.ndarray, threshold : float) -> float:
    binary_preds: np.ndarray = (probs_flat > threshold).astype(np.float32)
    intersection: float = float(np.sum(binary_preds * targets_flat))
    dice_hard: float = float((2. * intersection) / (np.sum(binary_preds) + np.sum(targets_flat) + 1e-7))
    
    return dice_hard

def calculate_local_minima(probs_flat : np.ndarray, nbins : int = 25) -> np.ndarray:
    hist, bin_edges= np.histogram(probs_flat, bins=nbins)
    minima_indices = argrelextrema(hist, np.less)[0]
    local_minima_values = bin_edges[minima_indices]

    return local_minima_values

def update_metrics_df(metrics_df : pd.DataFrame, 
                      image_name : str,
                      mask : np.ndarray,
                      segmentation : np.ndarray,
                      model_name : str,
                      dataset_name : str,
                      verbose : bool) -> pd.DataFrame:
    metrics = calculate_metrics(segmentation.flatten(), mask.flatten())

    if ('model_name' in metrics_df) and ('dataset_name' in metrics_df) and ('image_name' in metrics_df) and \
        (df_mask := ((metrics_df['model_name'] == model_name) & (metrics_df['dataset_name'] == dataset_name) & (metrics_df['image_name'] == image_name))).any():

        for col, value in metrics.items():
            metrics_df.loc[df_mask, col] =  value
    else:
        expanded_metrics : Dict = metrics.copy()
        expanded_metrics['model_name'] = model_name
        expanded_metrics['dataset_name'] = dataset_name
        expanded_metrics['image_name'] = image_name

        metrics_df = pd.concat([metrics_df, pd.DataFrame([expanded_metrics])], ignore_index=True)
    
    return metrics_df

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net model."
    )

    parser.add_argument('-sd', '--savedir',
                        type=str,
                        help="Path to inference save files (for caching)")

    parser.add_argument('-i', '--input', 
                        type=str, 
                        required=True,
                        help='Dataset Name')
    
    parser.add_argument('-md', '--modeldir', 
                        type=Path, 
                        help='Models Path (Leave blank to use environment variable value)')

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument('-c', '--count',
                        type=int,
                        default=0,
                        help="Max number of samples to use (Leave blank to use everything in directory)")

    parser.add_argument('-ow', '--overwrite', 
                        action='store_true', 
                        help='Overwrite inferences, even when inferences saved to file')

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    dotenv_path = Path(os.environ['PYTHONPATH']) / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    verbose = args.verbose
    overwrite_save_file = args.overwrite

    max_num_samples = args.count

    device = get_device(verbose=verbose)

    models_path = get_path_argument(cl_args=args,
                                    cl_arg_name='modeldir',
                                    env_var_name='MODELS_PATH')

    save_dir = get_path_argument(cl_args=args,
                                    cl_arg_name='savedir',
                                    env_var_name='INFERENCE_SAVE_DIR')

    data_dir = Path(os.environ['DATA_DIR']) / args.input
    if not data_dir.is_absolute():
        data_dir = Path(os.environ['PYTHONPATH']) / data_dir

    model_name = args.name

    model_dir = models_path / model_name
    
    if verbose:
        print(f"\nModel directory: {model_dir}")

    config = load_json5(model_dir / 'config.json5')
    model_file_path = model_dir / "model.pth"

    patch_size = config['model']['patch_size']

    model = load_model(device=device,
                        config=config,
                        model_dir=model_dir,
                        verbose=verbose)

    if verbose:
        print("Calculating metrics...")
    metrics_df_file_path = Path(os.environ['PYTHONPATH']) / 'evaluation' / "metrics.csv"

    if metrics_df_file_path.exists():
        metrics_df = pd.read_csv(metrics_df_file_path)
    else:
        metrics_df = pd.DataFrame()

    if verbose:
        print(f"Data Dir: {data_dir}")
    image_mask_pairs = sorted(list(iterate_image_mask_pairs(data_dir)), key=lambda pair: pair[0].stem)

    for image_file_path, mask_file_path in image_mask_pairs:
        segmentation = query_inference(model=model,
                                       device=device,
                                       image_file_path=image_file_path,
                                       model_file_path=model_file_path,
                                       patch_size=patch_size,
                                       save_dir=save_dir,
                                       overwrite_save_file=overwrite_save_file,
                                       verbose=verbose)
        mask = load_grayscale_image(mask_file_path)

        metrics_df = update_metrics_df(metrics_df,
                                        image_name=image_file_path.stem,
                                        mask=mask,
                                        segmentation=segmentation,
                                        model_name=model_name,
                                        dataset_name=args.input,
                                        verbose=verbose)

    if verbose:
        print(f"Writing metrics to metrics.csv file...")
    metrics_df.to_csv("metrics.csv", index=False)



