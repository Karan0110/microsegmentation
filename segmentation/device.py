import torch

def get_device(verbose : bool) -> torch.device:
    # General device handling: Check for CUDA/GPU, else fallback to CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Silicon GPU'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    if verbose:
        print(f"Using device: {device_name}")

    return device
