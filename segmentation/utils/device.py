import torch

def get_device(verbose : bool) -> torch.device:
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
        print(f"\nUsing device: {device_name} ({device})")

    return device
