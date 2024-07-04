from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from resnet import ResNet

def load_model(model_file_path : Path, device) -> Tuple[nn.Module, dict]:
    model_data : dict = torch.load(model_file_path)
    model_params : dict = model_data['model_params']

    patch_size = model_params['patch_size']
    layers = model_params['layers']
    in_channels = model_params['in_channels']
    num_classes = model_params['num_classes']

    model = ResNet(layers=layers, in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(model_data['state_dict'])

    model.eval()  

    return model, model_params
