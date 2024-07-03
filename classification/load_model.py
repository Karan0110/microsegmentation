from pathlib import Path

import torch
import torch.nn as nn

from resnet import ResNet

def load_model(model_file_path : Path, device) -> nn.Module:
    model_data = torch.load(model_file_path)

    layers = model_data['layers']
    in_channels = model_data['in_channels']
    num_classes = model_data['num_classes']

    model = ResNet(layers=layers, in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(model_data['state_dict'])
    model.eval()  # Set the model to evaluation mode

    return model
