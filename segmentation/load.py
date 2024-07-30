from pathlib import Path
from typing import Tuple, Union
import json5
import pickle

import torch
import torch.nn as nn

from unet import UNet

def load_json5_config(file_path : Path) -> dict:
    if file_path.suffix != '.json5':
        raise ValueError(f"Invalid config file path {file_path}\n" + f"Config file must be .json5")

    with file_path.open('r') as file:
        config = json5.load(file)
    if not isinstance(config, dict):
        raise TypeError(f"JSON5 config file {file_path} is of invalid format! It should be a dictionary")

    return config

def load_model(model_file_path : Path,
               model_config_file_path : Union[Path, None]) -> Tuple[nn.Module, dict]:
    model_data : dict = torch.load(model_file_path)

    # This isn't really an option - only included to allow dealing with deprecated savefiles that didn't save the config file
    # separately.
    config : dict 
    if model_config_file_path is None:
        config = model_data['config']
    else:
        with open(model_config_file_path, 'rb') as model_config_file:
            config = pickle.load(model_config_file)    

    model_config = config['model']

    depth = model_config['depth']
    base_channel_num = model_config['base_channel_num']
    in_channels = model_config['in_channels']
    out_channels = model_config['out_channels']
    padding_mode = model_config['convolution_padding_mode']

    model = UNet(depth=depth,
                 base_channel_num=base_channel_num,
                 in_channels=in_channels,
                 out_channels=out_channels,
                 padding_mode=padding_mode)
    model.load_state_dict(model_data['state_dict'])

    return model, config