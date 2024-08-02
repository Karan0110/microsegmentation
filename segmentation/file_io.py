from pathlib import Path
from typing import Tuple, Any, Union
import json5
import os

import torch
import torch.nn as nn
from torch.optim import Optimizer

from unet import UNet

def get_model_save_file_paths(model_dir : Path,
                        model_name : str) -> Tuple[Path, Path]:
    state_save_file_path = model_dir / model_name / f"{model_name}.pth"
    config_save_file_path = model_dir / model_name / f"config.json5"

    return state_save_file_path, config_save_file_path

def save_model(model : nn.Module,
               optimizer : Optimizer,
               scheduler : Any,
               config : dict,
               model_dir : Path, 
               model_name : str,
               epoch : int,
               verbose : bool = False) -> Path:
    model_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    state_save_file_path, config_save_file_path = get_model_save_file_paths(model_dir=model_dir,
                                                                            model_name=model_name)

    os.makedirs(state_save_file_path.parent, exist_ok=True)
    os.makedirs(config_save_file_path.parent, exist_ok=True)

    try:
        # Save the model state
        torch.save(model_state, state_save_file_path)
        
        # Save the configuration
        with config_save_file_path.open('w') as f:
            json5.dump({**config, 
                        'epoch': epoch},
                        fp=f,
                        indent=4,)
    except KeyboardInterrupt:
        # Handle the keyboard interrupt or any cleanup here if necessary
        if verbose:
            print("Training interrupted midway through saving results.")
        # You may optionally delete the partially saved files here if appropriate
        if state_save_file_path.exists():
            state_save_file_path.unlink()
        if config_save_file_path.exists():
            config_save_file_path.unlink()
        raise

    if verbose:
        print(f"Saved model to {state_save_file_path}.")
        print(f"Saved model config to {config_save_file_path}.")
    
    return state_save_file_path

# Either loads a .json5 config file or a folder of .json5 config files.
def load_config(path : Path) -> dict:
    config = {}

    if path.is_dir():
        for config_file_path in path.glob('*.json5'):
            key = config_file_path.stem
            
            with config_file_path.open('r') as file:
                sub_config = json5.load(file)
                if not isinstance(sub_config, dict):
                    raise ValueError(f"The file {config_file_path} is not a dictionary!")
                config[key] = sub_config
    elif path.is_file() and path.suffix == '.json5':
        with path.open('r') as file:
            config = json5.load(file)
        if not isinstance(config, dict):
            raise ValueError(f"The file {path} is not a dictionary!")
    else:
        raise ValueError(f"Invalid config path: {path}\nMust be a .json5 file or a directory!")

    return config

def load_model(model_file_path : Path,
               config_file_path : Path) -> Tuple[nn.Module, dict]:
    model_data : dict = torch.load(model_file_path,
                                   weights_only=True)

    config : dict 
    with config_file_path.open('r') as model_config_file:
        config = json5.load(model_config_file) #type: ignore
    model_config = config['model']
    training_config = config['training']

    depth = model_config['depth']
    base_channel_num = model_config['base_channel_num']
    in_channels = model_config['in_channels']
    out_channels = model_config['out_channels']
    padding_mode = model_config['convolution_padding_mode']
    dropout_rate = training_config['dropout_rate']

    model = UNet(depth=depth,
                 base_channel_num=base_channel_num,
                 in_channels=in_channels,
                 out_channels=out_channels,
                 dropout_rate=dropout_rate,
                 padding_mode=padding_mode)
    model.load_state_dict(model_data['state_dict'])

    return model, model_config
