from pathlib import Path
from typing import Tuple, Any
import os
import json5

import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils import instantiate_from_dict, load_json5
import models

def get_model_save_file_paths(model_dir : Path,
                        model_name : str) -> Tuple[Path, Path]:
    state_save_file_path = model_dir / model_name / f"{model_name}.pth"
    config_save_file_path = model_dir / model_name / f"config.json5"

    return state_save_file_path, config_save_file_path

def save_model_to_file(model : nn.Module,
                    optimizer : Optimizer,
                    scheduler : Any,
                    config : dict,
                    model_dir : Path, 
                    model_name : str,
                    epoch : int,
                    pretty_print : bool = True,
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

    # Wrap in try-except block to make saving atomic
    try:
        # Save the model state
        torch.save(model_state, state_save_file_path)
        
        # Save the configuration
        with config_save_file_path.open('w') as f:
            indent = 4 if pretty_print else None
            json5.dump({**config, 
                        'epoch': epoch},
                        fp=f,
                        indent=indent)
    except KeyboardInterrupt:
        if verbose:
            print("Training interrupted midway through saving results.")

        if state_save_file_path.exists():
            state_save_file_path.unlink()
        if config_save_file_path.exists():
            config_save_file_path.unlink()
        raise

    if verbose:
        print(f"Saved model to {state_save_file_path}.")
        print(f"Saved model config to {config_save_file_path}.")
    
    return state_save_file_path


def load_model_from_file(model_file_path : Path,
                         config_file_path : Path) -> Tuple[nn.Module, dict]:
    model_data : dict = torch.load(model_file_path,
                                   weights_only=True)

    config = load_json5(config_file_path)
    
    model = instantiate_from_dict(models,
                                  information=config['model'])

    model.load_state_dict(model_data['state_dict'])

    return model, config
