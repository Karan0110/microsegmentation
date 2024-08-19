import argparse
import os
from typing import Any, Type
from pathlib import Path

# Load argument from CL, defaulting to environment var value
def get_argument(cl_args : argparse.Namespace, 
                 cl_arg_name : str, 
                 env_var_name : str,
                 ArgumentType : Type) -> Any:
    if getattr(cl_args,cl_arg_name) is None:
        return ArgumentType(os.environ[env_var_name])
    else:
        return getattr(cl_args, cl_arg_name)

def get_path_argument(cl_args : argparse.Namespace, 
                      cl_arg_name : str, 
                      env_var_name : str) -> Path:
    path_argument : Path
    path_argument = get_argument(cl_args=cl_args,
                                 cl_arg_name=cl_arg_name,
                                 env_var_name=env_var_name,
                                 ArgumentType=Path)
    
    if not path_argument.is_absolute():
        base_dir = Path(os.environ['BASE_DIR'])
        path_argument = base_dir / path_argument

    return path_argument
