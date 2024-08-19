from pathlib import Path
from typing import Any, Union
import json5

def _load_json5_file(file_path : Path) -> Union[list,dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = json5.load(f)
    
    if not (isinstance(contents, dict) or isinstance(contents, list)):
        raise ValueError(f"JSON5 file {file_path} is of invalid format - should be a dict or list")

    return contents

def _update_nested_dict_from_path(root, sub_path, value) -> None:
    parts = sub_path.parts
    
    current = root
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    
    suffix_length = len('.json5')
    last_part = parts[-1]
    last_part = last_part[:-suffix_length]

    current[last_part] = value

def load_json5(path: Path) -> Union[dict, list]:
    data = {}
    
    if path.is_file() and path.suffix == '.json5':
        data = _load_json5_file(path)
    elif path.is_dir():
        for json5_file_path in path.rglob('*.json5'):
            relative_path = json5_file_path.relative_to(path)
            nested_data = _load_json5_file(json5_file_path)

            _update_nested_dict_from_path(data, relative_path, nested_data)
    else:
        raise ValueError(f"The path {path} corresponds to neither a .json5 file nor a directory!") 
    
    return data
