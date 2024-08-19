from pathlib import Path
import json5

def save_json5(data : dict,
               path: Path,
               pretty_print : bool = False) -> None:
    with path.open('w') as f:
        indent = 4 if pretty_print else None
        json5.dump(data,
                   fp=f,
                   indent=indent)
