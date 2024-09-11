from pathlib import Path
import hashlib
from typing import List, Iterable

def _file_content_hash(file_path : Path) -> str:
    """Generate a hash for the file content"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def hash_files(files : Iterable[Path]) -> str:
    raw_input = ("_".join(map(_file_content_hash, files))).encode()
    hashed = hashlib.md5(raw_input).hexdigest()

    return hashed
