"""Shared numerical exchange and audit table helpers; original serialization retained."""
from pathlib import Path
import csv
import datetime
import hashlib
import json

import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def _toml(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(str(k))+' = '+_toml(v) for k, v in value.items() if v is not None) + '}'
    if isinstance(value, (tuple, list)):
        return '[' + ', '.join(_toml(v) for v in value) + ']'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f'Unsupported TOML value {type(value)}')



def write_toml(path, values):
    Path(path).write_text('\n'.join(json.dumps(k)+' = '+_toml(v) for k, v in values.items() if v is not None)+'\n')



def read_matrix(path, shape, *, integer=False):
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    if a.shape != tuple(shape) or not np.isfinite(a).all():
        raise ValueError(f'{Path(path).name}: expected finite matrix {tuple(shape)}, got {a.shape}')
    if integer:
        if np.any(a != np.floor(a)) or np.any(np.abs(a) > 2**53):
            raise ValueError(f'{Path(path).name}: invalid integer IDs')
        a = a.astype(np.int64)
    return a



def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()



def write_rows(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields);writer.writeheader();writer.writerows(rows)

