"""Hash-checked Taylor data loading and whole-trajectory split contract."""
import csv
import tomllib
import numpy as np
from src.bridge.numeric_io import file_hash, read_matrix


def validate_splits(groups, splits):
    groups, splits = np.asarray(groups), np.asarray(splits)
    if groups.shape != splits.shape or groups.ndim != 1:
        raise ValueError('Split/group shapes differ')
    sets = [set(groups[splits == name]) for name in ('train', 'validation', 'test')]
    if not all(sets) or set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('All three nonempty splits required')
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('A trajectory crosses data splits')
    return dict(zip(('train', 'validation', 'test'), map(len, sets)))



def load_dataset(output, seed):
    directory = output/f'seed_{seed}'/'data'
    metadata = tomllib.loads((directory/'metadata.toml').read_text())
    for name, expected in metadata['file_sha256'].items():
        if file_hash(directory/name) != expected:
            raise ValueError('Dataset hash mismatch: '+name)
    n, p, k = metadata['N'], metadata['P'], metadata['K_total']
    rows = list(csv.DictReader((directory/'sample_manifest.csv').open()))
    groups = np.array([int(row['trajectory_id']) for row in rows])
    splits = np.array([row['split'] for row in rows])
    validate_splits(groups, splits)
    return dict(directory=directory, metadata=metadata, groups=groups, splits=splits,
        theta=read_matrix(directory/'theta.csv', (p, k)),
        Z=read_matrix(directory/'standardized_theta.csv', (p, k)),
        Y=read_matrix(directory/'Y.csv', (n, k)), d=read_matrix(directory/'monomial_indices.csv', (p, 3), integer=True),
        means=read_matrix(directory/'feature_means.csv', (p, 1))[:, 0],
        scales=read_matrix(directory/'feature_scales.csv', (p, 1))[:, 0])

