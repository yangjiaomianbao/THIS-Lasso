"""Paired seed-cluster bootstrap; missing values remain missing."""
import numpy as np


def validate_full(c):
    if c['experiment'] != 'E1' or c['phase'] != 'full' or not c['full_30_seed_authorized']:
        raise ValueError('This entry requires authorization for R5 full E1')
    if c['seeds'] != list(range(22261017, 22261047)):
        raise ValueError('Full E1 requires the exact 30 preregistered consecutive seeds')
    if c['E2_authorized'] or c['E3_authorized'] or c['method_order'] != ['M0', 'M1', 'M2', 'M3']:
        raise ValueError('E2/E3 are outside this E1 extension')
    if c['noise_ratios'] != [0., .05, .10, .20] or c['workers'] != 4:
        raise ValueError('Locked noise levels and four single-thread workers required')
    if c['bootstrap_replicates'] != 10000 or c['confidence_level'] != .95:
        raise ValueError('Locked bootstrap settings changed')
    if c.get('reuse_pilot_clean_data') is not False or c.get('trajectory_seed_stride') != 10000 or c.get('trajectory_seed_origin') != 1000000000:
        raise ValueError('Independent nonoverlapping trajectory RNG namespaces are mandatory')


def trajectory_seed_base(c, seed):
    return c['trajectory_seed_origin']+(c['seeds'].index(seed)+1)*c['trajectory_seed_stride']


def bootstrap_indices(n, repeats, seed):
    if n < 1 or repeats < 1:
        raise ValueError('Positive sample and replicate counts required')
    return np.random.default_rng(seed).integers(0, n, size=(repeats, n))


def describe(values, indices):
    values = np.asarray(values, float)
    if values.ndim != 1 or indices.ndim != 2 or indices.shape[1] != len(values):
        raise ValueError('Require one value per original seed and full seed-cluster indices')
    finite = np.isfinite(values);valid = values[finite]
    if not len(valid):
        return dict(n_total=len(values), n_available=0, mean=float('nan'), std=float('nan'),
                    ci_low=float('nan'), ci_high=float('nan'), bootstrap_available=0)
    sampled = values[indices]
    available = np.isfinite(sampled)
    counts = available.sum(axis=1)
    means = np.sum(np.where(available, sampled, 0.), axis=1)[counts > 0]/counts[counts > 0]
    low, high = np.quantile(means, [.025, .975])
    return dict(n_total=len(values), n_available=len(valid), mean=float(valid.mean()),
        std=float(valid.std(ddof=1)) if len(valid) > 1 else float('nan'),
        ci_low=float(low), ci_high=float(high), bootstrap_available=len(means))


def paired_values(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape:
        raise ValueError('Pair both methods by the same seed order')
    return np.where(np.isfinite(a) & np.isfinite(b), a-b, np.nan)
