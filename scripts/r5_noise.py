"""E1 response noise only. Neither test data nor measured noise sets the scale."""
import numpy as np


def validate_config(c):
    if c['experiment'] != 'E1' or c['phase'] != 'smoke' or len(c['seeds']) != 2 or len(set(c['seeds'])) != 2:
        raise ValueError('Only E1 smoke with two distinct fixed seeds is enabled')
    if any(c[k] for k in ('full_30_seed_authorized', 'E2_authorized', 'E3_authorized')):
        raise ValueError('This entry does not implement authorized batch/E2/E3; use a separately reviewed entry')
    ratios = np.asarray(c['noise_ratios'], float)
    if ratios.ndim != 1 or len(ratios) != 4 or not np.array_equal(ratios, [0., .05, .10, .20]):
        raise ValueError('Smoke noise levels must be the four preregistered standard deviation ratios')
    if c['method_order'] != ['M0', 'M1', 'M2', 'M3']:
        raise ValueError('Required four methods must keep their identities')


def noise_slot(ratio):
    return f'noise_{int(round(ratio*1000)):03d}'


def response_noise_basis(Y_clean, splits, clean_seed, master_seed):
    Y = np.asarray(Y_clean, float)
    splits = np.asarray(splits)
    if Y.ndim != 2 or splits.shape != (Y.shape[1],) or set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Invalid response/split contract')
    if not np.isfinite(Y).all():
        raise ValueError('Nonfinite clean responses')
    scales = np.std(Y[:, splits == 'train'], axis=1, ddof=0)
    unit = np.zeros_like(Y)
    streams = []
    for code, name in enumerate(('train', 'validation', 'test'), 1):
        mask = splits == name
        for node in range(Y.shape[0]):
            entropy = [int(master_seed), int(clean_seed), code, node+1]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
            unit[node, mask] = rng.standard_normal(int(mask.sum()))
            streams.append(dict(split=name, node_julia_index=node+1, entropy=entropy))
    return scales, unit, streams


def inject_response_noise(Y_clean, scales, unit, ratio):
    Y, scales, unit = np.asarray(Y_clean, float), np.asarray(scales, float), np.asarray(unit, float)
    if unit.shape != Y.shape or scales.shape != (Y.shape[0],) or not np.isfinite(ratio) or ratio < 0:
        raise ValueError('Invalid E1 injection shapes/ratio')
    eta = float(ratio)*scales[:, None]*unit
    return Y+eta, eta


def true_strength_group(value, weak=.10, medium=.17):
    amplitude = abs(float(value))
    return 'weak' if amplitude <= weak else ('medium' if amplitude <= medium else 'strong')
