"""Restore coefficients to original THIS units without changing bias semantics."""
import numpy as np


def restore_coefficients(C, means, scales):
    C, means, scales = np.asarray(C, float), np.asarray(means, float), np.asarray(scales, float)
    if C.ndim != 2 or means.shape != (C.shape[1],) or scales.shape != means.shape or np.any(scales <= 0):
        raise ValueError('Invalid scale restoration shapes')
    raw = C/scales
    raw[:, 0] = C[:, 0]-raw[:, 1:]@means[1:]
    return raw

