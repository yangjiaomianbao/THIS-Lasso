"""Compatibility exports for R4/R5 metrics, scaling and split checks."""
from src.metrics.structure import binary_metrics, edge_scores
from src.preprocessing.taylor import restore_coefficients
from src.data.contracts import validate_splits


def validate_phase(config):
    if config['phase'] != 'A' or config.get('full_phase_B_authorized') or len(config['seeds']) != 2:
        raise ValueError('This first-run entry only permits phase A with exactly two seeds; phase B needs new explicit authorization')
    if len(set(config['seeds'])) != 2 or config['nodes'] != 8 or config['dmax'] != 3 or config['pearson_enabled']:
        raise ValueError('R4 A requires two distinct seeds, N=8, dmax=3, no Pearson')
    expected = config['train_trajectories']*(config['integration_steps']//config['save_every']+1)
    if config['train_K'] != expected or expected not in (300, 1000):
        raise ValueError('Smoke case must have K_train=300 or 1000')

