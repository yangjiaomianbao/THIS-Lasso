"""Verify actual initial states as well as RNG namespace disjointness."""
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from src.data_bridge import write_toml
from r4_experiment import stamp


def count_initials(root):
    states = set();count = 0
    for path in sorted(root.glob('seed_*/data/initial_conditions.csv')):
        initial = np.loadtxt(path, delimiter=',', ndmin=2)
        for column in initial.T:
            states.add(np.ascontiguousarray(column).tobytes());count += 1
    return count, len(states)


def main():
    output = ROOT/'results/r5/e1_full_independent'
    c = tomllib.loads((output/'config_locked.toml').read_text())
    streams = set()
    for seed in c['seeds']:
        m = tomllib.loads((output/'clean'/f'seed_{seed}/data/metadata.toml').read_text())
        local = set(m['trajectory_rng_seeds'])
        assert len(local) == 160 and not streams & local
        streams.update(local)
    count, unique = count_initials(output/'clean')
    assert count == unique == len(streams) == 4800
    previous_count, previous_unique = count_initials(ROOT/'results/r5/e1_full/clean')
    write_toml(output/'trajectory_independence.toml', dict(verified_at=stamp(), corrected_rng_stream_count=len(streams),
        corrected_initial_states=count, corrected_unique_initial_states=unique,
        aborted_initial_states=previous_count, aborted_unique_initial_states=previous_unique,
        aborted_scored=False, prior_generator_unchanged=True))
    print(f'CORRECTED_INITIAL_STATES={count}, UNIQUE={unique}; ABORTED_INITIAL_STATES={previous_count}, UNIQUE={previous_unique}')


if __name__ == '__main__':main()
