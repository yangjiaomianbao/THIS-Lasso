"""One seed's twelve Python models, without truth scoring or test evaluation."""
import argparse
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_bridge import write_toml
from r4_experiment import load_dataset, stamp
from r5_experiment import fit_trial, noisy_data
from r5_noise import noise_slot
from r5_batch_stats import validate_full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True);args = p.parse_args()
    c = tomllib.loads(args.config.read_text());validate_full(c)
    if args.seed not in c['seeds']:
        raise ValueError('Seed not preregistered')
    base = tomllib.loads((ROOT/c['base_config']).read_text())
    clean = load_dataset(args.output/'clean', args.seed);start = time.perf_counter()
    print('SEED_WORKER_START seed='+str(args.seed)+' at='+stamp(), flush=True)
    for ratio in c['noise_ratios']:
        cell = args.output/f'seed_{args.seed}'/noise_slot(ratio);data = noisy_data(clean, cell)
        for method in ('M1', 'M2', 'M3'):
            fit_trial(method, data, base, cell/method, ratio, cell/'data/metadata.toml')
    write_toml(args.output/f'seed_{args.seed}'/'fitting_done.toml',
               dict(seed=args.seed, fitting_completed_at=stamp(), seconds=time.perf_counter()-start, test_evaluated=False))
    print('SEED_WORKER_DONE seed='+str(args.seed), flush=True)


if __name__ == '__main__':
    main()
