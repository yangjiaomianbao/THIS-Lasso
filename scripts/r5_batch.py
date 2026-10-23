"""Authorized full E1 orchestration. All 480 model choices lock before scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_bridge import file_hash, read_matrix, write_toml
from r4_experiment import stamp, write_rows, load_dataset, evaluate_trial
from r5_experiment import protect_r4, generate_noise, noisy_data, strength_rows
from r5_noise import noise_slot
from r5_batch_stats import validate_full

MUTABLE_DOCS = ['reports/R5_report.md', 'reports/R5_learning.md', 'docs/r5_followup_noise_plan.md']


def protect_smoke(output):
    protected = protect_r4(output)
    previous_path = ROOT/'results/r5/artifact_manifest.toml'
    previous = tomllib.loads(previous_path.read_text())
    history = output/'history';history.mkdir()
    shutil.copyfile(previous_path, history/'smoke_artifact_manifest.toml')
    for name, expected in previous['file_sha256'].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError('R5 smoke artifact changed: '+name)
        if name in MUTABLE_DOCS:
            shutil.copyfile(ROOT/name, history/Path(name).name)
        else:
            protected[name] = expected
    write_toml(output/'preflight.toml', dict(created_at=stamp(), protected_sha256=protected,
        mutable_docs_archived=MUTABLE_DOCS, previous_manifest_sha256=file_hash(previous_path)))
    return protected


def julia(task, config, output):
    command = [str(ROOT/'environment/r0/julia.sh'), '--startup-file=no', '--project='+str(ROOT/'environment/r0/julia'),
               str(ROOT/'scripts/r5_batch_julia.jl'), task, str(config), str(output)]
    print('COMMAND='+shlex.join(command), flush=True);subprocess.run(command, cwd=ROOT, check=True)


def seed_worker(seed, config, output):
    command = [str(ROOT/'.venv-r0/bin/python'), str(ROOT/'scripts/r5_batch_worker.py'),
               '--config', str(config), '--output', str(output), '--seed', str(seed)]
    with (output/'logs'/f'seed_{seed}.log').open('x', buffering=1) as f:
        f.write('START='+stamp()+'\nCOMMAND='+shlex.join(command)+'\nENV=BLAS=1 JULIA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n')
        result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        f.write('EXIT_CODE='+str(result.returncode)+'\nEND='+stamp()+'\n')
    if result.returncode:
        raise RuntimeError(f'Seed worker failed: {seed}; see its retained log')
    return seed


def evaluate_all(output, c, base, lock_hash):
    rows, strata = [], []
    with (output/'logs/evaluation.log').open('x', buffering=1) as log:
        for seed in c['seeds']:
            clean = load_dataset(output/'clean', seed)
            for ratio in c['noise_ratios']:
                cell = output/f'seed_{seed}'/noise_slot(ratio);data = noisy_data(clean, cell)
                for method in c['method_order']:
                    trial = cell/method
                    with contextlib.redirect_stdout(log):
                        print(f'E1_FULL_EVALUATION_CONTEXT seed={seed} ratio={ratio} method={method}', flush=True)
                        row = evaluate_trial(data, base, method, trial, lock_hash)
                    row.update(noise_ratio=ratio, experiment='E1')
                    if row['status'] == 'OK':
                        C = read_matrix(trial/'coefficients.csv', (8, 165))
                        metrics = tomllib.loads((trial/'metrics.toml').read_text())
                        for name in ('train', 'validation', 'test'):
                            mask = clean['splits'] == name
                            value = float(np.mean((clean['Y'][:, mask]-C@clean['theta'][:, mask])**2))
                            metrics['regression'][name+'_clean_response_MSE'] = value;row[name+'_clean_response_MSE'] = value
                        metrics.update(noise_ratio=ratio, experiment='E1')
                        write_toml(trial/'metrics.toml', metrics)
                    rows.append(row);strata.extend(strength_rows(clean, trial, base, c, method, ratio))
            print(f'SEED_EVALUATED seed={seed}', flush=True)
    write_rows(output/'trials.csv', rows, list(dict.fromkeys(k for r in rows for k in r)))
    write_rows(output/'strength_recall.csv', strata)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/r5_e1_full_independent.toml')
    p.add_argument('--output', type=Path, default=ROOT/'results/r5/e1_full_independent');args = p.parse_args()
    config = args.config.resolve();output = args.output.resolve();c = tomllib.loads(config.read_text());validate_full(c)
    if output != ROOT/'results/r5/e1_full_independent':
        raise ValueError('This preregistered full run uses results/r5/e1_full_independent')
    output.mkdir(parents=True, exist_ok=False);(output/'logs').mkdir();started = time.perf_counter()
    protected = protect_smoke(output)
    shutil.copyfile(config, output/'config_locked.toml')
    base_path = ROOT/c['base_config'];shutil.copyfile(base_path, output/'r4_base_locked.toml')
    base = tomllib.loads(base_path.read_text())
    write_toml(output/'protocol.toml', dict(created_at=stamp(), experiment='E1', phase='full',
        authorization=c['authorization'], R4_full_completed=False, E2_E3_authorized=False,
        config_sha256=file_hash(config), base_sha256=file_hash(base_path),
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'scripts').glob('r5_batch*')) if p.is_file()},
        seed_order=c['seeds'], pilot_clean_data_reused=False, pilot_noise_streams_included=True, new_28_sensitivity=True,
        trajectory_rng_namespaces_disjoint=True, correction=c['correction_note'],
        numerical_budget='unchanged from smoke; retained warnings and failures',
        statistics='10000 paired-seed percentile bootstrap, pointwise 95%, finite-case availability explicit'))
    try:
        julia('prepare', config, output)
        with (output/'logs/noise_generation.log').open('x') as log, contextlib.redirect_stdout(log):
            for seed in c['seeds']:
                generate_noise(load_dataset(output/'clean', seed), c, output)
        print('E1_FULL_NOISE_PREPARED 30 seeds, 120 paired response matrices', flush=True)
        julia('stlsq', config, output)
        with ThreadPoolExecutor(max_workers=c['workers']) as pool:
            futures = {pool.submit(seed_worker, seed, config, output):seed for seed in c['seeds']}
            for done, future in enumerate(as_completed(futures), 1):
                seed = future.result();print(f'SEED_FIT_DONE {done}/30 seed={seed} elapsed_seconds={time.perf_counter()-started:.1f}', flush=True)
        locks = {}
        for seed in c['seeds']:
            for ratio in c['noise_ratios']:
                for method in c['method_order']:
                    trial = output/f'seed_{seed}'/noise_slot(ratio)/method
                    locks[f'{seed}_{noise_slot(ratio)}_{method}'] = dict(status_sha256=file_hash(trial/'status.toml'),
                        coefficient_sha256=file_hash(trial/'coefficients.csv') if (trial/'coefficients.csv').exists() else 'NA_OR_FAILED')
        write_toml(output/'selection_locked.toml', dict(locked_at=stamp(), test_evaluated_before_lock=False,
            true_labels_used_for_selection=False, trial_count=480, trials=locks))
        print('ALL_480_MODEL_CHOICES_LOCKED_BEFORE_SCORING', flush=True)
        julia('decode', config, output)
        rows = evaluate_all(output, c, base, file_hash(output/'selection_locked.toml'))
        command = [str(ROOT/'.venv-r0/bin/python'), str(ROOT/'scripts/r5_batch_analyse.py'), '--output', str(output)]
        print('COMMAND='+shlex.join(command), flush=True);subprocess.run(command, cwd=ROOT, check=True)
        for name, expected in protected.items():
            if file_hash(ROOT/name) != expected:
                raise ValueError('Protected earlier artifact changed: '+name)
        counts = {name:sum(r['status'] == name for r in rows) for name in ('OK', 'NA', 'FAILED')}
        write_toml(output/'completion.toml', dict(completed_at=stamp(), E1_full_completed=True,
            seed_count=30, trial_count=len(rows), counts=counts, excluded_seeds=[], E2_executed=False, E3_executed=False,
            all_selected_models_converged=counts['FAILED'] == 0, wall_seconds=time.perf_counter()-started,
            aggregate_fit_seconds=sum(float(r['fit_seconds']) for r in rows), workers=4, protected_files_unchanged=True))
        print('R5_E1_FULL_30_SEEDS_COMPLETE '+str(counts), flush=True)
    except Exception:
        write_toml(output/'failure.toml', dict(failed_at=stamp(), traceback=traceback.format_exc(), excluded_seeds=[]))
        raise


if __name__ == '__main__':
    main()
