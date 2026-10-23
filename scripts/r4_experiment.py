"""R4 phase A: exactly two predeclared seeds, with locked choices before scoring."""
import argparse
import csv
from pathlib import Path
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
from src.solvers import BICNotApplicable
from src.bridge.numeric_io import stamp, write_rows
from src.data.contracts import load_dataset
from r4_methods import fit_node
from r4_metrics import binary_metrics, edge_scores, restore_coefficients, validate_phase, validate_splits


def run_julia(script, config_path, output):
    command = [str(ROOT/'environment/r0/julia.sh'), '--startup-file=no',
               '--project='+str(ROOT/'environment/r0/julia'), str(ROOT/'scripts'/script), str(config_path), str(output)]
    print('COMMAND='+__import__('shlex').join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def protect_previous(output):
    paths = set()
    for filename in ('results/r1/artifact_manifest.toml', 'results/r3/artifact_manifest.toml'):
        path = ROOT/filename
        manifest = tomllib.loads(path.read_text())
        for name, expected in manifest['file_sha256'].items():
            if file_hash(ROOT/name) != expected:
                raise ValueError('Previous stage artifact changed: '+name)
            paths.add(ROOT/name)
        paths.add(path)
    preflight = tomllib.loads((ROOT/'results/r3/preflight.toml').read_text())
    paths.update(ROOT/name for name in preflight['protected_sha256'])
    hashes = {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}
    write_toml(output/'preflight.toml', dict(created_at=stamp(), protected_sha256=hashes))
    return hashes


def fit_python_trial(method, data, config, trial):
    trial.mkdir()
    train, val = data['splits'] == 'train', data['splits'] == 'validation'
    X, Xv = data['Z'][:, train].T, data['Z'][:, val].T
    coefficients, nodes = [], []
    started = time.perf_counter()
    for node in range(config['nodes']):
        validation = dict(X_validation=Xv, y_validation=data['Y'][node, val],
            train_groups=data['groups'][train], validation_groups=data['groups'][val])
        try:
            coef, diagnostics = fit_node(method, X, data['Y'][node, train], config, validation)
            coefficients.append(coef)
            nodes.append(dict(node_julia_index=node+1, status='OK', **diagnostics))
        except BICNotApplicable as exc:
            if method not in ('M1', 'M2'):
                raise
            nodes.append(dict(node_julia_index=node+1, status='NA', reason=str(exc)))
    elapsed = time.perf_counter()-started
    status = dict(method=method, seed=data['metadata']['seed'], fit_seconds=elapsed,
        runtime_scope='Python fit including per-node guards and validation paths; process startup excluded',
        data_metadata_sha256=file_hash(data['directory']/'metadata.toml'), preprocessing='common standardized',
        description=config['methods'][method])
    if any(row['status'] == 'NA' for row in nodes):
        status.update(status='NA', reason='BIC unavailable for at least one node; no fallback or zero coefficients substituted',
                      unavailable_nodes=[row['node_julia_index'] for row in nodes if row['status'] == 'NA'])
    else:
        Cz = np.asarray(coefficients)
        C = restore_coefficients(Cz, data['means'], data['scales'])
        restored = C@data['theta'][:, train]
        standardized = Cz@data['Z'][:, train]
        if not np.allclose(restored, standardized, atol=1e-12, rtol=1e-12):
            raise ValueError('Common preprocessing coefficient restoration failed')
        for name, matrix in [('standardized_coefficients.csv', Cz), ('coefficients.csv', C)]:
            np.savetxt(trial/name, matrix, delimiter=',', fmt='%.17g')
        status.update(status='OK', prediction_restore_max_abs=float(np.max(np.abs(restored-standardized))))
    write_toml(trial/'diagnostics.toml', dict(nodes=nodes))
    write_toml(trial/'status.toml', status)
    print(f'R4_FIT seed={status["seed"]} method={method} status={status["status"]} seconds={elapsed:.4f}', flush=True)


def evaluate_trial(data, config, method, trial, lock_hash):
    status = tomllib.loads((trial/'status.toml').read_text())
    base = dict(seed=data['metadata']['seed'], method=method, status=status['status'], reason=status.get('reason', ''),
                fit_seconds=status['fit_seconds'], data_metadata_sha256=status['data_metadata_sha256'])
    if status['status'] != 'OK':
        return base
    C = read_matrix(trial/'coefficients.csv', (config['nodes'], data['metadata']['P']))
    # This is the first semantic read of truth/test responses for scoring.
    truth = read_matrix(data['directory']/'true_coefficients.csv', C.shape)
    decoded = list(csv.DictReader((trial/'inferred_structure.csv').open()))
    metrics, audits = {}, []
    for name, order in [('pairwise', 2), ('triple', 3)]:
        keys, scores = edge_scores(C, data['d'], order)
        truth_keys, truth_scores = edge_scores(truth, data['d'], order)
        assert keys == truth_keys
        labels = truth_scores > config['truth_threshold']
        metrics[name] = binary_metrics(labels, scores, config['decode_threshold'])
        predicted_keys = {key for key, score in zip(keys, scores) if score > config['decode_threshold']}
        julia_keys = {(int(row['target']), *[int(row['source_'+str(i)]) for i in range(1, order)])
                      for row in decoded if int(row['order']) == order}
        if julia_keys != predicted_keys:
            raise ValueError('Python scores disagree with Julia THIS decoding')
        for key, label, score in zip(keys, labels, scores):
            audits.append(dict(order=order, target=key[0], source_1=key[1], source_2=key[2] if order == 3 else 0,
                truth=int(label), score=float(score), predicted=int(score > config['decode_threshold'])))
        base.update({name+'_'+key:value for key, value in metrics[name].items()})
    metrics['taylor_support'] = binary_metrics((np.abs(truth) > config['truth_threshold']).ravel(),
                                             np.abs(C).ravel(), config['decode_threshold'])
    support_rows = [dict(target=i+1, feature_julia=j+1, d_1=int(data['d'][j, 0]), d_2=int(data['d'][j, 1]),
        d_3=int(data['d'][j, 2]), true_coefficient=float(truth[i, j]), coefficient=float(C[i, j]),
        truth=int(abs(truth[i, j]) > config['truth_threshold']), predicted=int(abs(C[i, j]) > config['decode_threshold']))
        for i in range(C.shape[0]) for j in range(C.shape[1])]
    write_rows(trial/'edge_scores.csv', audits)
    write_rows(trial/'taylor_support.csv', support_rows)
    metrics['regression'] = {}
    for name in ('train', 'validation', 'test'):
        mask = data['splits'] == name
        mse = float(np.mean((data['Y'][:, mask]-C@data['theta'][:, mask])**2))
        metrics['regression'][name+'_MSE'] = mse;base[name+'_MSE'] = mse
    metrics['selection_lock_sha256'] = lock_hash
    metrics['evaluated_at'] = stamp()
    metrics['threshold_rule'] = 'strict abs(restored raw coefficient)>1e-8; no threshold sweep'
    metrics['AUPRC_definition'] = 'sklearn average_precision_score (AP), all eligible candidate edges; not trapezoidal PR area'
    metrics['ignored_returned_rows'] = {str(order):sum(int(row['order']) == order for row in decoded) for order in (1, 4)}
    write_toml(trial/'metrics.toml', metrics)
    for key, value in metrics['taylor_support'].items():
        base['taylor_'+key] = value
    print(f'R4_EVALUATED seed={base["seed"]} {method}: pair_F1={metrics["pairwise"]["F1"]:.4f} triple_F1={metrics["triple"]["F1"]:.4f}', flush=True)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/r4_dry_run.toml')
    parser.add_argument('--output', type=Path, default=ROOT/'results/r4/dry_run')
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT/'results/r4'):
        raise ValueError('Output must be inside results/r4')
    config = tomllib.loads(config_path.read_text());validate_phase(config)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    protected = protect_previous(output)
    shutil.copyfile(config_path, output/'config_locked.toml')
    source_hashes = {str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'scripts').glob('r4_*')) if p.is_file()}
    write_toml(output/'protocol.toml', dict(created_at=stamp(), phase='A', B_authorized=False,
        config_sha256=file_hash(config_path), source_sha256=source_hashes,
        seeds=config['seeds'], threshold=config['decode_threshold'], threshold_scan=False,
        label_use='Truth only evaluated after all method choices locked; no test-driven model selection',
        planned_comparisons=['M2-M1 (BIC, NA if inapplicable)', 'M3-M1v (same independent validation strategy)',
                             'M3-M2 (selection strategy, NA if M2 unavailable)', 'M0s-M0 (preprocessing control)']))
    try:
        run_julia('r4_prepare.jl', config_path, output)
        for seed in config['seeds']:
            data = load_dataset(output, seed)
            for method in ('M1', 'M2', 'M3', 'M1v'):
                fit_python_trial(method, data, config, output/f'seed_{seed}'/method)
        locks = {}
        for seed in config['seeds']:
            for method in config['method_order']:
                trial = output/f'seed_{seed}'/method
                locks[f'{seed}_{method}'] = dict(status=tomllib.loads((trial/'status.toml').read_text())['status'],
                    status_sha256=file_hash(trial/'status.toml'),
                    coefficient_sha256=file_hash(trial/'coefficients.csv') if (trial/'coefficients.csv').exists() else 'NA')
        write_toml(output/'selection_locked.toml', dict(locked_at=stamp(), test_evaluated_before_lock=False,
                   structural_labels_used_for_selection=False, trials=locks))
        print('ALL_METHOD_CHOICES_LOCKED_BEFORE_TEST_EVALUATION', flush=True)
        run_julia('r4_decode.jl', config_path, output)
        lock_hash = file_hash(output/'selection_locked.toml')
        rows = []
        for seed in config['seeds']:
            data = load_dataset(output, seed)
            for method in config['method_order']:
                rows.append(evaluate_trial(data, config, method, output/f'seed_{seed}'/method, lock_hash))
        fields = list(dict.fromkeys(key for row in rows for key in row))
        write_rows(output/'trials.csv', rows, fields)
        paired = []
        for seed in config['seeds']:
            by_method = {row['method']:row for row in rows if row['seed'] == seed}
            for a, b in [('M2', 'M1'), ('M3', 'M1v'), ('M3', 'M2'), ('M0s', 'M0')]:
                available = by_method[a]['status'] == by_method[b]['status'] == 'OK'
                paired.append(dict(seed=seed, contrast=a+'-'+b, status='OK' if available else 'NA',
                    pairwise_F1_difference=by_method[a]['pairwise_F1']-by_method[b]['pairwise_F1'] if available else '',
                    triple_F1_difference=by_method[a]['triple_F1']-by_method[b]['triple_F1'] if available else '',
                    triple_AP_difference=by_method[a]['triple_AUPRC_AP']-by_method[b]['triple_AUPRC_AP'] if available else '',
                    reason='' if available else 'At least one preregistered method is NA; no replacement'))
        write_rows(output/'paired_dry_run.csv', paired)
        for name, expected in protected.items():
            if file_hash(ROOT/name) != expected:
                raise ValueError('Protected prior artifact changed: '+name)
        write_toml(output/'completion.toml', dict(completed_at=stamp(), phase_A_completed=True,
            phase_B_executed=False, phase_B_requires_explicit_authorization=True, r5_executed=False,
            seed_count=len(config['seeds']), trial_count=len(rows),
            OK_count=sum(row['status'] == 'OK' for row in rows), NA_count=sum(row['status'] == 'NA' for row in rows),
            excluded_failed_seeds=[], total_seconds=time.perf_counter()-start, protected_files_unchanged=True,
            interpretation='Two-seed pipeline check only; no 30-seed estimates, CIs, or superiority claim'))
        print('R4_PHASE_A_DRY_RUN_PASS; TWO_SEEDS_ONLY; STOP_BEFORE_PHASE_B_AND_R5', flush=True)
    except Exception:
        write_toml(output/'failure.toml', dict(failed_at=stamp(), traceback=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter()-start, excluded_seeds=[]))
        raise


if __name__ == '__main__':
    main()
