"""R5 E1 two-seed smoke, fixed clean R4 features and paired response noise."""
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
from src.solvers import BICNotApplicable, SolverConvergenceError
from r4_experiment import stamp, write_rows, protect_previous, load_dataset, evaluate_trial
from r4_methods import fit_node
from r4_metrics import edge_scores, restore_coefficients
from r5_noise import validate_config, noise_slot, response_noise_basis, inject_response_noise, true_strength_group


def julia(script, config, base, output):
    command = [str(ROOT/'environment/r0/julia.sh'), '--startup-file=no',
        '--project='+str(ROOT/'environment/r0/julia'), str(ROOT/'scripts'/script), str(config), str(base), str(output)]
    print('COMMAND='+__import__('shlex').join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def protect_r4(output):
    protected = protect_previous(output)
    path = ROOT/'results/r4/artifact_manifest.toml'
    prior = tomllib.loads(path.read_text())
    for name, expected in prior['file_sha256'].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError('R4 artifact changed: '+name)
        protected[name] = expected
    protected[str(path.relative_to(ROOT))] = file_hash(path)
    write_toml(output/'preflight.toml', dict(created_at=stamp(), protected_sha256=protected))
    return protected


def generate_noise(clean, config, output):
    seed = clean['metadata']['seed']
    seed_dir = output/f'seed_{seed}';seed_dir.mkdir()
    scales, unit, streams = response_noise_basis(clean['Y'], clean['splits'], seed, config['noise_master_seed'])
    np.savetxt(seed_dir/'unit_noise.csv', unit, delimiter=',', fmt='%.17g')
    np.savetxt(seed_dir/'signal_scales.csv', scales[:, None], delimiter=',', fmt='%.17g')
    write_toml(seed_dir/'noise_streams.toml', dict(bit_generator='PCG64', numpy_version=np.__version__,
        streams=streams, scale_fit_split='clean train only', std_ddof=0,
        unit_noise_sha256=file_hash(seed_dir/'unit_noise.csv'), signal_scales_sha256=file_hash(seed_dir/'signal_scales.csv')))
    for ratio in config['noise_ratios']:
        directory = seed_dir/noise_slot(ratio)/'data';directory.mkdir(parents=True)
        Y, eta = inject_response_noise(clean['Y'], scales, unit, ratio)
        np.savetxt(directory/'Y_noisy.csv', Y, delimiter=',', fmt='%.17g')
        np.savetxt(directory/'eta.csv', eta, delimiter=',', fmt='%.17g')
        actual = {name:np.std(eta[:, clean['splits'] == name], axis=1, ddof=0)
                  for name in ('train', 'validation', 'test')}
        source_names = ['states.csv', 'Y.csv', 'theta.csv', 'standardized_theta.csv', 'monomial_indices.csv',
                        'true_coefficients.csv', 'sample_manifest.csv', 'feature_means.csv', 'feature_scales.csv']
        write_toml(directory/'metadata.toml', dict(experiment='E1', seed=seed, noise_ratio=ratio,
            ratio_is='standard deviation multiplier, not variance', injection_location='Y only, all three disjoint splits',
            source_data_dir=str(clean['directory']), source_metadata_sha256=file_hash(clean['directory']/'metadata.toml'),
            clean_file_sha256={name:file_hash(clean['directory']/name) for name in source_names},
            clean_train_signal_std=scales, target_noise_std=ratio*scales, realized_noise_std=actual,
            source_state_and_feature_arrays_unchanged=True, common_random_numbers_across_levels=True,
            Y_noisy_sha256=file_hash(directory/'Y_noisy.csv'), eta_sha256=file_hash(directory/'eta.csv')))
    print(f'E1_NOISE_PREPARED seed={seed} ratios={config["noise_ratios"]} signal_std={scales.tolist()}', flush=True)


def noisy_data(clean, cell):
    return dict(clean, Y=read_matrix(cell/'data/Y_noisy.csv', clean['Y'].shape))


def fit_trial(method, data, base, trial, ratio, noise_metadata):
    trial.mkdir()
    train, val = data['splits'] == 'train', data['splits'] == 'validation'
    X = data['Z'][:, train].T
    nodes, coefficients = [], []
    started = time.perf_counter()
    for node in range(base['nodes']):
        validation = dict(X_validation=data['Z'][:, val].T, y_validation=data['Y'][node, val],
            train_groups=data['groups'][train], validation_groups=data['groups'][val])
        try:
            coef, diagnostics = fit_node(method, X, data['Y'][node, train], base, validation)
            nodes.append(dict(node_julia_index=node+1, status='OK', **diagnostics));coefficients.append(coef)
        except BICNotApplicable as exc:
            if method not in ('M1', 'M2'):
                raise
            nodes.append(dict(node_julia_index=node+1, status='NA', reason=str(exc)))
        except SolverConvergenceError as exc:
            nodes.append(dict(node_julia_index=node+1, status='FAILED', reason=str(exc), traceback=traceback.format_exc()))
    status = dict(method=method, seed=data['metadata']['seed'], noise_ratio=ratio,
        fit_seconds=time.perf_counter()-started, runtime_scope='Python guards and fitting; excludes process startup',
        data_metadata_sha256=file_hash(data['directory']/'metadata.toml'),
        noise_metadata_sha256=file_hash(noise_metadata), noisy_response_sha256=file_hash(noise_metadata.parent/'Y_noisy.csv'))
    unavailable = [n for n in nodes if n['status'] != 'OK']
    if unavailable:
        label = 'FAILED' if any(n['status'] == 'FAILED' for n in unavailable) else 'NA'
        status.update(status=label, reason='; '.join(f'node {n["node_julia_index"]}: {n["reason"]}' for n in unavailable),
                      unavailable_nodes=[n['node_julia_index'] for n in unavailable])
    else:
        Cz = np.asarray(coefficients);C = restore_coefficients(Cz, data['means'], data['scales'])
        difference = float(np.max(np.abs(C@data['theta'][:, train]-Cz@data['Z'][:, train])))
        if difference > 1e-11:
            raise ValueError('E1 original-unit restoration failed')
        np.savetxt(trial/'standardized_coefficients.csv', Cz, delimiter=',', fmt='%.17g')
        np.savetxt(trial/'coefficients.csv', C, delimiter=',', fmt='%.17g')
        status.update(status='OK', prediction_restore_max_abs=difference)
    write_toml(trial/'diagnostics.toml', dict(nodes=nodes));write_toml(trial/'status.toml', status)
    print(f'E1_FIT seed={status["seed"]} ratio={ratio:.2f} method={method} status={status["status"]} seconds={status["fit_seconds"]:.3f}', flush=True)


def strength_rows(clean, trial, base, config, method, ratio):
    status = tomllib.loads((trial/'status.toml').read_text())['status']
    truth = read_matrix(clean['directory']/'true_coefficients.csv', (base['nodes'], clean['metadata']['P']))
    C = read_matrix(trial/'coefficients.csv', truth.shape) if status == 'OK' else None
    rows = []
    for name, order in [('pairwise', 2), ('triple', 3)]:
        keys, true_values = edge_scores(truth, clean['d'], order)
        scores = edge_scores(C, clean['d'], order)[1] if C is not None else None
        for strength in ('weak', 'medium', 'strong'):
            mask = np.array([v > base['truth_threshold'] and true_strength_group(v, config['strength_weak_upper'],
                              config['strength_medium_upper']) == strength for v in true_values])
            count = int(np.sum(mask))
            tp = int(np.sum(scores[mask] > base['decode_threshold'])) if scores is not None else None
            rows.append(dict(seed=clean['metadata']['seed'], noise_ratio=ratio, method=method, status=status,
                order=name, strength=strength, true_count=count, TP=tp if tp is not None else '',
                FN=count-tp if tp is not None else '', recall=tp/count if tp is not None and count else ''))
    return rows


def summarize(rows, config, output):
    summaries, pairs = [], []
    metrics = ['precision', 'recall', 'F1', 'MCC', 'AUPRC_AP', 'FP', 'FN']
    for ratio in config['noise_ratios']:
        for method in config['method_order']:
            group = [r for r in rows if r['noise_ratio'] == ratio and r['method'] == method]
            summary = dict(noise_ratio=ratio, method=method, attempted=len(group),
                valid_count=sum(r['status'] == 'OK' for r in group), NA_count=sum(r['status'] == 'NA' for r in group),
                failed_count=sum(r['status'] == 'FAILED' for r in group), interpretation='two-seed descriptive mean, not 30-seed estimate; no CI')
            for order in ('pairwise', 'triple'):
                for metric in metrics:
                    key = order+'_'+metric
                    values = [float(r[key]) for r in group if r['status'] == 'OK' and np.isfinite(float(r[key]))]
                    summary[key+'_mean'] = float(np.mean(values)) if values else ''
                    summary[key+'_available_count'] = len(values)
            summaries.append(summary)
        for seed in config['seeds']:
            group = {r['method']:r for r in rows if r['seed'] == seed and r['noise_ratio'] == ratio}
            for a, b in [('M2', 'M1'), ('M2', 'M0'), ('M3', 'M2')]:
                available = group[a]['status'] == group[b]['status'] == 'OK'
                pair = dict(seed=seed, noise_ratio=ratio, contrast=a+'-'+b, status='OK' if available else 'NA',
                            reason='' if available else 'At least one NA/FAILED trial; no replacement')
                for metric in ('F1', 'AUPRC_AP', 'FP', 'FN'):
                    key = 'triple_'+metric
                    pair[key+'_difference'] = group[a][key]-group[b][key] if available else ''
                pairs.append(pair)
    write_rows(output/'smoke_summary.csv', summaries)
    write_rows(output/'paired_smoke.csv', pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/r5_e1_smoke.toml')
    parser.add_argument('--output', type=Path, default=ROOT/'results/r5/e1_smoke')
    args = parser.parse_args();config_path = args.config.resolve();output = args.output.resolve()
    config = tomllib.loads(config_path.read_text());validate_config(config)
    if not output.is_relative_to(ROOT/'results/r5'):
        raise ValueError('R5 output must remain inside results/r5')
    base_path = ROOT/config['base_config'];base = tomllib.loads(base_path.read_text())
    source = ROOT/config['source_root']
    if base['nodes'] != 8 or base['dmax'] != 3 or base['seeds'] != config['seeds']:
        raise ValueError('R4/R5 clean data contract differs')
    output.mkdir(parents=True, exist_ok=False);started = time.perf_counter()
    protected = protect_r4(output)
    shutil.copyfile(config_path, output/'config_locked.toml');shutil.copyfile(base_path, output/'r4_base_locked.toml')
    write_toml(output/'protocol.toml', dict(created_at=stamp(), experiment='E1', phase='two-seed smoke',
        R4_phase_A_verified=True, R4_phase_B_completed=False, user_explicitly_requested_R5=True,
        config_sha256=file_hash(config_path), base_sha256=file_hash(base_path),
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'scripts').glob('r5_*')) if p.is_file()},
        strength_bins=[.10, .17], groups_use='true physical coefficients, never predicted values',
        evaluation_rule='Lock all models before truth scoring and test errors; E1 shares all clean arrays and all noise across methods'))
    try:
        for seed in config['seeds']:
            generate_noise(load_dataset(source, seed), config, output)
        julia('r5_stlsq.jl', config_path, base_path, output)
        for seed in config['seeds']:
            clean = load_dataset(source, seed)
            for ratio in config['noise_ratios']:
                cell = output/f'seed_{seed}'/noise_slot(ratio)
                data = noisy_data(clean, cell)
                for method in ('M1', 'M2', 'M3'):
                    fit_trial(method, data, base, cell/method, ratio, cell/'data/metadata.toml')
        locks = {}
        for seed in config['seeds']:
            for ratio in config['noise_ratios']:
                for method in config['method_order']:
                    trial = output/f'seed_{seed}'/noise_slot(ratio)/method
                    locks[f'{seed}_{noise_slot(ratio)}_{method}'] = dict(status_sha256=file_hash(trial/'status.toml'),
                        coefficient_sha256=file_hash(trial/'coefficients.csv') if (trial/'coefficients.csv').exists() else 'NA_OR_FAILED')
        write_toml(output/'selection_locked.toml', dict(locked_at=stamp(), test_evaluated_before_lock=False,
            true_labels_used_for_selection=False, trials=locks))
        print('E1_ALL_MODELS_LOCKED_BEFORE_TRUTH_AND_TEST_EVALUATION', flush=True)
        julia('r5_decode.jl', config_path, base_path, output)
        lock_hash = file_hash(output/'selection_locked.toml');rows, strength = [], []
        for seed in config['seeds']:
            clean = load_dataset(source, seed)
            for ratio in config['noise_ratios']:
                cell = output/f'seed_{seed}'/noise_slot(ratio);data = noisy_data(clean, cell)
                for method in config['method_order']:
                    trial = cell/method
                    row = evaluate_trial(data, base, method, trial, lock_hash)
                    row.update(noise_ratio=ratio, experiment='E1')
                    if row['status'] == 'OK':
                        C = read_matrix(trial/'coefficients.csv', (base['nodes'], clean['metadata']['P']))
                        metrics = tomllib.loads((trial/'metrics.toml').read_text())
                        for name in ('train', 'validation', 'test'):
                            mask = clean['splits'] == name
                            value = float(np.mean((clean['Y'][:, mask]-C@clean['theta'][:, mask])**2))
                            metrics['regression'][name+'_clean_response_MSE'] = value;row[name+'_clean_response_MSE'] = value
                        metrics.update(noise_ratio=ratio, experiment='E1', noisy_regression_definition='Original regression keys score against Y_noisy; clean_response keys score against Y_clean')
                        write_toml(trial/'metrics.toml', metrics)
                    rows.append(row);strength.extend(strength_rows(clean, trial, base, config, method, ratio))
        write_rows(output/'trials.csv', rows, list(dict.fromkeys(key for row in rows for key in row)))
        write_rows(output/'strength_recall.csv', strength);summarize(rows, config, output)
        for name, expected in protected.items():
            if file_hash(ROOT/name) != expected:
                raise ValueError('Previous stage changed: '+name)
        counts = {name:sum(r['status'] == name for r in rows) for name in ('OK', 'NA', 'FAILED')}
        write_toml(output/'completion.toml', dict(completed_at=stamp(), E1_smoke_completed=True,
            seed_count=2, noise_level_count=4, trial_count=len(rows), counts=counts, excluded_seeds=[],
            full_30_seed_executed=False, E2_executed=False, E3_executed=False,
            all_selected_models_converged=counts['FAILED'] == 0, total_seconds=time.perf_counter()-started,
            prior_stage_files_unchanged=True, final_conclusion='Smoke only, no 30-seed confidence intervals or superiority claim'))
        print('R5_E1_SMOKE_COMPLETE '+str(counts)+'; STOP_BEFORE_30_SEEDS_E2_E3', flush=True)
    except Exception:
        write_toml(output/'failure.toml', dict(failed_at=stamp(), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-started, excluded_seeds=[]))
        raise


if __name__ == '__main__':
    main()
