"""Independent artifact/metric audit of the authorized full E1 run."""
import csv
from collections import Counter
from pathlib import Path
import os
import re
import subprocess
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_bridge import file_hash, read_matrix, write_toml
from r4_experiment import load_dataset, stamp
from r4_metrics import binary_metrics, edge_scores, restore_coefficients
from r5_noise import response_noise_basis, inject_response_noise, noise_slot, true_strength_group
from r5_batch_stats import validate_full, trajectory_seed_base


def same(a, b):
    return bool((np.isnan(a) and np.isnan(b)) or np.isclose(a, b, atol=1e-24, rtol=1e-9))


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def independently_describe(values, indices):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(n_available=0, mean=np.nan, std=np.nan, ci_low=np.nan, ci_high=np.nan)
    draws = values[indices];counts = np.isfinite(draws).sum(axis=1);good = counts > 0
    boot = np.nansum(draws[good], axis=1)/counts[good]
    return dict(n_available=len(finite), mean=float(finite.mean()), std=float(finite.std(ddof=1)) if len(finite)>1 else np.nan,
                ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)))


def audit_statistics(output, c, trials):
    indices = np.loadtxt(output/'bootstrap_seed_indices.csv', delimiter=',', dtype=int, ndmin=2)
    expected = np.random.default_rng(c['bootstrap_seed']).integers(0, 30, (10000, 30))
    np.testing.assert_array_equal(indices, expected)
    lookup = {(int(r['seed']), float(r['noise_ratio']), r['method']):r for r in trials}
    def vector(ratio, method, order, metric):
        result = []
        for seed in c['seeds']:
            r = lookup[seed, ratio, method];value = r.get(order+'_'+metric, '')
            result.append(float(value) if r['status'] == 'OK' and value != '' else np.nan)
        return np.asarray(result)
    summaries = rows(output/'summary_30.csv')
    for r in summaries:
        values = vector(float(r['noise_ratio']), r['method'], r['order'], r['metric'])
        actual = independently_describe(values, indices)
        assert int(r['n_total']) == 30
        for key, value in actual.items():assert same(value, float(r[key])), (key, r)
    for r in rows(output/'paired_summary_30.csv'):
        a, b = r['contrast'].split('-');ratio = float(r['noise_ratio'])
        left = vector(ratio, a, r['order'], r['metric']);right = vector(ratio, b, r['order'], r['metric'])
        delta = np.where(np.isfinite(left)&np.isfinite(right), left-right, np.nan)
        for key, value in independently_describe(delta, indices).items():assert same(value, float(r[key]))
    for r in rows(output/'paired_seed_differences.csv'):
        a, b = r['contrast'].split('-');seed = int(r['seed']);ratio = float(r['noise_ratio'])
        for metric in ('F1', 'AUPRC_AP', 'precision', 'recall', 'MCC', 'TP', 'FP', 'FN', 'TN'):
            l, q = lookup[seed, ratio, a], lookup[seed, ratio, b]
            actual = float(l[r['order']+'_'+metric])-float(q[r['order']+'_'+metric]) if l['status']==q['status']=='OK' else np.nan
            assert same(actual, float(r[metric+'_difference']))
    strata = rows(output/'strength_recall.csv')
    for r in rows(output/'strength_summary_30.csv'):
        group = {int(s['seed']):s for s in strata if float(s['noise_ratio'])==float(r['noise_ratio']) and s['method']==r['method']
                 and s['order']==r['order'] and s['strength']==r['strength']}
        values = np.array([float(group[s]['recall']) if group[s]['status']=='OK' and group[s]['recall']!='' else np.nan for s in c['seeds']])
        for key, value in independently_describe(values, indices).items():assert same(value, float(r[key]))
    for r in rows(output/'sensitivity_new_28.csv'):
        values = vector(float(r['noise_ratio']), r['method'], r['order'], r['metric'])[2:]
        finite = values[np.isfinite(values)]
        assert int(r['n_total']) == 28 and int(r['n_available']) == len(finite)
        assert same(float(finite.mean()) if len(finite) else np.nan, float(r['mean']))
    print('BOOTSTRAP_PAIRED_CIS_AND_NEW_28_SENSITIVITY_RECOMPUTED', flush=True)


def main():
    output = ROOT/'results/r5/e1_full_independent';c = tomllib.loads((output/'config_locked.toml').read_text());validate_full(c)
    base = tomllib.loads((output/'r4_base_locked.toml').read_text())
    completion = tomllib.loads((output/'completion.toml').read_text())
    assert completion['E1_full_completed'] and completion['seed_count'] == 30 and completion['trial_count'] == 480
    assert not completion['E2_executed'] and not completion['E3_executed'] and not completion['excluded_seeds']
    preflight = tomllib.loads((output/'preflight.toml').read_text())
    for name, expected in preflight['protected_sha256'].items():assert file_hash(ROOT/name) == expected, name
    old = output/'history/smoke_artifact_manifest.toml'
    assert file_hash(old) == preflight['previous_manifest_sha256']
    old_m = tomllib.loads(old.read_text())
    for name in preflight['mutable_docs_archived']:
        assert file_hash(output/'history'/Path(name).name) == old_m['file_sha256'][name]
    protocol = tomllib.loads((output/'protocol.toml').read_text())
    assert all(file_hash(ROOT/name) == expected for name, expected in protocol['source_sha256'].items())
    assert protocol['config_sha256'] == file_hash(ROOT/'configs/r5_e1_full_independent.toml') == file_hash(output/'config_locked.toml')
    assert protocol['base_sha256'] == file_hash(ROOT/c['base_config']) == file_hash(output/'r4_base_locked.toml')
    for repo in ('THIS', 'Su-AdaptiveLASSO'):
        assert not subprocess.check_output(['git', '-C', str(ROOT/'upstream'/repo), 'status', '--porcelain'], text=True).strip()
    print('PREVIOUS_ARTIFACTS_AND_ARCHIVED_SMOKE_REPORTS_UNCHANGED', flush=True)
    trials = rows(output/'trials.csv');strata = rows(output/'strength_recall.csv')
    assert len(trials)==480 and len(strata)==2880
    keys = {(int(r['seed']), float(r['noise_ratio']), r['method']) for r in trials}
    assert keys == {(s, n, m) for s in c['seeds'] for n in c['noise_ratios'] for m in c['method_order']}
    locks = tomllib.loads((output/'selection_locked.toml').read_text())
    assert len(locks['trials'])==480 and not locks['test_evaluated_before_lock'] and not locks['true_labels_used_for_selection']
    inventory = {key:r for key, r in zip([(int(r['seed']), float(r['noise_ratio']), r['method']) for r in trials], trials)}
    counts = Counter();warnings = [];pilot_max = 0.;seen_trajectory_seeds = set()
    true_hash = file_hash(ROOT/'data/r1/revision/true_coefficients.csv')
    for seed in c['seeds']:
        clean = load_dataset(output/'clean', seed)
        expected_streams = set(range(trajectory_seed_base(c, seed)+1, trajectory_seed_base(c, seed)+161))
        assert set(clean['metadata']['trajectory_rng_seeds']) == expected_streams
        assert not seen_trajectory_seeds & expected_streams
        seen_trajectory_seeds.update(expected_streams)
        assert file_hash(clean['directory']/'true_coefficients.csv') == true_hash
        truth = read_matrix(clean['directory']/'true_coefficients.csv', (8, 165))
        scales, unit, _ = response_noise_basis(clean['Y'], clean['splits'], seed, c['noise_master_seed'])
        np.testing.assert_array_equal(read_matrix(output/f'seed_{seed}/unit_noise.csv', (8, 480)), unit)
        np.testing.assert_array_equal(read_matrix(output/f'seed_{seed}/signal_scales.csv', (8, 1))[:, 0], scales)
        if seed in c['pilot_seeds']:
            prior_unit = read_matrix(ROOT/'results/r5/e1_smoke'/f'seed_{seed}'/'unit_noise.csv', (8, 480))
            pilot_max = max(pilot_max, float(np.max(np.abs(unit-prior_unit))))
            np.testing.assert_array_equal(unit, prior_unit)
        for ratio in c['noise_ratios']:
            slot = noise_slot(ratio);cell = output/f'seed_{seed}'/slot
            meta = tomllib.loads((cell/'data/metadata.toml').read_text())
            for name, expected in meta['clean_file_sha256'].items():assert file_hash(clean['directory']/name)==expected
            Y, eta = inject_response_noise(clean['Y'], scales, unit, ratio)
            np.testing.assert_array_equal(read_matrix(cell/'data/Y_noisy.csv', (8, 480)), Y)
            np.testing.assert_array_equal(read_matrix(cell/'data/eta.csv', (8, 480)), eta)
            for method in c['method_order']:
                trial = cell/method;status = tomllib.loads((trial/'status.toml').read_text());counts[status['status']] += 1
                lock = locks['trials'][f'{seed}_{slot}_{method}'];record = inventory[seed, ratio, method]
                assert record['status'] == status['status']
                assert file_hash(trial/'status.toml') == lock['status_sha256']
                assert status['noise_metadata_sha256'] == file_hash(cell/'data/metadata.toml')
                assert status['noisy_response_sha256'] == file_hash(cell/'data/Y_noisy.csv') == meta['Y_noisy_sha256']
                assert status['data_metadata_sha256'] == file_hash(clean['directory']/'metadata.toml')
                if status['status'] != 'OK':
                    assert not (trial/'coefficients.csv').exists() and record['triple_F1']=='' and status['reason']
                    if status['status']=='NA':assert method in ('M1', 'M2')
                    continue
                assert file_hash(trial/'coefficients.csv') == lock['coefficient_sha256']
                C = read_matrix(trial/'coefficients.csv', (8, 165))
                if method != 'M0':
                    Cz = read_matrix(trial/'standardized_coefficients.csv', (8, 165))
                    np.testing.assert_allclose(C, restore_coefficients(Cz, clean['means'], clean['scales']), atol=1e-13, rtol=1e-12)
                metrics = tomllib.loads((trial/'metrics.toml').read_text())
                assert metrics['selection_lock_sha256']==file_hash(output/'selection_locked.toml') and metrics['evaluated_at']>=locks['locked_at']
                decoded = rows(trial/'inferred_structure.csv')
                for order, name in ((2, 'pairwise'), (3, 'triple')):
                    candidates, scores = edge_scores(C, clean['d'], order);_, true_scores = edge_scores(truth, clean['d'], order)
                    expected = binary_metrics(true_scores>base['truth_threshold'], scores, base['decode_threshold'])
                    for metric in ('TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'F1', 'MCC', 'AUPRC_AP'):
                        assert same(expected[metric], metrics[name][metric]) and same(expected[metric], float(record[name+'_'+metric]))
                    predicted = {key for key, score in zip(candidates, scores) if score>base['decode_threshold']}
                    julia_keys = {(int(r['target']), *[int(r['source_'+str(i)]) for i in range(1, order)]) for r in decoded if int(r['order'])==order}
                    assert predicted == julia_keys
                    for label in ('weak', 'medium', 'strong'):
                        mask = np.array([v>1e-8 and true_strength_group(v)==label for v in true_scores])
                        sr = next(r for r in strata if int(r['seed'])==seed and float(r['noise_ratio'])==ratio and r['method']==method and r['order']==name and r['strength']==label)
                        tp = int(np.sum(scores[mask]>1e-8));total = int(mask.sum())
                        assert int(sr['true_count'])==total and int(sr['TP'])==tp and int(sr['FN'])==total-tp
                for split in ('train', 'validation', 'test'):
                    mask = clean['splits']==split;prediction = C@clean['theta'][:, mask]
                    assert same(float(np.mean((Y[:, mask]-prediction)**2)), metrics['regression'][split+'_MSE'])
                    assert same(float(np.mean((clean['Y'][:, mask]-prediction)**2)), metrics['regression'][split+'_clean_response_MSE'])
                if method != 'M0':
                    nodes = tomllib.loads((trial/'diagnostics.toml').read_text())['nodes']
                    for node in nodes:
                        assert node['status']=='OK' and node['kkt_scaled_violation']<=base['lasso']['kkt_tolerance']
                        if 'validation_path' in node:
                            selected = node['selected_path_index'];assert not node['validation_path'][selected]['convergence_warning']
                            for index, candidate in enumerate(node['validation_path']):
                                for warning in candidate['warnings']:
                                    warnings.append(dict(seed=seed, noise_ratio=ratio, method=method, node=node['node_julia_index'],
                                        alpha=candidate['alpha'], selected=int(index==selected), warning=warning))
                        else:
                            for warning in node['warnings']:
                                warnings.append(dict(seed=seed, noise_ratio=ratio, method=method, node=node['node_julia_index'],
                                    alpha=node['alpha'], selected=1, warning=warning))
        assert 'EXIT_CODE=0' in (output/'logs'/f'seed_{seed}.log').read_text()
        fitting = tomllib.loads((output/f'seed_{seed}/fitting_done.toml').read_text())
        assert not fitting['test_evaluated'] and fitting['fitting_completed_at'] <= locks['locked_at']
        print('FULL_SEED_AUDITED='+str(seed), flush=True)
    assert dict(counts)=={k:v for k,v in completion['counts'].items() if v}
    with (output/'candidate_warnings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['seed', 'noise_ratio', 'method', 'node', 'alpha', 'selected', 'warning'])
        writer.writeheader();writer.writerows(warnings)
    audit_statistics(output, c, trials)
    for name in ('triple_f1_ap', 'triple_fp_fn', 'strength_recall', 'paired_triple_differences'):
        for suffix in ('.png', '.pdf'):assert (output/'figures'/(name+suffix)).stat().st_size > 1000
    for filename in ('full_test_independent.log', 'full_run_independent.log'):assert 'EXIT_CODE=0' in (ROOT/'results/r5/logs'/filename).read_text()
    write_toml(output/'audit_results.toml', dict(verified_at=stamp(), all_480_records_verified=True, counts=dict(counts),
        only_Y_changed=True, original_sources_unchanged=True, global_choice_lock_before_test=True,
        bootstrap_10000_seed_clusters_verified=True, pilot_noise_unit_max_difference=pilot_max,
        independent_trajectory_seed_count=len(seen_trajectory_seeds),
        candidate_warning_count=len(warnings), selected_warning_count=sum(w['selected'] for w in warnings), E2_E3_executed=False))
    files = set()
    for pattern in ('configs/r5_*', 'scripts/r5_*', 'tests/test_r5_*', 'reports/R5_report.md', 'reports/R5_learning.md', 'docs/r5_followup_noise_plan.md'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    files.update(p for p in (ROOT/'results/r5').rglob('*') if p.is_file())
    manifests = [ROOT/'results/r5/artifact_manifest.toml', output/'artifact_manifest.toml']
    for path in manifests:files.discard(path)
    active = os.environ.get('R5_FULL_ACTIVE_LOG')
    if active:files.discard(ROOT/active)
    value = dict(verified_at=stamp(), E1_full_verified=True, E1_smoke_preserved=True, only_Y_changed=True,
        previous_smoke_manifest_sha256=preflight['previous_manifest_sha256'], counts=dict(counts), E2_executed=False, E3_executed=False,
        file_count=len(files), excluded_self_referential_files=[str(p.relative_to(ROOT)) for p in manifests]+([active] if active else []),
        file_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)})
    for path in manifests:write_toml(path, value)
    for name in ('reports/R5_report.md', 'reports/R5_learning.md', 'docs/r5_followup_noise_plan.md'):
        p = ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' not in target and not target.startswith('#'):assert (p.parent/target.split('#')[0]).exists(), (name, target)
    print('R5_E1_FULL_AUDIT_PASS counts='+str(dict(counts))+' candidate_warnings='+str(len(warnings)), flush=True)


if __name__ == '__main__':main()
