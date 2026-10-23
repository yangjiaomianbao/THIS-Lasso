"""Audit phase-A data, locks, NA handling, metrics, and prior-stage immutability."""
import csv
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_bridge import file_hash, read_matrix, write_toml
from r4_experiment import load_dataset
from r4_metrics import binary_metrics, edge_scores, restore_coefficients, validate_phase, validate_splits


def equal_number(a, b):
    return bool((np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-10, atol=1e-28))


def main():
    output = ROOT/'results/r4/dry_run'
    c = tomllib.loads((output/'config_locked.toml').read_text());validate_phase(c)
    done = tomllib.loads((output/'completion.toml').read_text())
    assert done['phase_A_completed'] and not done['phase_B_executed'] and not done['r5_executed']
    assert done['seed_count'] == 2 and done['trial_count'] == 12 and not done['excluded_failed_seeds']
    assert not (output/'failure.toml').exists()
    preflight = tomllib.loads((output/'preflight.toml').read_text())
    for name, expected in preflight['protected_sha256'].items():
        assert file_hash(ROOT/name) == expected, name
    print('PRIOR_STAGE_FILES_UNCHANGED='+str(len(preflight['protected_sha256'])))
    for name, lock in json.loads((ROOT/'environment/r0/upstream_lock.json').read_text()).items():
        repo = ROOT/'upstream'/name
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == lock['head']
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
        assert all(file_hash(repo/filename) == expected for filename, expected in lock['tracked_file_sha256'].items())
    protocol = tomllib.loads((output/'protocol.toml').read_text())
    assert file_hash(ROOT/'configs/r4_dry_run.toml') == file_hash(output/'config_locked.toml') == protocol['config_sha256']
    assert all(file_hash(ROOT/name) == expected for name, expected in protocol['source_sha256'].items())
    locks = tomllib.loads((output/'selection_locked.toml').read_text())
    assert not locks['test_evaluated_before_lock'] and not locks['structural_labels_used_for_selection']
    assert len(locks['trials']) == 12
    rows = list(csv.DictReader((output/'trials.csv').open()))
    assert len(rows) == 12
    counts = dict(OK=0, NA=0)
    warning_rows = []
    assert {p.name for p in output.glob('seed_*') if p.is_dir()} == {f'seed_{seed}' for seed in c['seeds']}
    for seed in c['seeds']:
        data = load_dataset(output, seed)
        masks = {name:data['splits'] == name for name in ('train', 'validation', 'test')}
        assert validate_splits(data['groups'], data['splits']) == dict(train=100, validation=30, test=30)
        assert [int(mask.sum()) for mask in masks.values()] == [300, 90, 90]
        mu = data['theta'][:, masks['train']].mean(axis=1);mu[0] = 0.
        scales = np.sqrt(np.mean((data['theta'][:, masks['train']]-mu[:, None])**2, axis=1));scales[0] = 1.
        np.testing.assert_allclose(data['means'], mu, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(data['scales'], scales, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(data['Z'], (data['theta']-mu[:, None])/scales[:, None], atol=1e-13, rtol=1e-12)
        p = data['metadata']['P'];n = c['nodes']
        truth = read_matrix(data['directory']/'true_coefficients.csv', (n, p))
        np.testing.assert_array_equal(truth, np.loadtxt(ROOT/'data/r1/revision/true_coefficients.csv', delimiter=','))
        assert data['metadata']['feature_order_hash'] == file_hash(ROOT/'data/r1/revision/monomial_indices.csv')
        print(f'SEED={seed} SHARED_INPUTS_AND_TRAIN_ONLY_TRANSFORM_VERIFIED rank={data["metadata"]["raw_train_rank"]} condition={data["metadata"]["raw_train_condition"]:.3f}')
        for method in c['method_order']:
            trial = output/f'seed_{seed}'/method
            status = tomllib.loads((trial/'status.toml').read_text())
            key = f'{seed}_{method}';lock = locks['trials'][key]
            assert file_hash(trial/'status.toml') == lock['status_sha256']
            assert status['data_metadata_sha256'] == file_hash(data['directory']/'metadata.toml')
            counts[status['status']] += 1
            row = next(row for row in rows if int(row['seed']) == seed and row['method'] == method)
            assert row['status'] == status['status'] == lock['status']
            if status['status'] == 'NA':
                assert method in ('M1', 'M2') and lock['coefficient_sha256'] == 'NA'
                assert not (trial/'coefficients.csv').exists() and row['triple_F1'] == ''
                nodes = tomllib.loads((trial/'diagnostics.toml').read_text())['nodes']
                assert len(nodes) == n and any(node['status'] == 'NA' and node['reason'] for node in nodes)
                print(f'NA_RETAINED seed={seed} {method} unavailable_nodes={status["unavailable_nodes"]}')
                continue
            C = read_matrix(trial/'coefficients.csv', (n, p))
            assert file_hash(trial/'coefficients.csv') == lock['coefficient_sha256']
            if method != 'M0':
                Cz = read_matrix(trial/'standardized_coefficients.csv', (n, p))
                np.testing.assert_allclose(C, restore_coefficients(Cz, data['means'], data['scales']), atol=1e-14, rtol=1e-13)
            if method in ('M3', 'M1v'):
                nodes = tomllib.loads((trial/'diagnostics.toml').read_text())['nodes']
                for node in nodes:
                    selected = node['selected_path_index']
                    assert not node['validation_path'][selected]['convergence_warning']
                    assert node['kkt_scaled_violation'] <= c['lasso']['kkt_tolerance']
                    for index, candidate in enumerate(node['validation_path']):
                        if candidate['warnings']:
                            warning_rows.append(dict(seed=seed, method=method, node=node['node_julia_index'],
                                alpha=candidate['alpha'], selected=int(index == selected),
                                warning=' | '.join(candidate['warnings'])))
            metrics = tomllib.loads((trial/'metrics.toml').read_text())
            assert metrics['evaluated_at'] >= locks['locked_at']
            assert metrics['selection_lock_sha256'] == file_hash(output/'selection_locked.toml')
            edges = list(csv.DictReader((trial/'edge_scores.csv').open()))
            for name, order in [('pairwise', 2), ('triple', 3)]:
                keys, scores = edge_scores(C, data['d'], order)
                _, truth_scores = edge_scores(truth, data['d'], order)
                expected = binary_metrics(truth_scores > c['truth_threshold'], scores, c['decode_threshold'])
                for metric in ('TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'F1', 'MCC', 'AUPRC_AP'):
                    assert equal_number(expected[metric], metrics[name][metric]), (key, name, metric)
                    assert equal_number(expected[metric], float(row[name+'_'+metric]))
                recorded = [edge for edge in edges if int(edge['order']) == order]
                assert len(recorded) == len(keys) == (56 if order == 2 else 168)
                assert sum(int(edge['truth']) for edge in recorded) == (10 if order == 2 else 3)
            support = list(csv.DictReader((trial/'taylor_support.csv').open()))
            assert len(support) == n*p == 1320
            expected = binary_metrics((np.abs(truth) > c['truth_threshold']).ravel(), np.abs(C).ravel(), c['decode_threshold'])
            assert all(equal_number(expected[k], metrics['taylor_support'][k]) for k in ('TP', 'FP', 'FN', 'TN', 'F1', 'MCC', 'AUPRC_AP'))
            for name, mask in masks.items():
                actual = float(np.mean((data['Y'][:, mask]-C@data['theta'][:, mask])**2))
                assert equal_number(actual, metrics['regression'][name+'_MSE'])
    assert counts['OK'] == done['OK_count'] and counts['NA'] == done['NA_count']
    print('TRIAL_COUNTS='+str(counts)+'; METRICS_AND_LOCKS_VERIFIED; NO_FAILED_SEED_EXCLUDED')
    with (ROOT/'results/r4/candidate_warnings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['seed', 'method', 'node', 'alpha', 'selected', 'warning'])
        writer.writeheader();writer.writerows(warning_rows)
    print('CANDIDATE_WARNINGS_RETAINED='+str(len(warning_rows))+'; SELECTED_CANDIDATE_WARNINGS=0')
    for filename in ('test.log', 'dry_run.log'):
        assert 'EXIT_CODE=0' in (ROOT/'results/r4/logs'/filename).read_text()
    files = set()
    for pattern in ('configs/r4_*', 'scripts/r4_*', 'tests/test_r4_*', 'reports/R4_report.md', 'reports/R4_learning.md'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    files.update(p for p in (ROOT/'results/r4').rglob('*') if p.is_file())
    manifest = ROOT/'results/r4/artifact_manifest.toml';files.discard(manifest)
    active = os.environ.get('R4_ACTIVE_LOG')
    if active:
        files.discard(ROOT/active)
    write_toml(manifest, dict(verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        phase_A_verified=True, phase_B_not_executed=True, prior_files_unchanged=True, no_R5=True,
        counts=counts, file_count=len(files),
        excluded_self_referential_files=['results/r4/artifact_manifest.toml']+([active] if active else []),
        file_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)}))
    for filename in ('R4_report.md', 'R4_learning.md'):
        path = ROOT/'reports'/filename
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).exists(), (filename, target)
    print('R4_FINAL_AUDIT_PASS; PHASE_A_ONLY; WAIT_FOR_PHASE_B_AUTHORIZATION')


if __name__ == '__main__':
    main()
