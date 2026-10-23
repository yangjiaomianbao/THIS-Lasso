"""Audit E1-only injection, paired inputs, locks, scores, strata and failures."""
import csv
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
from r4_experiment import load_dataset, stamp, write_rows
from r4_metrics import binary_metrics, edge_scores
from r5_noise import validate_config, noise_slot, response_noise_basis, inject_response_noise, true_strength_group


def same(a, b):
    return bool((np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-9, atol=1e-26))


def main():
    output = ROOT/'results/r5/e1_smoke'
    c = tomllib.loads((output/'config_locked.toml').read_text());validate_config(c)
    base = tomllib.loads((output/'r4_base_locked.toml').read_text())
    completion = tomllib.loads((output/'completion.toml').read_text())
    assert completion['E1_smoke_completed'] and completion['trial_count'] == 32 and not completion['excluded_seeds']
    assert not any(completion[key] for key in ('full_30_seed_executed', 'E2_executed', 'E3_executed'))
    assert not (output/'failure.toml').exists()
    preflight = tomllib.loads((output/'preflight.toml').read_text())
    for name, expected in preflight['protected_sha256'].items():
        assert file_hash(ROOT/name) == expected, name
    print('PRIOR_STAGE_FILES_UNCHANGED='+str(len(preflight['protected_sha256'])))
    for name, lock in json.loads((ROOT/'environment/r0/upstream_lock.json').read_text()).items():
        repo = ROOT/'upstream'/name
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == lock['head']
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    protocol = tomllib.loads((output/'protocol.toml').read_text())
    assert not protocol['R4_phase_B_completed'] and protocol['user_explicitly_requested_R5']
    assert all(file_hash(ROOT/name) == expected for name, expected in protocol['source_sha256'].items())
    assert file_hash(ROOT/'configs/r5_e1_smoke.toml') == file_hash(output/'config_locked.toml') == protocol['config_sha256']
    assert file_hash(ROOT/c['base_config']) == file_hash(output/'r4_base_locked.toml') == protocol['base_sha256']
    locks = tomllib.loads((output/'selection_locked.toml').read_text())
    assert len(locks['trials']) == 32 and not locks['test_evaluated_before_lock'] and not locks['true_labels_used_for_selection']
    rows = list(csv.DictReader((output/'trials.csv').open()))
    strata = list(csv.DictReader((output/'strength_recall.csv').open()))
    assert len(rows) == 32 and len(strata) == 192
    counts = dict(OK=0, NA=0, FAILED=0);warning_rows, weak_diagnostics = [], []
    assert {p.name for p in output.glob('seed_*') if p.is_dir()} == {f'seed_{s}' for s in c['seeds']}
    for seed in c['seeds']:
        clean = load_dataset(ROOT/c['source_root'], seed)
        scales, unit, streams = response_noise_basis(clean['Y'], clean['splits'], seed, c['noise_master_seed'])
        np.testing.assert_array_equal(read_matrix(output/f'seed_{seed}/unit_noise.csv', clean['Y'].shape), unit)
        np.testing.assert_array_equal(read_matrix(output/f'seed_{seed}/signal_scales.csv', (8, 1))[:, 0], scales)
        truth = read_matrix(clean['directory']/'true_coefficients.csv', (8, 165))
        for ratio in c['noise_ratios']:
            slot = noise_slot(ratio);cell = output/f'seed_{seed}'/slot
            m = tomllib.loads((cell/'data/metadata.toml').read_text())
            assert m['experiment'] == 'E1' and m['noise_ratio'] == ratio
            for name, expected in m['clean_file_sha256'].items():
                assert file_hash(clean['directory']/name) == expected
            expected_Y, expected_eta = inject_response_noise(clean['Y'], scales, unit, ratio)
            Y = read_matrix(cell/'data/Y_noisy.csv', clean['Y'].shape)
            eta = read_matrix(cell/'data/eta.csv', clean['Y'].shape)
            np.testing.assert_array_equal(Y, expected_Y);np.testing.assert_array_equal(eta, expected_eta)
            assert file_hash(cell/'data/Y_noisy.csv') == m['Y_noisy_sha256']
            assert file_hash(cell/'data/eta.csv') == m['eta_sha256']
            print(f'E1_ONLY_VERIFIED seed={seed} ratio={ratio:.2f}: states/features/truth unchanged; Y noise exactly reproducible')
            for method in c['method_order']:
                trial = cell/method;status = tomllib.loads((trial/'status.toml').read_text())
                counts[status['status']] += 1
                lock = locks['trials'][f'{seed}_{slot}_{method}']
                assert file_hash(trial/'status.toml') == lock['status_sha256']
                assert status['noisy_response_sha256'] == m['Y_noisy_sha256']
                assert status['noise_metadata_sha256'] == file_hash(cell/'data/metadata.toml')
                assert status['data_metadata_sha256'] == file_hash(clean['directory']/'metadata.toml')
                row = next(r for r in rows if int(r['seed']) == seed and float(r['noise_ratio']) == ratio and r['method'] == method)
                assert row['status'] == status['status']
                if status['status'] != 'OK':
                    assert not (trial/'coefficients.csv').exists() and row['triple_F1'] == '' and status['reason']
                    continue
                assert file_hash(trial/'coefficients.csv') == lock['coefficient_sha256']
                C = read_matrix(trial/'coefficients.csv', (8, 165))
                node_diagnostics = (tomllib.loads((trial/'diagnostics.toml').read_text())['nodes'][2]
                                    if (trial/'diagnostics.toml').exists() else {})
                weak_diagnostics.append(dict(seed=seed, noise_ratio=ratio, method=method,
                    target=3, sources='1,2', feature_julia=12,
                    true_coefficient=float(truth[2, 11]), restored_coefficient=float(C[2, 11]),
                    true_standardized_coefficient=float(truth[2, 11]*clean['scales'][11]),
                    initial_standardized_coefficient=node_diagnostics.get('initial_coef', ['']*165)[11],
                    adaptive_weight=node_diagnostics.get('weights', ['']*165)[11],
                    alpha=node_diagnostics.get('alpha', '')))
                if ratio == 0 and method in ('M0', 'M3'):
                    reference = ROOT/c['source_root']/f'seed_{seed}'/method/'coefficients.csv'
                    np.testing.assert_allclose(C, read_matrix(reference, C.shape), atol=1e-12, rtol=1e-12)
                metrics = tomllib.loads((trial/'metrics.toml').read_text())
                assert metrics['selection_lock_sha256'] == file_hash(output/'selection_locked.toml')
                assert metrics['evaluated_at'] >= locks['locked_at']
                for name, order in [('pairwise', 2), ('triple', 3)]:
                    keys, scores = edge_scores(C, clean['d'], order)
                    _, true_scores = edge_scores(truth, clean['d'], order)
                    values = binary_metrics(true_scores > base['truth_threshold'], scores, base['decode_threshold'])
                    for metric in ('TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'F1', 'MCC', 'AUPRC_AP'):
                        assert same(values[metric], metrics[name][metric])
                        assert same(values[metric], float(row[name+'_'+metric]))
                    for strength in ('weak', 'medium', 'strong'):
                        mask = np.array([v > base['truth_threshold'] and true_strength_group(v) == strength for v in true_scores])
                        record = next(r for r in strata if int(r['seed']) == seed and float(r['noise_ratio']) == ratio
                                      and r['method'] == method and r['order'] == name and r['strength'] == strength)
                        tp = int(np.sum(scores[mask] > base['decode_threshold']));total = int(mask.sum())
                        assert int(record['true_count']) == total and int(record['TP']) == tp and int(record['FN']) == total-tp
                        assert same(float(record['recall']), tp/total)
                for split in ('train', 'validation', 'test'):
                    mask = clean['splits'] == split;prediction = C@clean['theta'][:, mask]
                    assert same(float(np.mean((Y[:, mask]-prediction)**2)), metrics['regression'][split+'_MSE'])
                    assert same(float(np.mean((clean['Y'][:, mask]-prediction)**2)), metrics['regression'][split+'_clean_response_MSE'])
                if (trial/'diagnostics.toml').exists():
                    nodes = tomllib.loads((trial/'diagnostics.toml').read_text())['nodes']
                    for node in nodes:
                        assert node['status'] == 'OK' and node['kkt_scaled_violation'] <= base['lasso']['kkt_tolerance']
                        if 'validation_path' in node:
                            chosen = node['selected_path_index']
                            assert not node['validation_path'][chosen]['convergence_warning']
                            for index, candidate in enumerate(node['validation_path']):
                                for warning in candidate['warnings']:
                                    warning_rows.append(dict(seed=seed, ratio=ratio, method=method,
                                        node=node['node_julia_index'], alpha=candidate['alpha'], selected=int(index == chosen), warning=warning))
                        else:
                            for warning in node['warnings']:
                                warning_rows.append(dict(seed=seed, ratio=ratio, method=method,
                                    node=node['node_julia_index'], alpha=node['alpha'], selected=1, warning=warning))
    assert counts == completion['counts']
    warning_path = ROOT/'results/r5/candidate_warnings.csv'
    with warning_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, ['seed', 'ratio', 'method', 'node', 'alpha', 'selected', 'warning'])
        writer.writeheader();writer.writerows(warning_rows)
    write_rows(ROOT/'results/r5/weak_edge_diagnostics.csv', weak_diagnostics)
    print('STATUS_COUNTS='+str(counts)+'; WARNING_ROWS='+str(len(warning_rows)))
    for filename in ('test.log', 'smoke.log'):
        assert 'EXIT_CODE=0' in (ROOT/'results/r5/logs'/filename).read_text()
    files = set()
    for pattern in ('configs/r5_*', 'scripts/r5_*', 'tests/test_r5_*', 'reports/R5_report.md',
                    'reports/R5_learning.md', 'docs/r5_followup_noise_plan.md'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    files.update(p for p in (ROOT/'results/r5').rglob('*') if p.is_file())
    manifest = ROOT/'results/r5/artifact_manifest.toml';files.discard(manifest)
    active = os.environ.get('R5_ACTIVE_LOG')
    if active:
        files.discard(ROOT/active)
    write_toml(manifest, dict(verified_at=stamp(), E1_smoke_verified=True, only_Y_changed=True,
        prior_files_unchanged=True, full_30_seed_executed=False, E2_executed=False, E3_executed=False,
        counts=counts, file_count=len(files), excluded_self_referential_files=['results/r5/artifact_manifest.toml']+([active] if active else []),
        file_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)}))
    for filename in ('reports/R5_report.md', 'reports/R5_learning.md', 'docs/r5_followup_noise_plan.md'):
        path = ROOT/filename
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).exists(), (filename, target)
    print('R5_FINAL_AUDIT_PASS; E1_SMOKE_ONLY; NO_BATCH_E2_E3')


if __name__ == '__main__':
    main()
