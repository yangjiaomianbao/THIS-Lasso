"""Verify R3 records and stage boundaries without rerunning a solver."""
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
from src.data_bridge import file_hash, load_request, read_matrix, write_toml


def main():
    preflight = tomllib.loads((ROOT/'results/r3/preflight.toml').read_text())
    for name, expected in preflight['protected_sha256'].items():
        assert file_hash(ROOT/name) == expected, name
    print(f'PROTECTED_FILES_UNCHANGED={len(preflight["protected_sha256"])}')
    r1 = tomllib.loads((ROOT/'results/r1/artifact_manifest.toml').read_text())
    for name, expected in r1['file_sha256'].items():
        assert file_hash(ROOT/name) == expected, name
    print(f'R1_MANIFEST_UNCHANGED={len(r1["file_sha256"])}')
    locks = json.loads((ROOT/'environment/r0/upstream_lock.json').read_text())
    count = 0
    for name, lock in locks.items():
        repo = ROOT/'upstream'/name
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == lock['head']
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
        for filename, expected in lock['tracked_file_sha256'].items():
            assert file_hash(repo/filename) == expected
            count += 1
    print(f'UPSTREAM_HASHES_MATCH={count}; BOTH_REPOSITORIES_CLEAN')
    protocol = tomllib.loads((ROOT/'results/r3/protocol.toml').read_text())
    assert not protocol['test_evaluated'] and not protocol['structural_truth_used_for_parameter_selection']
    reference = ROOT/'data/r1/revision'
    metadata = tomllib.loads((reference/'metadata.toml').read_text())
    assert protocol['data_metadata_sha256'] == file_hash(reference/'metadata.toml')
    theta = read_matrix(reference/'theta.csv', metadata['theta_shape'])
    Y = read_matrix(reference/'derivatives.csv', metadata['state_shape'])
    samples = [line.split(',') for line in (reference/'sample_manifest.csv').read_text().splitlines()[1:]]
    train = np.array([row[2] == 'train' for row in samples])
    val = np.array([row[2] == 'validation' for row in samples])
    results = {}
    for solver in ('stlsq', 'lasso', 'adaptive_lasso'):
        directory = ROOT/'results/r3'/solver
        C = read_matrix(directory/'coefficients.csv', (8, 165))
        results[solver] = C
        metrics = tomllib.loads((directory/'metrics.toml').read_text())
        for key, mask in [('train_MSE', train), ('validation_MSE', val)]:
            observed = np.mean((Y[:, mask]-C@theta[:, mask])**2)
            assert np.isclose(observed, metrics[key], rtol=1e-7, atol=1e-28), (solver, key)
        if solver == 'stlsq':
            saved = read_matrix(ROOT/'results/r1/revision/selected/coefficients.csv', (8, 165))
            np.testing.assert_array_equal(C, saved)
        else:
            request = load_request(directory)
            np.testing.assert_array_equal(request['theta'], theta[:, train])
            np.testing.assert_array_equal(request['Y'], Y[:, train])
            np.testing.assert_array_equal(request['validation']['X_validation'], theta[:, val].T)
            np.testing.assert_array_equal(request['validation']['Y_validation'], Y[:, val])
            response = tomllib.loads((directory/'response.toml').read_text())
            assert response['request_metadata_sha256'] == file_hash(directory/'metadata.toml')
            assert response['feature_order_hash'] == metadata['feature_order_hash']
            for name, expected in response['output_sha256'].items():
                assert file_hash(directory/name) == expected
            nodes = tomllib.loads((directory/'diagnostics.toml').read_text())['nodes']
            assert len(nodes) == 8
            for node, diagnostics in enumerate(nodes):
                assert diagnostics['node_julia_index'] == node+1 and diagnostics['converged']
                assert diagnostics['intercept'] == 0 and diagnostics['kkt_scaled_violation'] <= 1e-6
                path = diagnostics['validation_path']
                assert len(path) == 12
                selected = path[diagnostics['selected_path_index']]
                assert selected['alpha'] == diagnostics['alpha']
                score = np.mean((C[node]@theta[:, val]-Y[node, val])**2)
                assert np.isclose(score, selected['MSE'], rtol=1e-6, atol=1e-28)
        print(f'{solver}: COEFFICIENT_SHAPE={C.shape}; TRAIN_VALIDATION_AND_RECORDS_VERIFIED')
    trace = tomllib.loads((ROOT/'results/r3/feature_trace.toml').read_text())
    feature = trace['feature_julia']-1
    assert feature == trace['feature_python'] and sorted(x for x in trace['d_row'] if x) == [5, 6]
    for solver, C in results.items():
        assert trace[solver]['coefficient'] == C[6, feature]
        assert trace[solver]['decoded_Ainf_row'][:3] == [7., 5., 6.]
    print('FEATURE_TRACE_VERIFIED: Julia feature '+str(feature+1)+' -> Python '+str(feature)+' -> 7 <- {5,6}')
    for filename in ('python_test_final.log', 'julia_test_final.log', 'demo.log'):
        log = (ROOT/'results/r3/logs'/filename).read_text()
        assert 'EXIT_CODE=0' in log and 'COMMAND=' in log and 'ENV=' in log
    print('SUCCESSFUL_TEST_AND_DEMO_LOGS_VERIFIED')
    paths = set()
    for pattern in ('src/this_adapter.jl', 'src/data_bridge.py', 'scripts/r3_*', 'tests/test_r3_*',
                    'reports/R3_report.md', 'reports/R3_learning.md'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    paths.update(p for p in (ROOT/'results/r3').rglob('*') if p.is_file())
    manifest_path = ROOT/'results/r3/artifact_manifest.toml'
    active_log = os.environ.get('R3_ACTIVE_LOG')
    paths.discard(manifest_path)
    if active_log:
        paths.discard(ROOT/active_log)
    manifest = dict(verified_at=datetime.datetime.now().astimezone().isoformat(),
        protected_files_unchanged=True, r1_manifest_unchanged=True, upstream_hashes_match=count,
        shared_train_validation_inputs_exact=True, stlsq_native_coefficients_exact=True,
        feature_trace_verified=True, no_test_evaluation=True, no_r4_execution=True,
        excluded_self_referential_files=['results/r3/artifact_manifest.toml']+([active_log] if active_log else []),
        file_count=len(paths), file_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)})
    write_toml(manifest_path, manifest)
    for filename in ('reports/R3_report.md', 'reports/R3_learning.md'):
        p = ROOT/filename
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (p.parent/target.split('#')[0]).exists(), (filename, target)
    print('REPORT_LINKS_VERIFIED')
    print('R3_FINAL_AUDIT_PASS; ARTIFACT_MANIFEST_WRITTEN')


if __name__ == '__main__':
    main()
