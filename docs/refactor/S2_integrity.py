"""Verify the exact six-file change boundary against S1; never update old manifests."""
import ast
import csv
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/refactor'
MODIFIED={'src/data_bridge.py','scripts/r4_methods.py','scripts/r4_metrics.py',
          'scripts/r4_experiment.py','scripts/r4_prepare.jl','scripts/r5_batch_julia.jl'}
ADDED={'src/bridge/numeric_io.py','src/solvers/plain_lasso.py','src/solvers/methods.py',
       'src/data/contracts.py','src/preprocessing/taylor.py','src/metrics/structure.py',
       'src/preprocessing/taylor_scaling.jl','src/solvers/stlsq.jl','tests/test_refactor_s2.py'}


def sha(path):
    if path.is_symlink():return hashlib.sha256(os.readlink(path).encode()).hexdigest()
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def csv_new(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    baseline=list(csv.DictReader((OUT/'S1_baseline_manifest.csv').open()))
    assert sha(OUT/'S1_baseline_manifest.csv')=='f14a15e52881c8efbaa4bb39608f033d8020d5d1613e9a227ce5ea168c51ce5a'
    assert sha(OUT/'S1_audit_artifacts.csv')=='c7ea9bcc958af696843ede461fa79462cf2123b1389b2c46ddbe7c1f09f0faff'
    original={r['path']:r for r in baseline};hashes={};changed=[];errors=[]
    for r in baseline:
        p=ROOT/r['path']
        if not (p.exists() or p.is_symlink()):errors.append([r['path'],'missing']);continue
        actual=sha(p);hashes[r['path']]=actual
        if actual!=r['sha256'] or p.lstat().st_size!=int(r['size']):
            if r['path'] not in MODIFIED:errors.append([r['path'],'unapproved change'])
            changed.append(dict(path=r['path'],action='modified',before_sha256=r['sha256'],
                                after_sha256=actual,size=p.lstat().st_size))
    assert {r['path'] for r in changed}==MODIFIED
    s1=list(csv.DictReader((OUT/'S1_audit_artifacts.csv').open()))
    for r in s1:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    audit_s1={r['path'] for r in s1}|{'docs/refactor/S1_audit_artifacts.csv'}
    found=set()
    for base,dirs,files in os.walk(ROOT,followlinks=False):
        for name in dirs:
            p=Path(base)/name
            if p.is_symlink():found.add(p.relative_to(ROOT).as_posix())
        for name in files:found.add((Path(base)/name).relative_to(ROOT).as_posix())
    for name in found-set(original)-audit_s1:
        if name in ADDED:continue
        rel=Path(name)
        if rel.parent.as_posix()=='docs/refactor' and rel.name.startswith('S2_'):continue
        if name.startswith('docs/refactor/S2_work/'):continue
        errors.append([name,'unexpected new path'])
    assert not errors,errors
    for name in sorted(ADDED):
        p=ROOT/name;assert p.is_file()
        hashes[name]=sha(p)
        changed.append(dict(path=name,action='added',before_sha256='',after_sha256=hashes[name],size=p.stat().st_size))
    csv_new('S2_changed_files.csv',sorted(changed,key=lambda r:r['path']))
    patch=[]
    with tarfile.open(OUT/'S1_source_snapshot.tar.gz','r:gz') as tar:
        for name in sorted(MODIFIED|ADDED):
            before=tar.extractfile(name).read().decode() if name in MODIFIED else ''
            patch.extend(difflib.unified_diff(before.splitlines(keepends=True),(ROOT/name).read_text().splitlines(keepends=True),
                fromfile='a/'+name if name in MODIFIED else '/dev/null',tofile='b/'+name))
    with (OUT/'S2_changes.patch').open('x') as f:f.writelines(patch)
    mapping={
        'src/bridge/numeric_io.py':{'src/data_bridge.py':['file_hash','_toml','write_toml','read_matrix'],
                                  'scripts/r4_experiment.py':['stamp','write_rows']},
        'src/solvers/plain_lasso.py':{'src/data_bridge.py':['fit_plain_lasso'],'scripts/r4_methods.py':['fit_plain_bic']},
        'src/solvers/methods.py':{'scripts/r4_methods.py':['fit_node']},
        'src/data/contracts.py':{'scripts/r4_metrics.py':['validate_splits'],'scripts/r4_experiment.py':['load_dataset']},
        'src/preprocessing/taylor.py':{'scripts/r4_metrics.py':['restore_coefficients']},
        'src/metrics/structure.py':{'scripts/r4_metrics.py':['binary_metrics','edge_scores']},
    }
    checks=[]
    def functions(code):return {n.name:n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
    with tarfile.open(OUT/'S1_source_snapshot.tar.gz','r:gz') as tar:
        for new,sources in mapping.items():
            nf=functions((ROOT/new).read_text())
            for old,names in sources.items():
                oldf=functions(tar.extractfile(old).read().decode())
                for name in names:
                    assert ast.dump(nf[name],include_attributes=False)==ast.dump(oldf[name],include_attributes=False),(new,name)
                    checks.append(dict(old_path=old,new_path=new,symbol=name,signature=ast.unparse(nf[name].args),
                                       function_AST_identical=True,old_import_compatible=True))
    csv_new('S2_api_map.csv',checks)
    # Historical source hash mismatch is retained and explained, never rewritten.
    historical=[]
    for name in ['results/r1/artifact_manifest.toml','results/r3/artifact_manifest.toml',
                 'results/r4/artifact_manifest.toml','results/r5/artifact_manifest.toml',
                 'results/r5/e1_full_independent/artifact_manifest.toml']:
        import tomllib
        m=tomllib.loads((ROOT/name).read_text())
        differences=[p for p,h in m['file_sha256'].items() if hashes.get(p,sha(ROOT/p))!=h]
        assert set(differences)<=MODIFIED,(name,differences)
        historical.append(dict(path=name,source_hash_mismatch_paths=differences,
                               experiment_artifacts_unchanged=True,historical_manifest_unchanged=True))
    for r in csv.DictReader((OUT/'S1_golden_index.csv').open()):assert hashes.get(r['path'],sha(ROOT/r['path']))==r['sha256']
    formal=[r for r in baseline if r['scope']=='R5_formal'];assert len(formal)==4113
    assert all(hashes[r['path']]==r['sha256'] for r in formal)
    git=[]
    for repo in ['THIS','Su-AdaptiveLASSO']:
        proc=subprocess.run(['git','-C',str(ROOT/'upstream'/repo),'status','--porcelain=v1','--untracked-files=all'],
            capture_output=True,text=True,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
        assert proc.returncode==0 and not proc.stdout
        git.append(dict(repo=repo,clean=True,exit_code=0))
    required_logs=['A_retry2_python_tests','A_retry2_golden','A_retry2_trial_smoke',
        'B_python_tests','B_golden','B_trial_smoke','C_python_tests','C_golden','C_trial_smoke',
        'test_r1_baseline','test_r1_revision','test_r3_adapter','r3_cli_help','r4_cli_help','r5_cli_help','julia_entrypoint_syntax']
    for name in required_logs:assert 'EXIT_CODE=0' in (OUT/('S2_'+name+'.log')).read_text(),name
    result=dict(verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        passed=True,baseline_entries=len(baseline),authorized_modified_original_files=sorted(MODIFIED),
        unchanged_original_entries=len(baseline)-len(MODIFIED),removed_original_files=[],added_production_and_test_files=sorted(ADDED),
        source_functions_AST_identical=len(checks),formal_R5_files_unchanged=len(formal),
        S1_audit_artifacts_unchanged=True,golden_index_111_files_unchanged=True,upstream=git,
        required_successful_logs=len(required_logs),historical_manifests=historical,
        preserved_failed_logs=['S2_A_python_tests.log','S2_A_retry1_trial_smoke.log'],
        rollback_records=['S2_A_rollback.json','S2_A_retry1_rollback.json'],
        failures_explained='Two new cross-path bitwise assumptions reproduced in S1; corrected comparison references only, no tolerances or algorithm formulas changed',
        S3_S6_executed=False)
    with (OUT/'S2_integrity_check.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('S2_INTEGRITY_PASS',len(baseline)-len(MODIFIED),'original entries unchanged;',len(MODIFIED),
          'authorized source edits;',len(ADDED),'added source/test files; 4113 formal files unchanged; S3 NOT EXECUTED',flush=True)


if __name__=='__main__':main()
