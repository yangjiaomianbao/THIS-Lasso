"""S1 audit runner: existing files are read-only; all writes stay in docs/refactor/S1_*.

No legacy audit/experiment main is invoked. This is a baseline audit artifact,
not a replacement for the project's algorithms or an S2 implementation.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import tomllib
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/refactor'
FORMAL = ROOT / 'results/r5/e1_full_independent'
WORK = OUT / 'S1_work'
SEED = 22261035
JULIA = Path('/home/yg/.local/opt/julia-1.9.4/bin/julia')
PYTHON = ROOT / '.venv-r0/bin/python'


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def sha(path):
    path = Path(path)
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode()).hexdigest()
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def json_new(path, value):
    assert Path(path).is_relative_to(OUT)
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def csv_new(path, rows, fields=None):
    assert Path(path).is_relative_to(OUT)
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def existing_paths():
    found = []
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        # Exclude only this stage's new audit artifacts, not S0 or other docs.
        dirs[:] = [d for d in dirs if not
                   (Path(base) == OUT and d.startswith('S1_'))]
        for name in dirs:
            p = Path(base)/name
            if p.is_symlink():
                found.append(p)
        for name in files:
            p = Path(base)/name
            if p.parent == OUT and name.startswith('S1_'):
                continue
            found.append(p)
    return sorted(found)


def run(command, label):
    log = OUT / ('S1_'+label+'.log')
    env = os.environ.copy()
    temporary = WORK/'tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    depot = WORK/'julia_depot'
    depot.mkdir(exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', JULIA_NUM_THREADS='1', GIT_OPTIONAL_LOCKS='0',
               TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
               MPLCONFIGDIR=str(WORK/'matplotlib'), MPLBACKEND='Agg',
               JULIA_DEPOT_PATH=str(depot)+':'+str(ROOT/'.julia-depot-r0'),
               JULIA_PKG_PRECOMPILE_AUTO='0', JULIA_PKG_OFFLINE='true',
               JULIA_HISTORY=str(WORK/'julia_history_disabled'))
    started = time.monotonic()
    with log.open('x', buffering=1) as stream:
        lines = ['START='+stamp(), 'COMMAND='+shlex.join([str(x) for x in command]),
                 'CWD='+str(ROOT), 'ENV='+json.dumps({k:env[k] for k in
                 ['PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                  'JULIA_NUM_THREADS','GIT_OPTIONAL_LOCKS','TMPDIR','JULIA_DEPOT_PATH',
                  'JULIA_PKG_PRECOMPILE_AUTO','JULIA_PKG_OFFLINE']})]
        for line in lines:
            stream.write(line+'\n')
            print(line, flush=True)
        result = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in result.stdout:
            stream.write(line)
            print(line, end='', flush=True)
        code = result.wait()
        stream.write('EXIT_CODE='+str(code)+'\nEND='+stamp()+'\n')
        stream.write('WALL_SECONDS='+str(time.monotonic()-started)+'\n')
    print('EXIT_CODE='+str(code), flush=True)
    if code:
        raise RuntimeError(f'S1 BLOCKED: {label} exited {code}; do not continue to S2')


def freeze():
    paths = existing_paths()
    rows = []
    for p in paths:
        name = relative(p)
        scope = ('R5_formal' if p.is_relative_to(FORMAL) else
                 'upstream_original' if name.startswith('upstream/') else
                 'environment_runtime' if name.startswith(('.venv-r0/', '.julia-depot-r0/')) else
                 'project_and_history')
        rows.append(dict(path=name, type='symlink' if p.is_symlink() else 'file',
                         size=p.lstat().st_size, sha256=sha(p), scope=scope))
    csv_new(OUT/'S1_baseline_manifest.csv', rows)
    formal_rows = [r for r in rows if r['scope'] == 'R5_formal']
    assert len(formal_rows) == 4113
    csv_new(OUT/'S1_R5_formal_manifest.csv', formal_rows)
    hashes = {r['path']:r['sha256'] for r in rows}
    git_records = []
    for directory in ['.', 'upstream/THIS', 'upstream/Su-AdaptiveLASSO']:
        record = dict(directory=directory)
        for label, args in [('status', ['status','--porcelain=v1','--untracked-files=all']),
                            ('head',['rev-parse','HEAD'])]:
            p = subprocess.run(['git','-C',str(ROOT/directory)]+args,
                capture_output=True,text=True,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
            record[label] = dict(exit_code=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
        if directory != '.':
            assert record['status']['exit_code'] == 0 and not record['status']['stdout']
        else:
            assert record['status']['exit_code'] == 128
        git_records.append(record)
    upstream = json.loads((ROOT/'environment/r0/upstream_lock.json').read_text())
    upstream_checks = []
    for repo, value in upstream.items():
        for name, expected in value['tracked_file_sha256'].items():
            full = 'upstream/'+repo+'/'+name
            assert hashes[full] == expected, full
            upstream_checks.append(dict(path=full, sha256=expected, commit=value['head']))
    csv_new(OUT/'S1_upstream_manifest.csv', upstream_checks)
    legacy = []
    for name in ['results/r1/artifact_manifest.toml','results/r3/artifact_manifest.toml',
                 'results/r4/artifact_manifest.toml','results/r5/artifact_manifest.toml',
                 'results/r5/e1_full_independent/artifact_manifest.toml']:
        m = tomllib.loads((ROOT/name).read_text())
        for file, expected in m['file_sha256'].items():
            assert hashes[file] == expected, (name,file)
        legacy.append(dict(path=name, checked=len(m['file_sha256']), unchanged=True))
    packages = {}
    for line in (ROOT/'environment/r0/requirements.lock.txt').read_text().splitlines():
        if '==' in line:
            name, version = line.split('==')
            actual = importlib.metadata.version(name)
            assert actual == version, name
            packages[name] = actual
    julia_version = subprocess.check_output([str(JULIA),'--startup-file=no','--version'],text=True).strip()
    assert julia_version == 'julia version 1.9.4'
    snapshot_prefixes = ('src/','scripts/','tests/','configs/','environment/','reports/',
                         'codex任务文件夹/','Codex_THIS_Refactor_Plan/','.vscode/')
    snapshot_paths = [p for p in paths if not p.is_symlink() and
        (relative(p).startswith(snapshot_prefixes) or
         (relative(p).startswith('upstream/') and '/.git/' not in relative(p)) or
         (relative(p).startswith('docs/') and '/refactor/' not in relative(p)) or
         relative(p) in ['README.md','.gitignore'])]
    archive = OUT/'S1_source_snapshot.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for p in snapshot_paths:
            tar.add(p, arcname=relative(p), recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == hashes[member.name]
    csv_new(OUT/'S1_source_snapshot_index.csv', [r for r in rows if r['path'] in
            {relative(p) for p in snapshot_paths}])
    json_new(OUT/'S1_environment.json', dict(created_at=stamp(), python=sys.version,
        executable=str(PYTHON), environment_type='Conda prefix; not Python venv',
        julia=julia_version, julia_binary=str(JULIA), git=git_records, packages=packages,
        legacy_manifests=legacy, upstream_tracked_file_count=len(upstream_checks),
        main_project_commit=None, protection='SHA256 manifest + source snapshot inside audit directory; no Git init/commit',
        original_entries=len(rows), source_snapshot_members=len(snapshot_paths),
        source_snapshot_sha256=sha(archive), source_snapshot_bytes=archive.stat().st_size,
        excluded_from_snapshot='runtime environments and experiment data/results (covered by full manifest, retained in place)',
        independent_disaster_recovery_backup=False))
    print('S1_FREEZE_PASS',len(rows),'entries',len(formal_rows),'formal files',len(snapshot_paths),'snapshot members',flush=True)


def golden():
    # Index existing outputs only. No new fitting or missing-field regeneration.
    entries = {}
    def remember(path, fixture, role):
        path = Path(path)
        assert path.is_file(), path
        key = relative(path)
        item = entries.setdefault(key,dict(path=key,size=path.stat().st_size,
            sha256=sha(path),fixture=set(),role=set()))
        item['fixture'].add(fixture)
        item['role'].add(role)
    clean = FORMAL/f'clean/seed_{SEED}/data'
    for p in sorted(clean.iterdir()):
        if p.is_file():remember(p,'R5_paired','clean states/response, Taylor/d, scaling, truth, split and RNG metadata')
    details = []
    for slot in ['noise_000','noise_050']:
        fixture = f'R5_{SEED}_{slot}'
        cell = FORMAL/f'seed_{SEED}'/slot
        for p in sorted((cell/'data').iterdir()):remember(p,fixture,'paired noisy response and injection metadata')
        for method in ['M0','M1','M2','M3']:
            trial = cell/method
            for p in sorted(trial.iterdir()):
                if p.is_file():remember(p,fixture+'_'+method,'existing model outputs including status, coefficients, diagnostics, scores and metrics')
            status = tomllib.loads((trial/'status.toml').read_text())
            diag = tomllib.loads((trial/'diagnostics.toml').read_text()) if (trial/'diagnostics.toml').exists() else {}
            metrics = tomllib.loads((trial/'metrics.toml').read_text()) if (trial/'metrics.toml').exists() else {}
            expected = 'NA' if slot=='noise_000' and method in ['M1','M2'] else 'OK'
            assert status['status'] == expected
            nodes = []
            for node in diag.get('nodes',[]):
                if expected == 'NA':
                    assert node['status'] == 'NA'
                    nodes.append(dict(node_julia_index=node['node_julia_index'],status='NA',reason=node['reason'],
                        missing_fields='not applicable: BIC guard stops selection; no selected alpha/coefficient'))
                else:
                    fields = ['initial_coef','weights','alpha','feature_means','feature_scales','alpha_path','bic_path',
                              'validation_path','selected_path_index','warnings','convergence_warning']
                    nodes.append({k:node[k] for k in ['node_julia_index']+fields if k in node})
                    if method in ['M2','M3']:
                        assert all(k in node for k in ['initial_coef','weights','alpha','feature_means','feature_scales'])
            details.append(dict(fixture=fixture,method=method,status=expected,path=relative(trial),
                node_diagnostics=nodes,metrics={k:metrics[k] for k in ['pairwise','triple'] if k in metrics},
                field_interpretation='Ridge/weights apply to externally standardized solver input; full arrays are in hashed existing diagnostics',
                not_applicable=('M0 uses STLSQ lambda/rho/niter, no Ridge weights/LASSO alpha' if method=='M0' else
                                'M1 ordinary LASSO has no Ridge initializer/adaptive weights' if method=='M1' else '')))
    # Keep controls, selected independently of outcome; reuse the already available R4 seed.
    r4 = ROOT/'results/r4/dry_run/seed_22261017'
    for p in sorted((r4/'data').iterdir()):remember(p,'R4_controls','existing R4 common data, feature/scaling/split metadata')
    for method in ['M0s','M1v']:
        for p in sorted((r4/method).iterdir()):
            if p.is_file():remember(p,'R4_'+method,'existing standardized-STLSQ or matched validation control')
    for name in ['protocol.toml','selection_locked.toml','trajectory_independence.toml','artifact_manifest.toml',
                 'trials.csv','summary_30.csv','paired_summary_30.csv','weak_edge_diagnostics.csv','candidate_warnings.csv',
                 'bootstrap_seed_indices.csv','paired_seed_differences.csv','sensitivity_new_28.csv']:
        remember(FORMAL/name,'R5_global','protocol/selection/statistics/warning provenance; no regeneration')
    for p in [ROOT/'configs/r5_e1_full_independent.toml',ROOT/'configs/r4_dry_run.toml',
              ROOT/'results/r5/e1_full/aborted.toml',ROOT/'results/r5/e1_full/README_ABORTED.md',
              ROOT/'results/r5/logs/full_run.log',ROOT/'results/r5/logs/full_test_independent.log',
              ROOT/'results/r5/logs/full_run_independent.log',ROOT/'results/r5/logs/full_audit.log']:
        remember(p,'R5_provenance','fixed configuration and historical testing/abort provenance')
    for item in entries.values():
        item['fixture'] = ';'.join(sorted(item['fixture']))
        item['role'] = ';'.join(sorted(item['role']))
    csv_new(OUT/'S1_golden_index.csv',sorted(entries.values(),key=lambda x:x['path']))
    samples=list(csv.DictReader((clean/'sample_manifest.csv').open()))
    selection=tomllib.loads((FORMAL/'selection_locked.toml').read_text())
    json_new(OUT/'S1_golden_contract.json',dict(created_at=stamp(),seed=SEED,
        selection_reason='S0 recommended diagnostic seed; fixed for behavioral regression, not performance selection',
        data_slice='same complete 480 columns in both levels; 300 train,90 validation,90 test; no feature truncation',
        shapes=dict(states=[8,480],theta=[165,480],monomial_indices=[165,3],coefficients=[8,165]),
        python_sample_indices_by_split={name:[i for i,r in enumerate(samples) if r['split']==name]
            for name in ['train','validation','test']},
        index_contract='matrix positions Python 0-based, Julia 1-based; d node IDs remain Julia 1-based with zero padding',
        strict_decode_threshold=1e-8,
        global_selection_lock=dict(path=relative(FORMAL/'selection_locked.toml'),sha256=sha(FORMAL/'selection_locked.toml'),
            locked_at=selection['locked_at'],trial_count=selection['trial_count']),
        fixtures=details,missing_required_applicable_fields=[],
        generation_performed=False,new_fitting_for_golden_creation=False,
        controls=['results/r4/dry_run/seed_22261017/M0s','results/r4/dry_run/seed_22261017/M1v']))
    trials=list(csv.DictReader((FORMAL/'trials.csv').open()))
    counts={label:sum(r['status']==label for r in trials) for label in ['OK','NA','FAILED']}
    assert counts == {'OK':420,'NA':60,'FAILED':0}
    sentinel={.05:[.6700,.8363,.8769,.4960],.1:[.6700,.8363,.8224,.4234],.2:[.6600,.7995,.6681,.3489]}
    checks=[]
    summary=list(csv.DictReader((FORMAL/'summary_30.csv').open()))
    for ratio, values in sentinel.items():
        for method,expected in zip(['M0','M1','M2','M3'],values):
            group=[r for r in trials if float(r['noise_ratio'])==ratio and r['method']==method]
            assert len(group)==30 and all(r['status']=='OK' for r in group)
            actual=sum(float(r['triple_F1']) for r in group)/30
            sr=next(r for r in summary if float(r['noise_ratio'])==ratio and r['method']==method
                    and r['order']=='triple' and r['metric']=='F1')
            assert abs(actual-float(sr['mean']))<1e-14
            assert abs(actual-expected)<5.01e-5
            checks.append(dict(noise_ratio=ratio,method=method,triple_F1_mean=actual,reported_4dp=expected,match=True))
    warns=list(csv.DictReader((FORMAL/'candidate_warnings.csv').open()))
    assert len(warns)==979
    assert sum(r['selected'].lower() in ['1','true'] for r in warns)==0
    json_new(OUT/'S1_sentinel_checks.json',dict(counts=counts,checks=checks,
        existing_candidate_warning_count=len(warns),existing_selected_warning_count=0,
        purpose='read-only sentinel recomputation from saved trial scores; never parameter selection'))
    print('S1_EXISTING_GOLDEN_INDEX_PASS',len(entries),'files; no missing applicable fields',flush=True)


def compat_python():
    import numpy as np
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    from r4_experiment import load_dataset
    from r4_methods import fit_node
    from r4_metrics import restore_coefficients, edge_scores, binary_metrics
    from src.solvers import BICNotApplicable
    base=tomllib.loads((ROOT/'configs/r4_dry_run.toml').read_text())
    results=[]
    def compare(method,data,response,trial):
        train=data['splits']=='train';val=data['splits']=='validation'
        old=tomllib.loads((trial/'status.toml').read_text())
        previous=tomllib.loads((trial/'diagnostics.toml').read_text())['nodes']
        coefs=[];alphas=[];na=[];init_error=weight_error=alpha_error=0.
        for node in range(8):
            validation=dict(X_validation=data['Z'][:,val].T,y_validation=response[node,val],
                train_groups=data['groups'][train],validation_groups=data['groups'][val])
            try:
                c,d=fit_node(method,data['Z'][:,train].T,response[node,train],base,validation)
            except BICNotApplicable as exc:
                assert old['status']=='NA' and previous[node]['status']=='NA'
                assert str(exc)==previous[node]['reason']
                na.append(node+1);continue
            assert old['status']=='OK'
            old_node=previous[node]
            np.testing.assert_allclose(d['alpha'],old_node['alpha'],rtol=1e-12,atol=1e-15)
            alpha_error=max(alpha_error,abs(d['alpha']-old_node['alpha']))
            assert d['selection']==old_node['selection']
            for key in ['alpha_path','bic_path']:
                if key in old_node:np.testing.assert_allclose(d[key],old_node[key],rtol=1e-11,atol=1e-12)
            if 'initial_coef' in old_node:
                np.testing.assert_allclose(d['initial_coef'],old_node['initial_coef'],rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(d['weights'],old_node['weights'],rtol=1e-12,atol=1e-10)
                init_error=max(init_error,float(np.max(np.abs(d['initial_coef']-old_node['initial_coef']))))
                weight_error=max(weight_error,float(np.max(np.abs(d['weights']-old_node['weights']))))
            if 'selected_path_index' in old_node:
                assert d['selected_path_index']==old_node['selected_path_index']
                assert len(d['validation_path'])==len(old_node['validation_path'])
                np.testing.assert_allclose([r['MSE'] for r in d['validation_path']],
                    [r['MSE'] for r in old_node['validation_path']],rtol=1e-11,atol=1e-14)
            assert len(d.get('warnings',[]))==len(old_node.get('warnings',[]))
            coefs.append(c);alphas.append(float(d['alpha']))
        if na:
            assert len(na)==8 and not (trial/'coefficients.csv').exists()
            results.append(dict(trial=relative(trial),method=method,status='NA',nodes=na,
                                guard_reason_exact=True,no_fallback=True));return
        new=restore_coefficients(np.asarray(coefs),data['means'],data['scales'])
        saved=np.loadtxt(trial/'coefficients.csv',delimiter=',',ndmin=2)
        np.testing.assert_allclose(new,saved,rtol=1e-10,atol=1e-12)
        assert np.array_equal(np.abs(new)>1e-8,np.abs(saved)>1e-8)
        truth=np.loadtxt(data['directory']/'true_coefficients.csv',delimiter=',',ndmin=2)
        metrics=tomllib.loads((trial/'metrics.toml').read_text())
        orders={}
        for name,order in [('pairwise',2),('triple',3)]:
            keys,scores=edge_scores(new,data['d'],order)
            savedkeys,savedscores=edge_scores(saved,data['d'],order)
            assert keys==savedkeys
            np.testing.assert_allclose(scores,savedscores,rtol=1e-10,atol=1e-12)
            _,labels=edge_scores(truth,data['d'],order)
            m=binary_metrics(labels>1e-8,scores)
            assert all(m[k]==metrics[name][k] for k in ['TP','FP','FN','TN'])
            orders[name]={k:m[k] for k in ['TP','FP','FN','TN','F1']}
        results.append(dict(trial=relative(trial),method=method,status='OK',
            max_abs_coef_difference=float(np.max(np.abs(new-saved))),
            coefficients_bitwise_equal=bool(np.array_equal(new,saved)),
            support_exact=True,alpha_max_abs_difference=alpha_error,
            initial_coef_max_abs_difference=init_error,weight_max_abs_difference=weight_error,
            selected_alphas=alphas,metrics=orders))
        print('S1_PYTHON_COMPAT_PASS',relative(trial),flush=True)
    data=load_dataset(FORMAL/'clean',SEED)
    for slot in ['noise_000','noise_050']:
        cell=FORMAL/f'seed_{SEED}'/slot
        response=np.loadtxt(cell/'data/Y_noisy.csv',delimiter=',',ndmin=2)
        for method in ['M1','M2','M3']:
            compare(method,data,response,cell/method)
    control=ROOT/'results/r4/dry_run'
    r4=load_dataset(control,22261017)
    compare('M1v',r4,r4['Y'],control/'seed_22261017/M1v')
    json_new(OUT/'S1_python_compatibility.json',dict(passed=True,results=results,
        replay_scope='one paired R5 seed, two levels, plus existing R4 M1v control; no new experimental protocol',
        output_policy='in-memory fits only; no results directory writes'))


def verify():
    rows=list(csv.DictReader((OUT/'S1_baseline_manifest.csv').open()))
    errors=[]
    for r in rows:
        p=ROOT/r['path']
        if not p.exists() and not p.is_symlink():errors.append([r['path'],'missing']);continue
        if sha(p)!=r['sha256'] or p.lstat().st_size!=int(r['size']):errors.append([r['path'],'changed'])
    before={r['path'] for r in rows}
    now={relative(p) for p in existing_paths()}
    errors.extend([p,'unexpected added file outside S1 audit outputs'] for p in sorted(now-before))
    errors.extend([p,'removed'] for p in sorted(before-now))
    result=dict(verified_at=stamp(),passed=not errors,original_entries_checked=len(rows),errors=errors,
                all_original_files_unchanged=not errors,formal_file_count=sum(r['scope']=='R5_formal' for r in rows),
                excluded_new_audit_scope='docs/refactor/S1_* only',S2_executed=False)
    json_new(OUT/'S1_integrity_check.json',result)
    assert not errors, errors
    print('S1_ORIGINAL_INTEGRITY_PASS',len(rows),'entries; S2 not executed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['freeze','golden','tests','compat','verify'])
    args=parser.parse_args()
    if args.task=='freeze':freeze()
    elif args.task=='golden':golden()
    elif args.task=='tests':
        run([PYTHON,'-B','-m','unittest','discover','-s','tests','-p','test_r*.py','-v'],'python_tests')
        for script,label in [('tests/test_r1_baseline.jl','julia_r1_tests'),
                             ('tests/test_r1_revision.jl','julia_r1_revision_tests'),
                             ('tests/test_r3_adapter.jl','julia_r3_tests')]:
            run([JULIA,'--startup-file=no','--history-file=no','--compiled-modules=no',
                 '--project='+str(ROOT/'environment/r0/julia'),script],label)
    elif args.task=='compat':
        run([PYTHON,'-B',__file__,'compat_python_internal'],'python_compatibility')
    elif args.task=='verify':verify()


if __name__=='__main__':
    try:
        # Internal worker is separately logged; never call a legacy main.
        if sys.argv[1:]==['compat_python_internal']:compat_python()
        else:main()
    except Exception:
        traceback.print_exc()
        print('S1 BLOCKED: audit failed; no S2 execution',flush=True)
        raise
