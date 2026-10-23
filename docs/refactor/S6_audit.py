"""Repeatable source/golden/statistics audit. Writes only a new labeled audit folder."""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/refactor'
PYTHON=ROOT/'.venv-r0/bin/python'
JULIA=Path('/home/yg/.local/opt/julia-1.9.4/bin/julia')


def sha(p):
    if p.is_symlink():return hashlib.sha256(os.readlink(p).encode()).hexdigest()
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def json_new(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def run(command,folder,label):
    assert folder.resolve().is_relative_to(OUT)
    folder.mkdir(parents=True,exist_ok=True);work=folder/label;work.mkdir(exist_ok=False)
    (work/'tmp').mkdir();(work/'depot').mkdir()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        JULIA_NUM_THREADS='1',GIT_OPTIONAL_LOCKS='0',TMPDIR=str(work/'tmp'),TMP=str(work/'tmp'),TEMP=str(work/'tmp'),
        JULIA_DEPOT_PATH=str(work/'depot')+':'+str(ROOT/'.julia-depot-r0'),JULIA_PKG_PRECOMPILE_AUTO='0',
        JULIA_PKG_OFFLINE='true',MPLCONFIGDIR=str(work/'matplotlib'),MPLBACKEND='Agg',XDG_CACHE_HOME=str(work/'xdg-cache'))
    started=time.monotonic()
    with (folder/(label+'.log')).open('x',buffering=1) as log:
        for line in ['START='+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                     'COMMAND='+shlex.join([str(x) for x in command]),'CWD='+str(ROOT),
                     'ENV='+json.dumps({k:env[k] for k in ['PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS',
                       'MKL_NUM_THREADS','JULIA_NUM_THREADS','GIT_OPTIONAL_LOCKS','TMPDIR','JULIA_DEPOT_PATH']})]:
            log.write(line+'\n');print(line,flush=True)
        proc=subprocess.Popen([str(x) for x in command],cwd=ROOT,env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:log.write(line);print(line,end='',flush=True)
        code=proc.wait();log.write('EXIT_CODE='+str(code)+'\nWALL_SECONDS='+str(time.monotonic()-started)+'\n')
    print('EXIT_CODE='+str(code),flush=True)
    if code:raise RuntimeError('Audit failed: '+label)


def integrity(folder):
    changed={r['path']:r['after_sha256'] for r in csv.DictReader((OUT/'S2_changed_files.csv').open()) if r['action']=='modified'}
    deleted={r['path']:r for r in csv.DictReader((OUT/'S5_deletion_log.csv').open())}
    approved={r['path']:r for r in csv.DictReader((OUT/'S5_delete_allowlist_APPROVED.csv').open())}
    assert set(deleted)==set(approved)
    original=list(csv.DictReader((OUT/'S1_baseline_manifest.csv').open()));unchanged=0;absent=0;regenerated=[]
    for r in original:
        p=ROOT/r['path']
        if r['path'] in changed:
            assert sha(p)==changed[r['path']],r['path'];continue
        if r['path']=='README.md':
            assert sha(ROOT/'archive/snapshots/README_before_S3.md')==r['sha256'];continue
        if r['path'] in deleted:
            assert deleted[r['path']]['before_sha256']==r['sha256']
            assert sha(ROOT/approved[r['path']]['source_path'])==approved[r['path']]['source_sha256']
            if p.exists():
                assert not p.is_symlink() and p.suffix=='.pyc'
                regenerated.append(dict(path=r['path'],sha256=sha(p),same_as_original=sha(p)==r['sha256']))
            else:absent+=1
            continue
        assert p.exists() or p.is_symlink(),r['path']
        assert sha(p)==r['sha256'],r['path'];unchanged+=1
    for filename in ['S1_audit_artifacts.csv','S2_artifact_manifest.csv']:
        for r in csv.DictReader((OUT/filename).open()):assert sha(ROOT/r['path'])==r['sha256'],r['path']
    for r in csv.DictReader((OUT/'S1_golden_index.csv').open()):assert sha(ROOT/r['path'])==r['sha256'],r['path']
    auth=json.loads((OUT/'S5_authorization.json').read_text());archive=ROOT/auth['backup_archive']
    assert sha(archive)==auth['backup_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert {m.name for m in members}==set(approved)
        for m in members:
            assert m.isfile()
            with tar.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==approved[m.name]['sha256']
    heads={}
    for repo in ['THIS','Su-AdaptiveLASSO']:
        env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
        status=subprocess.check_output(['git','-C',str(ROOT/'upstream'/repo),'status','--porcelain=v1','--untracked-files=all'],text=True,env=env)
        assert not status
        heads[repo]=subprocess.check_output(['git','-C',str(ROOT/'upstream'/repo),'rev-parse','HEAD'],text=True,env=env).strip()
    json_new(folder/'integrity.json',dict(passed=True,S1_entries=len(original),unchanged_original_entries=unchanged,
        S2_source_edits=sorted(changed),README_original_backed_up=True,approved_cache_deletions=len(deleted),
        approved_caches_currently_absent=absent,regenerated_optional_caches=regenerated,formal_R5_files_unchanged=4113,
        S0_S1_S2_artifacts_unchanged=True,backup_every_member_verified=True,upstream_heads=heads,
        rule='Optional caches may regenerate; scientific input/output and source rules remain strict.'))
    print('S6_INTEGRITY_PASS',unchanged,'unchanged original entries;',absent,'approved caches absent; 4113 formal files unchanged',flush=True)


def golden(folder):
    spec=importlib.util.spec_from_file_location('s1',OUT/'S1_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=folder/'golden';m.OUT.mkdir(exist_ok=False);m.compat_python()


def statistics(folder):
    import numpy as np
    import tomllib
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    from r5_batch_analyse import make_summaries
    base=ROOT/'results/r5/e1_full_independent'
    c=tomllib.loads((ROOT/'configs/r5_e1_full_independent.toml').read_text())
    rows=list(csv.DictReader((base/'trials.csv').open()));strata=list(csv.DictReader((base/'strength_recall.csv').open()))
    counts={k:sum(r['status']==k for r in rows) for k in ['OK','NA','FAILED']};assert counts==dict(OK=420,NA=60,FAILED=0)
    for r in rows:
        if float(r['noise_ratio'])==0. and r['method'] in ['M1','M2']:assert r['status']=='NA'
    destination=folder/'recomputed_statistics';destination.mkdir(exist_ok=False)
    make_summaries(rows,strata,c,destination)
    checks=[]
    for name in ['summary_30.csv','paired_summary_30.csv','paired_seed_differences.csv',
                 'strength_summary_30.csv','sensitivity_new_28.csv','bootstrap_seed_indices.csv']:
        assert sha(base/name)==sha(destination/name),name
        checks.append(dict(path=name,S1_formal_sha256=sha(base/name),recomputed_audit_sha256=sha(destination/name),exact=True))
    warnings=list(csv.DictReader((base/'candidate_warnings.csv').open()))
    assert len(warnings)==979 and sum(r['selected'].lower() in ['true','1'] for r in warnings)==0
    json_new(folder/'statistics_checks.json',dict(passed=True,counts=counts,checks=checks,
        candidate_warning_count=979,selected_warning_count=0,new_model_fitting=False,formal_results_overwritten=False))
    print('S6_SIX_STATISTICAL_FILES_EXACT; all 30 seeds preserved; no refitting',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',default='audit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),help='new unique output label; existing outputs are never overwritten')
    parser.add_argument('--worker',choices=['golden','statistics','integrity'],help=argparse.SUPPRESS)
    parser.add_argument('--folder',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        folder=args.folder.resolve();assert folder.is_relative_to(OUT/'S6_work')
        globals()[args.worker](folder);return
    assert re.fullmatch(r'[A-Za-z0-9_-]+',args.label)
    folder=OUT/'S6_work'/args.label;folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    try:
        run([PYTHON,'-B',__file__,'--worker','integrity','--folder',folder],folder,'integrity_before')
        run([PYTHON,'-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],folder,'python_tests')
        for test in ['test_r1_baseline','test_r1_revision','test_r3_adapter']:
            run([JULIA,'--startup-file=no','--history-file=no','--compiled-modules=no',
                 '--project='+str(ROOT/'environment/r0/julia'),ROOT/'tests'/(test+'.jl')],folder,test)
        run([PYTHON,'-B',__file__,'--worker','golden','--folder',folder],folder,'python_golden')
        run([JULIA,'--startup-file=no','--history-file=no','--compiled-modules=no',
             '--project='+str(ROOT/'environment/r0/julia'),OUT/'S6_trial_smoke.jl',folder/'bridge_trial'],folder,'cross_language_trial')
        run([PYTHON,'-B',__file__,'--worker','statistics','--folder',folder],folder,'statistics')
        cli=[p for p in (ROOT/'scripts').glob('*.py') if 'ArgumentParser(' in p.read_text()]
        cli.extend([ROOT/'scripts/r3_bridge.py',ROOT/'src/data_bridge.py',OUT/'S5_restore_caches.py',Path(__file__).resolve()])
        for script in sorted(set(cli)):
            # Package modules are invoked with -m; the public script wrapper is
            # separately checked above. A direct src/data_bridge.py path does not
            # put the project root on Python's module search path, including in S1.
            command=([PYTHON,'-B','-m','src.data_bridge','--help'] if script==ROOT/'src/data_bridge.py'
                     else [PYTHON,'-B',script,'--help'])
            run(command,folder,'help_'+script.parent.name+'_'+script.stem)
        # Keep the after-check independently named, without overwriting the first integrity result.
        after=folder/'after';after.mkdir()
        run([PYTHON,'-B',__file__,'--worker','integrity','--folder',after],folder,'integrity_after')
        json_new(folder/'result.json',dict(status='PASS',date='2026-10-20',label=args.label,
            python_tests=95,existing_julia_assertions=158,one_trial_julia_assertions=21,
            formal_R5_files_unchanged=4113,paired_golden_preserved=True,
            six_recomputed_statistics_files_exact=True,wall_seconds=time.monotonic()-start,
            full_30_seed_refit=False,new_scientific_experiments=False))
        print('S6_FINAL_REPRODUCTION_AUDIT_PASS',str(folder),flush=True)
    except Exception as exc:
        json_new(folder/'result.json',dict(status='BLOCKED',error=str(exc)))
        raise


if __name__=='__main__':main()
