"""Stage-specific audit; never run historical write-back audit/experiment mains."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/refactor'
WORK = OUT/'S2_work'
PYTHON = ROOT/'.venv-r0/bin/python'
JULIA = Path('/home/yg/.local/opt/julia-1.9.4/bin/julia')


def run(command, label):
    work = WORK/label
    work.mkdir(parents=True, exist_ok=True)
    (work/'tmp').mkdir(exist_ok=True)
    (work/'depot').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JULIA_NUM_THREADS='1', GIT_OPTIONAL_LOCKS='0', TMPDIR=str(work/'tmp'), TMP=str(work/'tmp'),
        TEMP=str(work/'tmp'), JULIA_DEPOT_PATH=str(work/'depot')+':'+str(ROOT/'.julia-depot-r0'),
        JULIA_PKG_PRECOMPILE_AUTO='0', JULIA_PKG_OFFLINE='true', MPLBACKEND='Agg',
        MPLCONFIGDIR=str(work/'matplotlib'))
    started=time.monotonic()
    with (OUT/('S2_'+label+'.log')).open('x',buffering=1) as log:
        for line in ['START='+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                     'COMMAND='+shlex.join([str(x) for x in command]),'CWD='+str(ROOT),
                     'ENV='+json.dumps({k:env[k] for k in ['PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS',
                     'MKL_NUM_THREADS','JULIA_NUM_THREADS','GIT_OPTIONAL_LOCKS','TMPDIR','JULIA_DEPOT_PATH',
                     'JULIA_PKG_PRECOMPILE_AUTO','JULIA_PKG_OFFLINE']})]:
            log.write(line+'\n');print(line,flush=True)
        proc=subprocess.Popen([str(x) for x in command],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:log.write(line);print(line,end='',flush=True)
        code=proc.wait();log.write('EXIT_CODE='+str(code)+'\nWALL_SECONDS='+str(time.monotonic()-started)+'\n')
    print('EXIT_CODE='+str(code),flush=True)
    if code:raise RuntimeError('S2 verification failed; stop and revert the current module before proceeding')


def golden(phase):
    # Replay the immutable S1 verification function, redirect only its new audit output.
    spec=importlib.util.spec_from_file_location('s1_frozen_audit',OUT/'S1_audit.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.OUT=WORK/phase/'golden';old.OUT.mkdir(parents=True,exist_ok=True)
    old.compat_python()
    print('S2_'+phase+'_S1_GOLDEN_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['phase','golden','julia_existing'])
    parser.add_argument('--phase',choices=['A','B','C']);parser.add_argument('--attempt',default='');args=parser.parse_args()
    phase=args.phase+('_'+args.attempt if args.attempt else '') if args.phase else None
    if args.task=='golden':golden(phase)
    elif args.task=='phase':
        assert args.phase
        run([PYTHON,'-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],phase+'_python_tests')
        command=[PYTHON,'-B',__file__,'golden','--phase',args.phase]
        if args.attempt:command+=['--attempt',args.attempt]
        run(command,phase+'_golden')
        run([JULIA,'--startup-file=no','--history-file=no','--compiled-modules=no',
             '--project='+str(ROOT/'environment/r0/julia'),OUT/'S2_trial_smoke.jl',str(WORK/phase)],phase+'_trial_smoke')
    else:
        for name in ['test_r1_baseline','test_r1_revision','test_r3_adapter']:
            run([JULIA,'--startup-file=no','--history-file=no','--compiled-modules=no',
                 '--project='+str(ROOT/'environment/r0/julia'),ROOT/'tests'/(name+'.jl')],name)
