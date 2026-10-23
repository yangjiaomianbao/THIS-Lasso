"""Immutable log entry for authorized R5 full-E1 tasks."""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('task', choices=['test', 'run', 'audit']);p.add_argument('--label')
    a = p.parse_args();label = a.label or 'full_'+a.task
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Invalid log label')
    py = str(ROOT/'.venv-r0/bin/python')
    commands = dict(test=[py, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_r5_*.py', '-v'],
        run=[py, 'scripts/r5_batch.py', '--config', 'configs/r5_e1_full_independent.toml', '--output', 'results/r5/e1_full_independent'],
        audit=[py, 'scripts/r5_batch_audit.py'])
    log = ROOT/'results/r5/logs'/(label+'.log');log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JULIA_NUM_THREADS='1', MPLBACKEND='Agg', R5_FULL_ACTIVE_LOG=str(log.relative_to(ROOT)))
    with log.open('x', buffering=1) as f:
        for line in ['START='+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'COMMAND='+shlex.join(commands[a.task]), 'CWD='+str(ROOT), 'PYTHON='+sys.version.replace('\n', ' '),
            'ENV=BLAS=1 JULIA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg']:
            f.write(line+'\n');print(line, flush=True)
        proc = subprocess.Popen(commands[a.task], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            f.write(line);print(line, end='', flush=True)
        code = proc.wait();f.write('EXIT_CODE='+str(code)+'\n');print('EXIT_CODE='+str(code), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
