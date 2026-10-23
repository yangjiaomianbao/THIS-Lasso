"""Command logger for R4 tests, phase-A dry run, and final audit."""
import argparse
import datetime
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['test', 'dry_run', 'audit'])
    parser.add_argument('--label')
    args = parser.parse_args();label = args.label or args.task
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Invalid log label')
    python = str(ROOT/'.venv-r0/bin/python')
    commands = dict(test=[python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_r4_*.py', '-v'],
        dry_run=[python, 'scripts/r4_experiment.py', '--config', 'configs/r4_dry_run.toml', '--output', 'results/r4/dry_run'],
        audit=[python, 'scripts/r4_audit.py'])
    logs = ROOT/'results/r4/logs';logs.mkdir(parents=True, exist_ok=True)
    path = logs/(label+'.log')
    env = os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', JULIA_NUM_THREADS='1', R4_ACTIVE_LOG=str(path.relative_to(ROOT)))
    with path.open('x', buffering=1) as f:
        for line in ['START='+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                     'COMMAND='+shlex.join(commands[args.task]), 'CWD='+str(ROOT), 'PYTHON='+sys.version.replace('\n', ' '),
                     'ENV=PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1']:
            f.write(line+'\n');print(line, flush=True)
        p = subprocess.Popen(commands[args.task], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line);print(line, end='', flush=True)
        code = p.wait();f.write('EXIT_CODE='+str(code)+'\n');print('EXIT_CODE='+str(code)+'\nLOG='+str(path), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
