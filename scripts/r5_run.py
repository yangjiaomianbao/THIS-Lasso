"""Log R5 tests, two-seed E1 smoke, and audit with no implicit batch action."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['test', 'smoke', 'audit']);p.add_argument('--label')
    args = p.parse_args();label = args.label or args.task
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Invalid log label')
    py = str(ROOT/'.venv-r0/bin/python')
    commands = dict(test=[py, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_r5_*.py', '-v'],
        smoke=[py, 'scripts/r5_experiment.py', '--config', 'configs/r5_e1_smoke.toml', '--output', 'results/r5/e1_smoke'],
        audit=[py, 'scripts/r5_audit.py'])
    directory = ROOT/'results/r5/logs';directory.mkdir(parents=True, exist_ok=True)
    log = directory/(label+'.log');env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', JULIA_NUM_THREADS='1',
               R5_ACTIVE_LOG=str(log.relative_to(ROOT)))
    with log.open('x', buffering=1) as f:
        for line in ['START='+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                     'COMMAND='+shlex.join(commands[args.task]), 'CWD='+str(ROOT), 'PYTHON='+sys.version.replace('\n', ' '),
                     'ENV=PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1']:
            f.write(line+'\n');print(line, flush=True)
        proc = subprocess.Popen(commands[args.task], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            f.write(line);print(line, end='', flush=True)
        code = proc.wait();f.write('EXIT_CODE='+str(code)+'\n');print('EXIT_CODE='+str(code)+'\nLOG='+str(log), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
