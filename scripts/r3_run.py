"""Run one R3 check and retain its actual command, environment, and exit code."""
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
    parser.add_argument('task', choices=['python_test', 'julia_test', 'demo', 'audit'])
    parser.add_argument('--label')
    args = parser.parse_args()
    label = args.label or args.task
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Invalid log label')
    python = str(ROOT/'.venv-r0/bin/python')
    julia = [str(ROOT/'environment/r0/julia.sh'), '--startup-file=no', '--project='+str(ROOT/'environment/r0/julia')]
    commands = dict(python_test=[python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_r3_*.py', '-v'],
        julia_test=julia+['tests/test_r3_adapter.jl'], demo=julia+['scripts/r3_demo.jl'],
        audit=[python, 'scripts/r3_audit.py'])
    logs = ROOT/'results/r3/logs';logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', JULIA_NUM_THREADS='1')
    path = logs/(label+'.log')
    env['R3_ACTIVE_LOG'] = str(path.relative_to(ROOT))
    with path.open('x', buffering=1) as f:
        for line in ['START='+datetime.datetime.now().astimezone().isoformat(), 'CWD='+str(ROOT),
                     'COMMAND='+shlex.join(commands[args.task]), 'PYTHON='+sys.version.replace('\n', ' '),
                     'ENV=PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1']:
            f.write(line+'\n');print(line, flush=True)
        process = subprocess.Popen(commands[args.task], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            f.write(line);print(line, end='', flush=True)
        code = process.wait()
        f.write('EXIT_CODE='+str(code)+'\nEND='+datetime.datetime.now().astimezone().isoformat()+'\n')
        print('EXIT_CODE='+str(code)+'\nLOG='+str(path), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
