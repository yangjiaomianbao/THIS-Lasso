"""Detached supervisor for the user-authorized V2 formal run and S7 analysis."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path('/home/yg/THIS-Lasso')
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-r0/bin/python'
RUN = ROOT / 'results/dynamics15/fair_comparison_v2_full_run01'
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', JULIA_NUM_THREADS='1', PYTHONUNBUFFERED='1')
STATE = dict(authorization='User: 启动正式的实验', supervisor_pid=os.getpid(),
             started_at=datetime.now(timezone.utc).isoformat(),
             result_directory=str(RUN), planned_models=900, workers=8)


def record(stage, **extra):
    STATE.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    tmp = HERE / 'status.json.tmp'
    tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(HERE / 'status.json')
    print(json.dumps(STATE, ensure_ascii=False), flush=True)


def command(stage, args, log):
    record(stage, child_pid=None)
    with log.open('x') as stream:
        proc = subprocess.Popen([str(PYTHON), '-B', *args], cwd=ROOT, env=ENV,
                                stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT)
        record(stage, child_pid=proc.pid, current_log=str(log))
        code = proc.wait()
    if code:
        raise RuntimeError(f'{stage} exited with code {code}; see {log}')


def audit(stage, name):
    code = "import sys; sys.path.insert(0, 'scripts'); from fair_v2 import audit_old; print(audit_old(" + repr('full_run01_startup/' + name) + '))'
    command(stage, ['-c', code], HERE / (stage + '.log'))


try:
    assert not RUN.exists(), 'Refusing to overwrite an existing formal run'
    audit('PRE_RUN_AUDIT', 'pre_run_integrity.json')
    command('RUNNING_EXPERIMENT', ['scripts/fair_v2.py', 'run', '--mode', 'full',
            '--label', RUN.name, '--workers', '8', '--confirm-full'],
            HERE.parent / 'full_run01_console.log')
    command('RUNNING_S7_ANALYSIS', ['scripts/fair_v2_analyse.py', str(RUN),
            '--label', 'formal_statistics01'], HERE / 's7_analysis.log')
    audit('POST_RUN_AUDIT', 'post_run_integrity.json')
    completion = json.loads((RUN / 'completion.json').read_text())
    record('COMPLETE' if completion['counts']['FAILED'] == 0 else 'COMPLETE_WITH_MODEL_FAILURES',
           child_pid=None, completion=completion)
except Exception as exc:
    record('STOPPED_WITH_ERROR', error=str(exc), child_pid=None)
    traceback.print_exc()
    raise
