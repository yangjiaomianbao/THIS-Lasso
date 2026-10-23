"""Run one R0 environment command and preserve its real output/exit status."""
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[1]
label, *command = sys.argv[1:]
if not label.replace('_', '').replace('-', '').isalnum() or not command:
    raise SystemExit('Usage: r0_run_logged.py LABEL COMMAND [ARG ...]')
log_dir = root / 'reports' / 'r0_logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f'{label}.log'
if log_path.exists():
    raise SystemExit(f'Log already exists; choose a new label: {log_path}')
with log_path.open('x', buffering=1) as log:
    log.write(f'START={datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}\n')
    log.write(f'CWD={root}\nCOMMAND={shlex.join(command)}\n')
    process = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in process.stdout:
        log.write(line)
        print(line, end='', flush=True)
    code = process.wait()
    log.write(f'EXIT_CODE={code}\n')
    log.write(f'END={datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}\n')
print(f'LOG={log_path}', flush=True)
raise SystemExit(code)
