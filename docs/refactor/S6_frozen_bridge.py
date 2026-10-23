"""Execute the exact S1 bridge source for side-by-side audit outputs only."""
import csv
import hashlib
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
directory=Path(sys.argv[1]).resolve()
assert directory.is_relative_to(ROOT/'docs/refactor/S6_work')
frozen={r['path']:r['sha256'] for r in csv.DictReader((ROOT/'docs/refactor/S1_baseline_manifest.csv').open())}
for name in ['src/solvers/adaptive_lasso.py','src/solvers/__init__.py']:
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==frozen[name]
with tarfile.open(ROOT/'docs/refactor/S1_source_snapshot.tar.gz','r:gz') as tar:
    code=tar.extractfile('src/data_bridge.py').read()
assert hashlib.sha256(code).hexdigest()==frozen['src/data_bridge.py']
namespace={'__name__':'frozen_S1_bridge'}
exec(compile(code,'S1_source_snapshot/src/data_bridge.py','exec'),namespace)
namespace['process_directory'](directory)
