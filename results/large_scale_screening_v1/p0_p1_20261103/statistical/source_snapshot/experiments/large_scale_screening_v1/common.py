"""Serialization, resource measurement and immutable run provenance."""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PROTECTED=['src/solvers/adaptive_lasso.py','src/solvers/methods.py','src/solvers/large_library.py',
 'src/preprocessing/taylor.py','src/preprocessing/pearson_filter.jl','src/bridge/physical_decode.jl',
 'src/this_adapter.jl','scripts/dynamics30.py','docs/data_contract.md','docs/REPRODUCE.md']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def table(path,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(clean(rows))


def load_config(path=None): return json.loads(Path(path or HERE/'configs/default.json').read_text())


def source_hashes():
    paths=set(ROOT/p for p in PROTECTED)
    paths.update(HERE.rglob('*.py'));paths.update(HERE.rglob('*.json'));paths.update(HERE.glob('*.md'))
    # Include all upstream code and environment locks without modifying them.
    paths.update((ROOT/'upstream/THIS').glob('*.jl'))
    paths.update(p for p in (ROOT/'environment/r0').rglob('*') if p.is_file() and p.suffix in ('.toml','.txt'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


class Monitor:
    """Absolute process RSS peak sampled during this operation (includes baseline)."""
    def __enter__(self):
        self.start=time.perf_counter();self.peak=0;self.done=threading.Event()
        self.thread=threading.Thread(target=self._sample,daemon=True);self.thread.start();return self
    def _sample(self):
        while not self.done.is_set():
            try:
                rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                self.peak=max(self.peak,rss)
            except (OSError,ValueError): pass
            self.done.wait(.01)
    def __exit__(self,*args):
        self.done.set();self.thread.join();self.seconds=time.perf_counter()-self.start
    def fields(self): return dict(seconds=self.seconds,cpu_peak_rss_bytes=self.peak,memory_definition='absolute process RSS, sampled at 10ms, baseline included',gpu_peak_bytes=None,gpu_reason='CPU implementation; CUDA not used')


@contextlib.contextmanager
def timeout(seconds):
    def expired(*_): raise TimeoutError(f'Operation exceeded {seconds}s')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally: signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
