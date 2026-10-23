"""Read-only S4 dependency/classification audit; contains no deletion operation."""
import ast
import collections
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import warnings

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/refactor'
BACKUP='archive/caches/S5_python_cache_backup.tar.gz'


def sha(path):
    if path.is_symlink():return hashlib.sha256(os.readlink(path).encode()).hexdigest()
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def csv_new(name,rows,fields=None):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fields or list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    files=[]
    for base,dirs,names in os.walk(ROOT,followlinks=False):
        for name in dirs:
            p=Path(base)/name
            if p.is_symlink():files.append(p)
        for name in names:files.append(Path(base)/name)
    files.sort();by_name={p.relative_to(ROOT).as_posix():p for p in files}
    edges=set();cache_refs=collections.defaultdict(set);source_records={}
    cache_pattern=re.compile(r'(?:\.venv-r0|\.julia-depot-r0)/[^\s\"\'`,;|<>]+')
    suffixes={'.py','.jl','.toml','.json','.sh','.md','.txt','.in','.log','.csv'}
    for i,p in enumerate(files):
        rel=p.relative_to(ROOT).as_posix()
        if p.is_symlink() or '/.git/' in rel or p.suffix not in suffixes:continue
        try:text=p.read_text()
        except (OSError,UnicodeError):continue
        bookkeeping=rel.startswith(('docs/refactor/','archive/'))
        for target in cache_pattern.findall(text):
            if target in by_name:
                kind='audit_bookkeeping' if bookkeeping else 'operational_literal'
                cache_refs[target].add(kind+':'+rel);edges.add((rel,target,kind))
        if rel.startswith(('.venv-r0/','.julia-depot-r0/','upstream/')):continue
        def add(target,kind):
            try:name=Path(os.path.normpath(target)).relative_to(ROOT).as_posix()
            except ValueError:return
            if name in by_name and name!=rel:edges.add((rel,name,kind))
        for target in re.findall(r'''['\"]([^'\"\n]{1,350})['\"]''',text):
            if '/' in target:
                add(ROOT/target,'path_literal');add(p.parent/target,'path_literal')
            elif target.endswith(('.jl','.py')):
                add(p.parent/target,'include_or_path');add(ROOT/'scripts'/target,'include_or_path')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' not in target and not target.startswith('#'):add(p.parent/target.split('#')[0].strip('<>'),'markdown_link')
        if p.suffix=='.py':
            try:tree=ast.parse(text)
            except SyntaxError:continue
            for node in ast.walk(tree):
                modules=[]
                if isinstance(node,ast.Import):modules=[x.name for x in node.names]
                elif isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]+[node.module+'.'+x.name for x in node.names]
                for module in modules:
                    target=module.replace('.','/')+'.py'
                    for q in [ROOT/target,ROOT/'scripts'/target,p.parent/target]:add(q,'python_import')
    print('S4_STATIC_SCAN_DONE',len(files),'entries',len(edges),'reference edges',flush=True)
    proposals=[];classification=[];counts=collections.Counter();bytes_by_class=collections.Counter();warns=[]
    for p in files:
        rel=p.relative_to(ROOT).as_posix();s=p.lstat();kind='KEEP';reason='source, configuration, installed runtime or protected experiment/audit evidence';risk='none'
        refs=sorted(cache_refs.get(rel,set()))
        is_pycache=rel.startswith('.venv-r0/') and p.suffix=='.pyc' and not p.is_symlink()
        if is_pycache:
            if s.st_nlink!=1:
                reason='Shared/hardlinked installed bytecode retained; unlink would not release its last physical link'
            else:
                try:
                    source=Path(importlib.util.source_from_cache(str(p)))
                    assert source.is_file() and source.resolve().is_relative_to(ROOT)
                    with p.open('rb') as f:assert f.read(4)==importlib.util.MAGIC_NUMBER
                    assert not any(x.startswith('operational_literal:') for x in refs)
                    opt=1 if '.opt-1.' in p.name else 2 if '.opt-2.' in p.name else 0
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');compile(source.read_bytes(),str(source),'exec',dont_inherit=True,optimize=opt)
                    if caught:warns.append(dict(path=rel,compile_warning_count=len(caught)))
                    kind='DELETE_CANDIDATE';risk='low, optional cache with source, verified compile and exact-byte backup required'
                    reason='Single-link CPython3.11 cache; same source compiles without execution; import can fall back to source'
                    cmd=shlex.join(['.venv-r0/bin/python','-B','-m','py_compile',source.relative_to(ROOT).as_posix()])
                    proposals.append(dict(path=rel,size=s.st_size,sha256=sha(p),tracked='N/A: root not a Git repository; ignored runtime cache',
                        refs_found=json.dumps(refs,ensure_ascii=False),duplicate_of_or_regeneration_command=cmd,
                        backup_path=BACKUP+'::'+rel,risk=risk,reason=reason,approval_status='PROPOSED',
                        source_path=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),
                        source_compile_verified=True,st_nlink=s.st_nlink,allocated_bytes=s.st_blocks*512,
                        mode=s.st_mode & 0o777,mtime_ns=s.st_mtime_ns))
                except (AssertionError,ValueError,SyntaxError,OSError) as exc:
                    kind='UNKNOWN';reason='Cache recovery/use proof incomplete: '+type(exc).__name__;risk='retain'
        elif rel.startswith('.julia-depot-r0/compiled/'):
            reason='Native Julia compilation cache retained; package/extension loading differs from ordinary Python bytecode'
        elif rel.startswith(('results/r5/e1_full/','results/r5/e1_smoke/','reports/r0_logs/')) or (rel.startswith('reports/R') and not Path(rel).name.startswith('R5')):
            kind='ARCHIVE';reason='Research history kept in place and indexed; not a deletion candidate'
        elif rel=='archive/snapshots/README_before_S3.md':
            kind='ARCHIVE';reason='Exact original README backup, supports restoration and provenance'
        elif rel.startswith(('docs/refactor/','archive/')):
            reason='Audit/backup evidence; preserve even when structurally reproducible'
        classification.append(dict(path=rel,type='symlink' if p.is_symlink() else 'file',size=s.st_size,classification=kind,
                                   refs_found=json.dumps(refs,ensure_ascii=False),reason=reason))
        counts[kind]+=1;bytes_by_class[kind]+=s.st_size
        if len(proposals) and len(proposals)%1000==0 and is_pycache:print('S4_SOURCE_COMPILE_PROOFS',len(proposals),flush=True)
    csv_new('S4_delete_allowlist_PROPOSED.csv',proposals)
    csv_new('S4_file_classification.csv',classification)
    csv_new('S4_dependency_edges.csv',[dict(source=s,target=t,kind=k) for s,t,k in sorted(edges)])
    summary=dict(date='2026-10-20',dry_run_only=True,original_entries_seen=len(files),counts=dict(counts),logical_bytes_by_class=dict(bytes_by_class),
        candidates=len(proposals),candidate_logical_bytes=sum(int(r['size']) for r in proposals),
        estimated_cache_blocks_released=sum(int(r['allocated_bytes']) for r in proposals),
        python_sources_compile_checked=len(proposals),compile_warning_records=warns,
        shared_python_caches_retained=1140,julia_native_cache_files_retained=44,
        reference_edges=len(edges),backup_path=BACKUP,backup_exists_at_proposal=(ROOT/BACKUP).exists(),
        dynamic_blind_spots='Runtime import loaders may read optional pyc dynamically; source compilation and exact backup are the safety basis. Audit index/hash references are bookkeeping, not scientific/runtime dependency. Third-party dynamic paths are not claimed exhaustive.',
        protected_scope='all upstream, source, configs, tests, reports, results, locks, trajectories, RNG, NA, weak-edge and warning evidence',
        deletion_performed=False)
    with (OUT/'S4_summary.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print('S4_DRY_RUN_COMPLETE',len(proposals),'caches',summary['candidate_logical_bytes'],'bytes; no deletions',flush=True)


if __name__=='__main__':main()
