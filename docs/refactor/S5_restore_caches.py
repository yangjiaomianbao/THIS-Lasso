"""Restore only the exact optional caches in the verified S5 backup; default is dry-run."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/refactor'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore',action='store_true',help='restore missing exact listed files; never overwrite a different file')
    args=p.parse_args()
    auth=json.loads((OUT/'S5_authorization.json').read_text())
    archive=ROOT/auth['backup_archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==auth['backup_sha256']
    rows={r['path']:r for r in csv.DictReader((OUT/'S5_delete_allowlist_APPROVED.csv').open())}
    restored=missing=identical=0
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert {m.name for m in members}==set(rows)
        for member in members:
            row=rows[member.name];target=ROOT/member.name
            assert member.isfile() and member.name.startswith('.venv-r0/') and target.suffix=='.pyc'
            assert target.resolve().is_relative_to(ROOT) and not target.is_symlink()
            with tar.extractfile(member) as f:data=f.read()
            assert hashlib.sha256(data).hexdigest()==row['sha256']
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'], 'Different file exists: '+member.name
                identical+=1;continue
            missing+=1
            if args.restore:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(data)
                target.chmod(int(row['mode']))
                os.utime(target,ns=(int(row['mtime_ns']),int(row['mtime_ns'])))
                restored+=1
    print(json.dumps(dict(dry_run=not args.restore,verified_archive=True,listed=len(rows),missing=missing,
                          identical_existing=identical,restored=restored)))


if __name__=='__main__':main()
