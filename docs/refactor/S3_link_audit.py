"""Local documentation links, preserving immutable historical snapshot bytes."""
import csv
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/refactor'


def scan(label):
    rows=[]
    for base,dirs,files in os.walk(ROOT,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ['.git','.venv-r0','.julia-depot-r0','upstream']]
        for name in files:
            p=Path(base)/name
            if p.suffix!='.md' and p!=ROOT/'reports/r0_logs/setup_history.log':continue
            rel=p.relative_to(ROOT).as_posix()
            for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if '://' in target or target.startswith(('#','mailto:')):continue
                clean=unquote(target.split('#')[0].strip('<>'))
                if not clean:continue
                direct=(p.parent/clean).resolve()
                status='DIRECT_OK' if direct.exists() else 'UNRESOLVED'
                resolved=direct if direct.exists() else None
                note=''
                origin=None
                if '/history/' in rel:
                    if name in ['R5_report.md','R5_learning.md']:origin=ROOT/'reports'/name
                    elif name=='r5_followup_noise_plan.md':origin=ROOT/'docs'/name
                elif rel=='archive/snapshots/README_before_S3.md':origin=ROOT/'README.md'
                if status=='UNRESOLVED' and origin is not None:
                    candidate=(origin.parent/clean).resolve()
                    if candidate.exists():
                        resolved=candidate;status='HISTORICAL_ORIGIN_RESOLVED'
                        note='Immutable snapshot retains original relative link; companion navigation uses original document base.'
                if status=='UNRESOLVED' and rel=='reports/r0_logs/setup_history.log':
                    if clean=='r0_environment_check.log':
                        resolved=p;status='HISTORICAL_MERGED_LOG';note='Original standalone check log is historical; consolidated setup record retained.'
                    elif clean=='../docs/r0_environment_setup.md':
                        resolved=ROOT/'environment/r0/README.md';status='HISTORICAL_REPLACEMENT_GUIDE'
                        note='Old installation guide retired before this refactor; current environment guide is a functional replacement, not identical historical bytes.'
                rows.append(dict(source=rel,target=target,status=status,
                    resolved_path=resolved.relative_to(ROOT).as_posix() if resolved and resolved.is_relative_to(ROOT) else '',note=note))
    report=OUT/(label+'_link_report.csv')
    with report.open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    bad=[r for r in rows if r['status']=='UNRESOLVED']
    historical=[r for r in rows if r['status'].startswith('HISTORICAL')]
    if label=='S3':
        lines=['# 历史链接伴随导航','',
            '整理日期2026-10-20。原history快照及安装历史日志保持原字节，部分相对路径按原目录解释。下表提供当前可达位置；功能替代说明不会冒充原文档。',
            '', '| 原文件 | 原链接 | 可达证据/说明 |','| --- | --- | --- |']
        for r in historical:
            source=os.path.relpath(ROOT/r['source'],ROOT/'archive')
            target=os.path.relpath(ROOT/r['resolved_path'],ROOT/'archive')
            lines.append(f"| [{r['source']}]({source}) | `{r['target']}` | [{r['resolved_path']}]({target})；{r['note']} |")
        with (ROOT/'archive/LINKS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    summary=dict(stage=label,local_links=len(rows),direct_ok=sum(r['status']=='DIRECT_OK' for r in rows),
        historical_links_resolved=len(historical),unresolved=bad,
        immutable_history_links_not_rewritten=True,scope='local own documentation and consolidated R0 log; network links and upstream docs not changed')
    with (OUT/(label+'_link_summary.json')).open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False))
    assert not bad,bad


if __name__=='__main__':scan(sys.argv[1] if len(sys.argv)>1 else 'S3')
