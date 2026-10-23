"""Recover locked first-pilot fits after the pinned one2dim Dict-key compatibility failure."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.bridge.numeric_io import file_hash, write_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    assert Path(args.label).name==args.label and args.label not in ('.','..')
    out=ROOT/'results/dynamics_pilot'/args.label
    assert (out/'failure.json').exists() and not (out/'completion.json').exists()
    assert 'sort(::Base.KeySet' in (out/'logs/julia_decode.log').read_text()
    for name in ['trials.csv','REPORT.md','artifact_manifest.csv','postprocess_recovery.json']:
        assert not (out/name).exists(), name
    assert not list(out.glob('*/seed_*/noise_*/*/native_physical_edges.csv'))
    protocol=json.loads((out/'protocol.json').read_text())
    source=out/'source_before_decode_repair'
    for name in ['scripts/dynamics_pilot.py','scripts/dynamics_pilot.jl']:
        assert file_hash(source/Path(name).name)==protocol['source_sha256'][name]
    for path,expected in protocol['source_sha256'].items():
        if path not in ('scripts/dynamics_pilot.py','scripts/dynamics_pilot.jl'):
            assert file_hash(ROOT/path)==expected,path
    old=ast.parse((source/'dynamics_pilot.py').read_text())
    new=ast.parse((ROOT/'scripts/dynamics_pilot.py').read_text())
    old_funcs={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
    new_funcs={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
    assert old_funcs.keys()==new_funcs.keys()
    assert all(old_funcs[k]==new_funcs[k] for k in old_funcs if k!='report')
    old_julia=(source/'dynamics_pilot.jl').read_text()
    current_julia=(ROOT/'scripts/dynamics_pilot.jl').read_text()
    normalized=current_julia.replace('include(joinpath(@__DIR__,"..","src","bridge","physical_decode.jl"))\n','')
    normalized=normalized.replace('            # Upstream aggregation with Dict-key sorting compatibility only; same max rule.\n',
                                  '            # Use upstream aggregation unchanged; filter physical self/repeated sources afterwards.\n')
    normalized=normalized.replace('physical=THISAdapter.one2dim_compat(', 'physical=THISAdapter.one2dim(')
    assert normalized==old_julia,'Only post-fit aggregation compatibility may differ'
    config=ROOT/'configs/dynamics_pilot.toml'
    assert config.read_bytes()==(out/'config.toml').read_bytes()
    c=tomllib.loads(config.read_text())
    locks=tomllib.loads((out/'selection_locked.toml').read_text())['trials']
    assert len(locks)==48
    def check_locks():
        for path,expected in locks.items():
            trial=out/path
            assert file_hash(trial/'status.toml')==expected['status_sha256']
            if expected['coefficient_sha256']!='NA_OR_FAILED':
                assert file_hash(trial/'coefficients.csv')==expected['coefficient_sha256']
    check_locks()
    node_hashes={str(p.relative_to(out)):file_hash(p) for p in out.glob('*/seed_*/noise_*/*/node_*.json')}
    spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/dynamics_pilot.py')
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JULIA_NUM_THREADS='1',
                      JULIA_DEPOT_PATH=str(out/'julia_depot')+':'+str(ROOT/'.julia-depot-r0'),MPLCONFIGDIR=str(out/'mpl_cache'))
    runtime=['/home/yg/.local/opt/julia-1.9.4/bin/julia','--startup-file=no','--project='+str(ROOT/'environment/r0/julia')]
    for name,command in [
        ('physical_decode_tests.log',runtime+[str(ROOT/'tests/test_physical_decode.jl')]),
        ('julia_decode_repaired.log',runtime+[str(ROOT/'scripts/dynamics_pilot.jl'),'decode',str(config),str(out)]),
    ]:
        with (out/'logs'/name).open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            log.write(f'\nEXIT_CODE={result.returncode}\n')
        assert result.returncode==0,name
    metrics,summaries=pilot.evaluate(c,out)
    pilot.plot(c,out,summaries)
    check_locks()
    assert all(file_hash(out/path)==expected for path,expected in node_hashes.items())
    pilot.save_json(out/'legacy_integrity_after.json',pilot.protect_legacy())
    recovery=dict(status='PASS',original_failure_retained=True,models_refitted=0,
        locked_trials_unchanged=len(locks),node_diagnostics_unchanged=len(node_hashes),
        old_and_new_fitting_function_ASTs_exact=True,Julia_prepare_and_fit_source_exact=True,
        repair='Only derivative function name and sort(collect(keys(Ainf))) in a new compatibility layer; upstream original file/function untouched',
        postprocess_source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [
            ROOT/'scripts/dynamics_pilot_finish.py',ROOT/'scripts/dynamics_pilot.py',ROOT/'scripts/dynamics_pilot.jl',
            ROOT/'src/bridge/physical_decode.jl',ROOT/'tests/test_physical_decode.jl']})
    pilot.save_json(out/'postprocess_recovery.json',recovery)
    pilot.report(c,out,summaries)
    with (out/'REPORT.md').open('a') as f:
        f.write('\n## 首轮后处理兼容性修复\n\n原生one2dim在Julia1.9直接sort字典键时报错，首轮[失败日志](logs/julia_decode.log)及[failure.json](failure.json)原样保留。新增适配层只把keys收集为向量再排序，未修改原论文源码。所有48个已锁模型、系数和全部节点诊断保持，0次重新拟合；生成/拟合函数与原执行源码核对一致。修复后6项Julia聚合断言及全部物理边的Python/Julia精确比对通过，见[修复记录](postprocess_recovery.json)。\n')
    statuses=[r['status'] for r in metrics if r['order']=='triple']
    completion=dict(status='COMPLETE_WITH_FAILURES' if 'FAILED' in statuses else 'COMPLETE',
        trials=len(statuses),counts={s:statuses.count(s) for s in ('OK','NA','FAILED')},
        legacy_files_unchanged=True,postprocess_recovered=True,models_refitted=0)
    pilot.save_json(out/'completion.json',completion)
    files=[p for p in out.rglob('*') if p.is_file() and 'julia_depot' not in p.parts and 'mpl_cache' not in p.parts]
    write_rows(out/'artifact_manifest.csv',[dict(path=str(p.relative_to(out)),size=p.stat().st_size,sha256=file_hash(p)) for p in sorted(files)])
    print(json.dumps(completion,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
