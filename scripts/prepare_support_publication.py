"""Prepare explicit Git publication paths without touching original experiments."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/diagnostics/this_lasso_support_3systems_10seeds_v2'
DOC=ROOT/'docs/publication/support_v2_20261026'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    completion=json.loads((RUN/'completion.json').read_text())
    assert completion['status']=='COMPLETE' and completion['models']==3240
    paths={ROOT/n for n in ['.gitignore','README.md','scripts/prepare_support_publication.py',
           'scripts/support_run.py','scripts/support_analyse.py','scripts/support_prepare.jl',
           'scripts/support_native_decode.jl','src/experiments/support_diagnostics.py','tests/test_support_diagnostics.py']}
    for directory in ['docs/experiment_reviews/support_full5_v2_20261026','reports/synchronization_audit_20261026']:
        paths.update(f for f in (ROOT/directory).rglob('*') if f.is_file())
    for n in ['preflight.json','python_tests.log','run.log']:
        f=ROOT/'reports/support_preflight_20261026'/n
        if f.exists():paths.add(f)
    for f in RUN.iterdir():
        if f.is_file() and f.name not in ['RUNNING.md','progress.json','postprocessing_progress.json',
                                        'runtime_estimate_live.json','native_decode_request.toml']:
            paths.add(f)
    for stage in ['A_fixed','B_networks']:
        paths.update(f for f in (RUN/stage).iterdir() if f.is_file() and f.suffix in ['.csv','.json','.md','.png','.pdf','.toml'])
        paths.update((RUN/stage).glob('*/config.toml'))
        for name in ['metadata.toml','feature_audit.json','monomial_indices.csv']:
            paths.update((RUN/stage).glob(f'*/*/seed_*/data/{name}'))
    for directory in ['source_snapshot','validation_logs','logs']:
        paths.update(f for f in (RUN/directory).rglob('*') if f.is_file() and f.suffix in ['.py','.jl','.json','.toml','.csv','.log'])
    # Flatten per-model state evidence into a reviewable table; coefficients stay local.
    models=[]
    for f in sorted(RUN.glob('**/M*/status.json')):
        row=json.loads(f.read_text());row['trial']=str(f.parent.relative_to(RUN))
        for k in ['node_counts']:row[k]=json.dumps(row[k],sort_keys=True) if k in row else ''
        coefficients=f.with_name('coefficients.npz')
        row['coefficient_sha256']=sha(coefficients) if coefficients.exists() else ''
        models.append(row)
    assert len(models)==3240
    table(DOC/'model_manifest.csv',models)
    original=Path('/home/yg/.codex/attachments/dd9ce634-f529-47f4-bc99-1104d26fc92f/已粘贴的文本.txt')
    if original.exists():
        (DOC/'ORIGINAL_PROPOSAL.md').write_text('> 原始五类/30种子方案；执行范围已由用户改为三类/10种子，见下方交付说明。\n\n'+original.read_text())
    readme='''# 2026-10-26 支持集与结构误检诊断：GitHub交付

执行范围：Kuramoto、Lotka–Volterra、SIS，15物理节点，每方程完整816项三阶Taylor库；0/5/10%导数响应噪声，六种原流程。A固定原网络10数据种子，B另加5网络各10数据种子。3240配置：3000 OK、240原BIC不适用、0 FAILED。

- [总报告](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/REPORT.md)
- [固定网络报告](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/A_fixed/REPORT.md)
- [多网络报告](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/B_networks/REPORT.md)
- [本次公开文件清单](published_files.csv)
- [3240模型状态与系数哈希](model_manifest.csv)
- [完整本地文件清单与哈希](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/artifact_manifest.csv)

## 发布范围

GitHub包括本次新增代码、测试、实验方案审阅、同步核查、原始与分阶段指标表、配对比较、相对alpha路径、逐方程支持统计、解码阈值诊断、有限冗余诊断、PNG/PDF图、固定种子与实例配置、数据元信息、运行记录及源码快照。

约61 GiB完整产物保留在本地。未放入Git历史：原始状态/导数与特征大矩阵、全部节点候选系数NPZ、每个BIC折点的压缩详细表、Julia包缓存、可再生预检数据。A/B的节点原生路径表分别约242 MiB/1190 MiB；公开的04_alpha_path.csv保留逐模型共同12点相对alpha诊断，05参数表保留逐方程选择信息。完整原生BIC折点数据仍在本地，不能将公开12点表误称为全部BIC折点。

published_files.csv列的是本次明确加入Git的文件；artifact_manifest.csv列的是完整本地产物，包括未上传文件。后者的某条目存在不代表该文件可从GitHub下载。

## 数据与核验

每个样本是独立t=0初始状态及解析瞬时导数，没有时间积分或同步预热。训练/验证/测试为1024/256/256，六方法共享数据和噪声底样本。三类系统沿用原五类配置的固定编号，配置中保留其他系统定义只为保持随机流命名，未运行Lorenz或Stuart–Landau。

原生THIS严格按abs(coef)>1e-8输出结构，AP使用不截断的连续分数。两者分别完成4725000项精确Julia/Python核对；6个非零但不超过阈值的分数保留用于AP。核验修复没有更改模型系数、参数或指标。

新增8项诊断测试通过。原全量Python检查144项通过，1项因本地历史R5 fixture缺失报错，见已提交验证日志。统计区间按固定网络数据种子或多网络网络均值重采样，不把同网络10次数据准备视为10个独立网络。

## 代码与复现条件

主入口为scripts/support_run.py，分析入口为scripts/support_analyse.py。Python/Julia依赖、上游源码获取方式见environment/r0/README.md。运行脚本沿用完整工作区中的既有基线目录，用它核对并复用原两个种子；本GitHub交付不包含这些大型基线输入，因此仅克隆公开表格不是完整的原始工作区。

在已配置环境、已保留原基线输入的完整工作区中，可用新的输出目录重放：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv-r0/bin/python -B -u scripts/support_run.py --out results/diagnostics/support_reproduce --workers 8
```

不要覆盖本次冻结结果目录。运行时源码快照及source_sha256记录对应实际实验；维护修复的before/after哈希与原因保存在logs中的回执。公开源码包含这些已验证的核验和恢复修复。
'''
    (DOC/'README.md').write_text(readme)
    paths.update(f for f in DOC.iterdir() if f.is_file() and f.name!='published_files.csv')
    paths={f for f in paths if f.is_file() and '__pycache__' not in f.parts}
    assert all(f.stat().st_size<90*1024**2 for f in paths)
    rows=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(paths)]
    table(DOC/'published_files.csv',rows)
    paths.add(DOC/'published_files.csv')
    pathspec=ROOT/'.git/support_publication_paths.nul'
    pathspec.write_bytes(b'\0'.join(str(f.relative_to(ROOT)).encode() for f in sorted(paths))+b'\0')
    print(json.dumps(dict(files=len(paths),MiB=sum(f.stat().st_size for f in paths)/1024**2,
                          largest=[(f.stat().st_size,str(f.relative_to(ROOT))) for f in sorted(paths,key=lambda f:f.stat().st_size,reverse=True)[:5]],
                          pathspec=str(pathspec)),indent=2))


if __name__=='__main__':main()
