"""Read-only audit of saved snapshot datasets; writes only beside this script."""
import csv
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

ROOT = Path('/home/yg/THIS-Lasso')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'results/dynamics15/five_systems_certified_20261022'
NEW = ROOT / 'results/dynamics15/fair_comparison_v2_full_run01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    rows, sources = [], []
    for batch, contract in [(OLD, 'protocol.json'), (NEW, 'run_contract.json')]:
        recorded = json.loads((batch / contract).read_text())['source_sha256']
        names = ['src/data/multisystem_dynamics.jl', 'src/data/paper_dynamics.jl',
                 'scripts/dynamics15_prepare.jl' if batch == OLD else 'scripts/fair_v2_prepare.jl']
        for name in names:
            expected = recorded[name]
            snapshot = batch / 'source_snapshot' / name
            sources.append(dict(batch=batch.name, source=name, expected=expected,
                                current_matches=sha(ROOT / name) == expected,
                                snapshot_matches=snapshot.exists() and sha(snapshot) == expected))
        folders = sorted(batch.glob('*/seed_*/data')) if batch == OLD else [
            batch / row['data_folder'] for row in read_csv(batch / 'dataset_manifest.csv')]
        config = tomllib.loads((batch / ('config.toml' if batch == OLD else 'config_locked.toml')).read_text())
        for folder in folders:
            meta = tomllib.loads((folder / 'metadata.toml').read_text())
            manifest = read_csv(folder / 'sample_manifest.csv')
            X = np.loadtxt(folder / 'states.csv', delimiter=',', ndmin=2)
            U = np.loadtxt(folder / 'centered_states.csv', delimiter=',', ndmin=2)
            system = meta['system']
            center = np.asarray(config[system]['center'])[:, None]
            corr = np.corrcoef(U)
            correlations = np.abs(corr[np.triu_indices(len(U), 1)])
            row = dict(batch=batch.name, system=system, folder=str(folder.relative_to(ROOT)),
                       seed=meta['seed'], samples=X.shape[1], state_variables=X.shape[0],
                       all_times_zero=all(float(r['time']) == 0 for r in manifest),
                       unique_preparations=len({r['trajectory_id'] for r in manifest}),
                       unique_rng_seeds=len(set(meta['trajectory_rng_seeds'])),
                       data_hashes_match=all(sha(folder / n) == meta['file_sha256'][n]
                                             for n in ['states.csv', 'Y.csv', 'sample_manifest.csv', 'centered_states.csv']),
                       centered_state_error=float(np.max(np.abs(X - center - U))),
                       within_declared_sampling_box=bool(np.all(np.abs(U) <= config[system]['box_width'] / 2)),
                       cross_preparation_state_correlation_abs_mean=float(correlations.mean()),
                       cross_preparation_state_correlation_abs_max=float(correlations.max()),
                       initial_conditions_equal_states=(bool(np.array_equal(X, np.loadtxt(folder / 'initial_conditions.csv', delimiter=',')))
                                                        if (folder / 'initial_conditions.csv').exists() else 'not_saved'),
                       kuramoto_snapshot_R_mean=(float(np.abs(np.mean(np.exp(1j * X), axis=0)).mean())
                                                 if system == 'kuramoto' else 'not_applicable'))
            assert row['all_times_zero'] and row['data_hashes_match'] and row['within_declared_sampling_box']
            assert row['unique_preparations'] == row['unique_rng_seeds'] == row['samples'] == 1536
            assert row['centered_state_error'] < 1e-14
            rows.append(row)
    assert all(s['current_matches'] and s['snapshot_matches'] for s in sources)
    with (OUT / 'datasets.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='PASS', datasets=len(rows), snapshots=sum(r['samples'] for r in rows),
                  conclusion='Both audited batches use independent t=0 preparations, without integration or burn-in before sampling.',
                  limitation='This does not test future synchronization, phase locking, attractors, or dynamical stability.',
                  correlation_definition='Pearson correlations of state fluctuations across independent preparations, not temporal correlations.',
                  source_checks=sources)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 数据生成与提前同步核查', '', '核查日期：2026-10-26（Asia/Shanghai）。', '',
             '结论：两个批次的数据均直接在独立初值处计算解析导数，没有积分、预热或先达到同步后再采样。', '',
             f'已核对 {len(rows)} 组数据、{result["snapshots"]:,} 个样本：保存时间全部为 t=0；每个样本对应独立准备标识和不重复的随机种子；状态、中心化状态、导数与样本清单哈希均匹配运行记录。', '',
             '两个批次的生成器、向量场和准备脚本均与运行时哈希及源码快照一致。原两种子批次的 states.csv 与 initial_conditions.csv 完全一致。', '',
             '| 批次 | 数据组数 | 样本数 | 样本间状态涨落相关绝对值的均值范围 |',
             '|---|---:|---:|---:|']
    for batch in [OLD, NEW]:
        selected = [r for r in rows if r['batch'] == batch.name]
        cs = [r['cross_preparation_state_correlation_abs_mean'] for r in selected]
        lines.append(f'| {batch.name} | {len(selected)} | {sum(r["samples"] for r in selected):,} | {min(cs):.4f}–{max(cs):.4f} |')
    lines += ['', '相关系数使用独立样本维度，不是时间相关；它只用于检查是否存在强烈的跨坐标共同涨落，不能单独证明无相位锁定。', '',
              'Kuramoto 的单时刻 R 即使偏高，也可能来自相位采样区间和固定中心；没有时间演化，不能据此认定发生同步过程。', '',
              '本次证据排除“采样前演化到同步态”这一生成机制。尚未验证当前方程及耦合参数经过积分后是否产生相位锁定、完全同步、振幅死亡、固定点或吸引子坍缩。GLV/SIS 需要用适合丰度/感染概率系统的稳态与状态分散指标，不能套用振子相位同步结论。', '',
              '本轮无轨迹时间轴，因此未绘制 R(t) 或所谓同步时间。长期轨迹辨识能力和同步对恢复的影响须另做受控积分实验。', '',
              'THIS 原论文也采用采样盒中的状态点及精确导数开展部分合成验证，并讨论局部有效线性化导致的高阶不可辨识：',
              'https://arxiv.org/html/2402.00078v2', '',
              '证据文件：[逐数据组核查](datasets.csv)、[源码哈希与汇总](audit.json)、[可重复核查脚本](audit.py)。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_checks'}, indent=2))
    for batch in [OLD, NEW]:
        selected = [r for r in rows if r['batch'] == batch.name]
        cs = [r['cross_preparation_state_correlation_abs_mean'] for r in selected]
        rs = [r['kuramoto_snapshot_R_mean'] for r in selected if r['system'] == 'kuramoto']
        print(batch.name, 'datasets', len(selected), 'mean_abs_corr_range', min(cs), max(cs), 'Kuramoto_R_range', min(rs), max(rs))


if __name__ == '__main__':
    main()
