"""Run isolated Kuramoto/Lorenz pilots using pinned THIS and existing solvers."""
import argparse
import csv
import datetime
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.bridge.numeric_io import file_hash, write_rows, write_toml
from src.data.contracts import validate_splits
from src.metrics.structure import binary_metrics, edge_scores
from src.preprocessing.taylor import restore_coefficients
from src.solvers.adaptive_lasso import BICNotApplicable
from src.solvers.methods import fit_node


def save_json(path, obj):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, default=convert, ensure_ascii=False, indent=2) + '\n')


def rows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def slot(ratio): return f'noise_{round(1000*ratio):03d}'


def physical_scores(C, d, physical_ids, order):
    """Match native variable-level decode, max aggregate, then remove physical self-edges."""
    physical_ids = np.asarray(physical_ids, int)
    n = int(physical_ids.max())
    keys = [(target, *ss) for target in range(1, n+1)
            for ss in itertools.combinations([j for j in range(1, n+1) if j != target], order-1)]
    mapping = dict.fromkeys(keys, 0.)
    variable_keys, scores = edge_scores(C, d, order)
    for key, value in zip(variable_keys, scores):
        target = int(physical_ids[key[0]-1])
        sources = tuple(sorted(int(physical_ids[j-1]) for j in key[1:]))
        if len(set(sources)) == order-1 and target not in sources:
            mapping[(target, *sources)] = max(mapping[(target, *sources)], float(value))
    return keys, np.array([mapping[k] for k in keys])


def load_data(output, system, seed):
    directory = output/system/f'seed_{seed}'/'data'
    metadata = tomllib.loads((directory/'metadata.toml').read_text())
    for name, expected in metadata['file_sha256'].items():
        assert file_hash(directory/name) == expected, name
    manifest = rows(directory/'sample_manifest.csv')
    groups = np.array([int(r['trajectory_id']) for r in manifest])
    splits = np.array([r['split'] for r in manifest])
    validate_splits(groups, splits)
    read = lambda name: np.loadtxt(directory/(name+'.csv'), delimiter=',', ndmin=2)
    return dict(directory=directory, metadata=metadata, groups=groups, splits=splits,
                theta=read('theta'), Z=read('standardized_theta'), Y=read('Y'),
                d=read('monomial_indices').astype(int), means=read('feature_means').ravel(),
                scales=read('feature_scales').ravel())


def julia(task, config, output):
    command = ['/home/yg/.local/opt/julia-1.9.4/bin/julia', '--startup-file=no',
               '--project='+str(ROOT/'environment/r0/julia'), str(ROOT/'scripts/dynamics_pilot.jl'),
               task, str(config), str(output)]
    with (output/'logs'/f'julia_{task}.log').open('x') as f:
        f.write('COMMAND='+repr(command)+'\n'); f.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        f.write(f'\nEXIT_CODE={result.returncode}\n')
    if result.returncode: raise RuntimeError(f'Julia {task} failed; see logs/julia_{task}.log')


def generate_noise(data, c, output, system, seed):
    train = data['splits'] == 'train'
    scales = np.std(data['Y'][:, train], axis=1, ddof=0)
    assert np.all(scales > 0)
    code = c['systems'].index(system)+1
    unit = np.empty_like(data['Y'])
    for split_code, split in enumerate(('train', 'validation', 'test'), 1):
        mask = data['splits'] == split
        for node in range(len(scales)):
            rng = np.random.default_rng(np.random.SeedSequence([c['noise_master_seed'], seed, code, split_code, node+1]))
            unit[node, mask] = rng.standard_normal(mask.sum())
    root = output/system/f'seed_{seed}'
    np.savetxt(root/'unit_response_noise.csv', unit, delimiter=',', fmt='%.17g')
    for ratio in c['noise_ratios']:
        cell = root/slot(ratio); cell.mkdir()
        np.savetxt(cell/'Y_noisy.csv', data['Y']+ratio*scales[:, None]*unit, delimiter=',', fmt='%.17g')
        save_json(cell/'noise.json', dict(ratio=ratio, train_Y_std=scales, noise_type='response derivative only',
                  source_metadata_sha256=file_hash(data['directory']/'metadata.toml'),
                  unit_noise_sha256=file_hash(root/'unit_response_noise.csv'),
                  Y_noisy_sha256=file_hash(cell/'Y_noisy.csv')))


def python_fit(data, c, output, system, seed, ratio, method):
    trial = output/system/f'seed_{seed}'/slot(ratio)/method; trial.mkdir()
    Y = np.loadtxt(trial.parent/'Y_noisy.csv', delimiter=',', ndmin=2)
    train, val = [data['splits'] == name for name in ('train', 'validation')]
    X, Xv = data['Z'][:, train].T, data['Z'][:, val].T
    statuses, coefficients = [], []
    started = time.perf_counter()
    for node in range(Y.shape[0]):
        validation = dict(X_validation=Xv, y_validation=Y[node, val],
                          train_groups=data['groups'][train], validation_groups=data['groups'][val])
        try:
            coef, diagnostics = fit_node(method, X, Y[node, train], c, validation)
            record = dict(node_julia_index=node+1, status='OK', **diagnostics)
            coefficients.append(coef)
        except BICNotApplicable as exc:
            record = dict(node_julia_index=node+1, status='NA', reason=str(exc)); coefficients.append(None)
        except Exception as exc:
            record = dict(node_julia_index=node+1, status='FAILED', reason=str(exc), traceback=traceback.format_exc())
            coefficients.append(None)
        save_json(trial/f'node_{node+1:02d}.json', record)
        statuses.append(record['status'])
    label = 'FAILED' if 'FAILED' in statuses else 'NA' if 'NA' in statuses else 'OK'
    if label == 'OK':
        Cz = np.array(coefficients)
        C = restore_coefficients(Cz, data['means'], data['scales'])
        np.savetxt(trial/'coefficients.csv', C, delimiter=',', fmt='%.17g')
        np.savetxt(trial/'standardized_coefficients.csv', Cz, delimiter=',', fmt='%.17g')
        assert np.allclose(C@data['theta'], Cz@data['Z'], atol=1e-10, rtol=1e-12)
    else:
        available = [dict(node=i+1, feature=j+1, coefficient_standardized=float(v))
                     for i, coef in enumerate(coefficients) if coef is not None for j, v in enumerate(coef)]
        if available: write_rows(trial/'available_nodes_only.csv', available)
    write_toml(trial/'status.toml', dict(status=label, method=method, node_statuses=statuses,
               fit_seconds=time.perf_counter()-started))
    print(f'FIT system={system} seed={seed} noise={ratio} method={method} status={label}', flush=True)


def evaluate(c, output):
    metrics, candidate_rows, warnings = [], [], []
    for system in c['systems']:
        for seed in c['seeds']:
            data = load_data(output, system, seed)
            test = data['splits'] == 'test'
            truth_rows = rows(data['directory']/'physical_truth.csv')
            local_truth = np.loadtxt(data['directory']/'local_taylor_coefficients.csv', delimiter=',', ndmin=2)
            for ratio in c['noise_ratios']:
                for method in c['method_order']:
                    trial = output/system/f'seed_{seed}'/slot(ratio)/method
                    status = tomllib.loads((trial/'status.toml').read_text())
                    base = dict(system=system, seed=seed, noise_ratio=ratio, method=method,
                                status=status['status'], fit_seconds=status['fit_seconds'])
                    for path in sorted(trial.glob('node_*.json')):
                        node = json.loads(path.read_text())
                        for warning in node.get('warnings', []):
                            warnings.append(dict(**base, node=node['node_julia_index'], kind='retained_warning', message=warning))
                        if node['status'] != 'OK':
                            warnings.append(dict(**base, node=node['node_julia_index'], kind=node['status'], message=node['reason']))
                    if status['status'] != 'OK':
                        metrics.append(dict(**base, order='pairwise'));metrics.append(dict(**base, order='triple'));continue
                    C = np.loadtxt(trial/'coefficients.csv', delimiter=',', ndmin=2)
                    error = C@data['theta'][:, test]-data['Y'][:, test]
                    clean = data['Y'][:, test]
                    nmse = np.sum(error**2)/np.sum(clean**2)
                    centered_nmse = np.sum(error**2)/np.sum((clean-clean.mean(axis=1, keepdims=True))**2)
                    native_rows = rows(trial/'native_physical_edges.csv')
                    for order, label in [(2, 'pairwise'), (3, 'triple')]:
                        keys, scores = physical_scores(C, data['d'], data['metadata']['physical_node_of_variable'], order)
                        true_keys = {(int(r['target']), *sorted([int(r['source_1'])]+([int(r['source_2'])] if order == 3 else [])))
                                     for r in truth_rows if int(r['order']) == order}
                        native = {(int(r['target']), *sorted([int(r['source_1'])]+([int(r['source_2'])] if order == 3 else []))):float(r['score'])
                                  for r in native_rows if int(r['order']) == order}
                        expected_native = {k:float(s) for k, s in zip(keys, scores) if s > c['decode_threshold']}
                        assert native.keys() == expected_native.keys(), (system, seed, method, order)
                        assert all(native[k] == expected_native[k] for k in native)
                        values = binary_metrics(np.array([k in true_keys for k in keys]), scores, c['decode_threshold'])
                        metrics.append(dict(**base, order=label, **values, clean_test_NMSE=float(nmse),
                            centered_clean_test_NMSE=float(centered_nmse),
                            local_Taylor_coefficient_relative_error=float(np.linalg.norm(C-local_truth)/np.linalg.norm(local_truth))))
                        for key, score in zip(keys, scores):
                            candidate_rows.append(dict(**base, order=label, target=key[0], source_1=key[1],
                                source_2=key[2] if order == 3 else 0, truth=int(key in true_keys), score=float(score)))
    fields = list(dict.fromkeys(k for row in metrics for k in row))
    write_rows(output/'trials.csv', metrics, fields)
    write_rows(output/'edge_scores.csv', candidate_rows)
    write_rows(output/'warnings_and_unavailable.csv', warnings,
               list(warnings[0]) if warnings else ['system','seed','noise_ratio','method','status','fit_seconds','node','kind','message'])
    summaries = []
    for system, ratio, method, order in itertools.product(c['systems'], c['noise_ratios'], c['method_order'], ['pairwise','triple']):
        selected = [r for r in metrics if (r['system'],r['noise_ratio'],r['method'],r['order']) == (system,ratio,method,order)]
        good = [r for r in selected if r['status'] == 'OK']
        record = dict(system=system, noise_ratio=ratio, method=method, order=order, attempted=len(selected),
                      OK=len(good), NA=sum(r['status']=='NA' for r in selected), FAILED=sum(r['status']=='FAILED' for r in selected))
        for metric in ['F1','precision','recall','AUPRC_AP','clean_test_NMSE','centered_clean_test_NMSE']:
            record[metric+'_mean'] = float(np.mean([r[metric] for r in good])) if good else ''
        summaries.append(record)
    write_rows(output/'summary.csv', summaries)
    return metrics, summaries


def plot(c, output, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for row, system in enumerate(c['systems']):
        for col, metric in enumerate(['F1','AUPRC_AP']):
            ax = axes[row,col]
            for ratio in c['noise_ratios']:
                group = {r['method']:r for r in summaries if r['system']==system and r['noise_ratio']==ratio and r['order']=='triple'}
                y = [float(group[m][metric+'_mean']) if group[m][metric+'_mean']!='' else np.nan for m in c['method_order']]
                ax.plot(c['method_order'], y, 'o-', label=f'noise SD ratio={ratio:g}')
            ax.set(title=system+' / triple '+metric, ylim=(-.03,1.03));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Two-seed local pilot; fixed 1e-8 support threshold; NA shown as gaps')
    fig.tight_layout();fig.savefig(output/'triple_comparison.png', dpi=160);plt.close(fig)


def protect_legacy():
    manifest = ROOT/'docs/refactor/S6_delivery_manifest.csv'
    old = rows(manifest)
    for r in old:
        path = ROOT/r['path']
        if r['type'] == 'symlink':
            import hashlib
            actual = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
        else: actual = file_hash(path)
        assert actual == r['sha256'], f'Historical delivery mismatch: {path}'
    return dict(entries_checked=len(old), all_original_delivery_entries_exact=True,
                manifest_sha256=file_hash(manifest), formal_R5_files_covered=4113)


def report(c, output, summaries):
    lines = ['# Kuramoto / Lorenz 新动力学小规模试跑', '',
        '模型来源：[THIS论文](https://arxiv.org/html/2402.00078v2)。使用论文中的高阶Kuramoto及耦合Lorenz方程形式，调用固定上游THIS核心和原LASSO实现。', '',
        '本试验是2 seed工程试跑：新拓扑、非零展开中心、短轨迹和解析导数。不是论文原图复现，不是长期混沌预测实验，也不足以作统计显著性或普遍优劣结论。', '',
        '每个系统128/32/32条独立训练/验证/测试轨迹，每条3个样本；每seed训练384点。噪声仅加在导数Y，所有方法共享，0%及5%为训练导数总体标准差的比例。测试误差相对无噪声解析导数。', '',
        'Kuramoto有4个相位变量、35个三阶Taylor特征；Lorenz有3个物理振子、9个状态变量、220个特征。所有特征基于u=x−固定中心；均值/RMS、Ridge初值/权重只用训练集。', '',
        'M0=原生THIS/STLSQ，M0s=共同标准化STLSQ；M1/M2=普通/Adaptive+BIC；M1v/M3=普通/Adaptive+独立轨迹验证。STLSQ保持λ=.1、ρ=1、10次；LASSO沿用既有网格与数值预算。', '',
        '结构以物理有向边为真值，Lorenz使用上游one2dim最大系数聚合规则（新增适配层仅将字典键转为向量后排序），再排除同一物理振子及重复物理源。AP使用全部候选连续得分；F1使用预先固定abs(coef)>1e−8，不用真值选阈值。', '',
        '高阶耦合也会贡献低阶Taylor系数；本小拓扑显式包含所配置三体作用对应的目标←源二体边。网络、中心和超参数在拟合前固定，没有按测试结果调整。', '',
        '## 两seed平均三体恢复', '',
        '| 系统 | 噪声 | 方法 | OK/NA/FAILED | F1 | AP | 干净测试NMSE |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in summaries:
        if r['order']!='triple': continue
        f = lambda key: f'{r[key]:.6g}' if r[key]!='' else 'NA'
        lines.append(f"| {r['system']} | {r['noise_ratio']:.0%} | {r['method']} | {r['OK']}/{r['NA']}/{r['FAILED']} | {f('F1_mean')} | {f('AUPRC_AP_mean')} | {f('clean_test_NMSE_mean')} |")
    lines += ['', '## 解释与边界', '',
        '0% Lorenz是三阶库可精确表达的多项式系统，BIC方差门槛可能不满足，必须保留NA；验证选参路径提供对应对照。Kuramoto不是精确有限阶多项式，0%响应噪声仍存在截断误差，BIC结果不能解释为已知真实噪声估计。', '',
        '局部短轨迹测试说明该接口能否处理新方程及局部结构，不说明长时混沌吸引子、有限差分导数、观测噪声或更大网络的效果。F1与AP衡量不同：严格支持阈值下小系数假阳性可使F1偏低，即使排序AP较高。', '',
        '原始证据：[逐trial指标](trials.csv)、[两seed汇总](summary.csv)、[全部物理候选得分](edge_scores.csv)、[警告与NA/FAILED理由](warnings_and_unavailable.csv)、[模型选择锁](selection_locked.toml)、[完整性检查](legacy_integrity_after.json)。', '',
        '![三体对照](triple_comparison.png)', '',
        '该目录保留配置副本、源码哈希、轨迹与种子、解析真值、噪声、每节点选参路径/权重/KKT、原尺度局部系数及原生结构解码。旧R5不覆盖；其历史清单不更新。', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/dynamics_pilot.toml')
    parser.add_argument('--label', default=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    if not args.label or Path(args.label).name != args.label or args.label in ('.','..'):
        parser.error('--label must be a single directory name')
    c = tomllib.loads(args.config.read_text())
    output = ROOT/'results/dynamics_pilot'/args.label
    output.mkdir(parents=True, exist_ok=False);(output/'logs').mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', JULIA_NUM_THREADS='1')
    # A local overlay receives any new Julia compilation; legacy depot remains read-only in practice.
    (output/'julia_depot').mkdir();(output/'mpl_cache').mkdir()
    os.environ['JULIA_DEPOT_PATH']=str(output/'julia_depot')+':'+str(ROOT/'.julia-depot-r0')
    os.environ['MPLCONFIGDIR']=str(output/'mpl_cache')
    started = time.perf_counter()
    try:
        save_json(output/'legacy_integrity_before.json', protect_legacy())
        (output/'config.toml').write_bytes(args.config.read_bytes())
        paths = [args.config, ROOT/'scripts/dynamics_pilot.py', ROOT/'scripts/dynamics_pilot.jl', ROOT/'src/data/paper_dynamics.jl']
        paths += list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.jl'))
        save_json(output/'protocol.json', dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            scope=c['scope'], source_paper=c['source'], config=c,
            source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(paths))},
            true_labels_used_for_selection=False, test_used_for_selection=False,
            shifted_coordinates='u=x-center; center fixed in config, not fit on test data'))
        julia('prepare', args.config, output)
        rng_seeds = []
        for system in c['systems']:
            for seed in c['seeds']:
                data = load_data(output, system, seed)
                rng_seeds += data['metadata']['trajectory_rng_seeds']
                generate_noise(data, c, output, system, seed)
        assert len(rng_seeds)==len(set(rng_seeds)), 'Cross-trial initial RNG overlap'
        julia('stlsq', args.config, output)
        for system in c['systems']:
            for seed in c['seeds']:
                data = load_data(output, system, seed)
                for ratio in c['noise_ratios']:
                    for method in c['method_order']:
                        if method not in ('M0','M0s'): python_fit(data, c, output, system, seed, ratio, method)
        locks = {}
        for system, seed, ratio, method in itertools.product(c['systems'], c['seeds'], c['noise_ratios'], c['method_order']):
            trial = output/system/f'seed_{seed}'/slot(ratio)/method
            locks[str(trial.relative_to(output))] = dict(status_sha256=file_hash(trial/'status.toml'),
                coefficient_sha256=file_hash(trial/'coefficients.csv') if (trial/'coefficients.csv').exists() else 'NA_OR_FAILED')
        write_toml(output/'selection_locked.toml', dict(test_evaluated_before_lock=False,
                   structural_truth_used_for_selection=False, trials=locks))
        julia('decode', args.config, output)
        metrics, summaries = evaluate(c, output)
        for trial, expected in locks.items():
            assert file_hash(output/trial/'status.toml') == expected['status_sha256']
            if expected['coefficient_sha256']!='NA_OR_FAILED':
                assert file_hash(output/trial/'coefficients.csv') == expected['coefficient_sha256']
        plot(c, output, summaries)
        save_json(output/'legacy_integrity_after.json', protect_legacy())
        report(c, output, summaries)
        trial_statuses = [r['status'] for r in metrics if r['order']=='triple']
        completion = dict(status='COMPLETE_WITH_FAILURES' if 'FAILED' in trial_statuses else 'COMPLETE',
            trials=len(trial_statuses), counts={s:trial_statuses.count(s) for s in ('OK','NA','FAILED')},
            initial_trajectories=len(rng_seeds), unique_initial_rng_seeds=len(set(rng_seeds)),
            legacy_files_unchanged=True, wall_seconds=time.perf_counter()-started)
        save_json(output/'completion.json', completion)
        files = [p for p in output.rglob('*') if p.is_file() and 'julia_depot' not in p.parts and 'mpl_cache' not in p.parts]
        write_rows(output/'artifact_manifest.csv', [dict(path=str(p.relative_to(output)), size=p.stat().st_size, sha256=file_hash(p)) for p in sorted(files)])
        print(json.dumps(dict(output=str(output), **completion), ensure_ascii=False), flush=True)
    except Exception:
        save_json(output/'failure.json', dict(traceback=traceback.format_exc(), wall_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__': main()
