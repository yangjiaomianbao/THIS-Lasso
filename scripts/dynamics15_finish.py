"""Independently validate and report the locked five-system 15-node benchmark."""
import argparse
from collections import Counter
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from src.bridge.numeric_io import file_hash,write_toml,write_rows
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/dynamics15.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
pilot=r.pilot
NAMES=dict(kuramoto='Kuramoto',lorenz='Lorenz',lotka_volterra='Lotka–Volterra',stuart_landau='Stuart–Landau',sis='SIS')
METHODS=dict(M0D='STLSQ raw',M0sD='STLSQ std',M1B='LASSO BIC',M2B='Adaptive BIC',M1vW='LASSO val',M3W='Adaptive val',M1vC='LASSO val',M3C='Adaptive val')


def verify_recorded_source(out,name,expected):
    actual=file_hash(ROOT/name)
    if actual==expected:
        return
    # Allow only the recorded maintenance edit, with the original run source intact.
    receipt_path=ROOT/'docs/maintenance/cleanup_dynamics30_20261023/cleanup_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='DELETED_AND_VERIFIED',receipt_path
    change=receipt['source_changes'][name]
    assert change['before_sha256']==expected and change['after_sha256']==actual,name
    assert file_hash(out/'source_snapshot'/name)==expected,name


def validate(c,out):
    complete=json.loads((out/'completion.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    locks=json.loads((out/'selection_locked.json').read_text())['trials']
    for name,expected in protocol['source_sha256'].items():verify_recorded_source(out,name,expected)
    resumed_files=0
    if (out/'execution_resume.json').exists():
        resume=json.loads((out/'execution_resume.json').read_text())
        assert file_hash(ROOT/'scripts/dynamics15_parallel.py')==resume['source_sha256']
        for name,expected in resume['preserved_completed_files'].items():assert file_hash(out/name)==expected,name
        resumed_files=len(resume['preserved_completed_files'])
    if (out/'execution_accelerator.json').exists():
        accelerator=json.loads((out/'execution_accelerator.json').read_text())
        for name,expected in accelerator['source_sha256'].items():verify_recorded_source(out,name,expected)
        assert accelerator['uniform_backend_for_all_validation_models'] and accelerator['models_reused']==0
    previous=ROOT/'results/dynamics15/five_systems_full_20261022'
    if (out/'previous_run_integrity.json').exists():
        receipt=json.loads((out/'previous_run_integrity.json').read_text())
        assert file_hash(previous/'stopped_artifact_manifest.csv')==receipt['manifest_sha256']
        for entry in pilot.rows(previous/'stopped_artifact_manifest.csv'):
            assert file_hash(previous/entry['path'])==entry['sha256'],entry['path']
    def check_locks():
        assert len(locks)==120
        for name,expected in locks.items():
            assert file_hash(out/name/'status.json')==expected['status_sha256']
            if expected['coefficients_sha256']!='UNAVAILABLE':assert file_hash(out/name/'coefficients.npz')==expected['coefficients_sha256']
    check_locks();datasets=[];rng_ids=[];expressions=[];checked_nodes=0;kkt_checks=0;backends=Counter()
    for system,seed in itertools.product(c['systems'],c['seeds']):
        data=r.d30.load_data(out,system,seed);folder=data['folder'];meta=data['meta']
        if (previous/system/f'seed_{seed}'/'data/metadata.toml').exists():
            original_meta=tomllib.loads((previous/system/f'seed_{seed}'/'data/metadata.toml').read_text())
            assert meta['file_sha256']==original_meta['file_sha256']
        assert meta['physical_nodes']==15 and meta['N']==15*c[system]['coordinates_per_node']
        assert meta['P']==(meta['N']+1)*(meta['N']+2)*(meta['N']+3)//6
        rng_ids.extend(meta['trajectory_rng_seeds'])
        assert len(set(data['groups']))==len(data['groups'])==1536
        X=np.loadtxt(folder/'states.csv',delimiter=',',ndmin=2)
        if system=='sis':assert np.all((X>0)&(X<1))
        if system=='lotka_volterra':assert np.all(X>0)
        phi=np.load(folder/'theta.npy',mmap_mode='r')
        tr=data['splits']=='train';val=data['splits']=='validation';test=data['splits']=='test'
        assert [int(mask.sum()) for mask in (tr,val,test)]==[1024,256,256]
        assert np.array_equal(data['d'][0],np.zeros(3)) and np.all(phi[:,0]==1)
        raw=np.asfortranarray(phi[tr]);means=raw.mean(axis=0);means[0]=0.
        scales=np.sqrt(np.mean((raw-means)**2,axis=0));scales[0]=1.
        assert np.array_equal(means,np.load(folder/'feature_means.npy'))
        assert np.array_equal(scales,np.load(folder/'feature_scales.npy'))
        Z=np.load(folder/'Z_train.npy',mmap_mode='r')
        for label,mask in [('train',tr),('validation',val),('test',test)]:
            assert np.array_equal((phi[mask]-means)/scales,np.load(folder/f'Z_{label}.npy',mmap_mode='r'))
        truth_rows=pilot.rows(folder/'local_taylor_sparse.csv')
        trueC=sparse.csr_matrix(([float(a['coefficient']) for a in truth_rows],
            ([int(a['node'])-1 for a in truth_rows],[int(a['feature'])-1 for a in truth_rows])),shape=(meta['N'],meta['P']))
        oracle_nmse=float(np.sum((trueC@phi[test].T-data['Y'][:,test])**2)/np.sum(data['Y'][:,test]**2))
        assert oracle_nmse<(1e-3 if system=='kuramoto' else 1e-24)
        for order,key in [(2,'pair_edges'),(3,'triple_edges')]:
            keys,scores=pilot.physical_scores(trueC.toarray(),data['d'],meta['physical_node_of_variable'],order)
            positive={(int(a[0]),*sorted(int(x) for x in a[1:order])) for a in c[system][key]}
            assert {k for k,v in zip(keys,scores) if v>c['decode_threshold']}==positive
        expressions.append(dict(system=system,seed=seed,physical_nodes=15,state_variables=meta['N'],full_features=meta['P'],
            train_samples=1024,validation_samples=256,test_samples=256,analytic_Taylor_test_NMSE=oracle_nmse,
            pair_candidate_count=210,triple_candidate_count=1365,true_pair_equation_coverage=1.,true_triple_equation_coverage=1.))
        unit=np.loadtxt(folder.parent/'unit_response_noise.csv',delimiter=',',ndmin=2)
        sigma=np.std(data['Y'][:,tr],axis=1,ddof=0);trials=[]
        for ratio in c['noise_ratios']:
            root=folder.parent/pilot.slot(ratio);Y=np.loadtxt(root/'Y_noisy.csv',delimiter=',',ndmin=2)
            assert np.array_equal(Y,data['Y']+ratio*sigma[:,None]*unit)
            noise=json.loads((root/'noise.json').read_text())
            assert noise['unit_noise_sha256']==file_hash(folder.parent/'unit_response_noise.csv')
            assert noise['Y_noisy_sha256']==file_hash(root/'Y_noisy.csv')
            init=np.load(root/'Ridge_initial_standardized.npy',mmap_mode='r')
            for method in c['method_order']:
                trial=root/method;state=json.loads((trial/'status.json').read_text())['status']
                for f in trial.glob('node_*.json'):
                    diag=json.loads(f.read_text());assert diag['process_isolated'];checked_nodes+=1
                    if method in ('M1vC','M3C'):
                        backends[(system,method,diag['status'],diag.get('numerical_backend','FAILED or unavailable'))]+=1
                    if diag['status']=='OK':assert diag['kkt_scaled_violation']<=c['lasso']['kkt_tolerance']+1e-12
                if state!='OK':continue
                C=sparse.load_npz(trial/'coefficients.npz')
                assert np.isfinite(C.data).all()
                if method in ('M1B','M2B','M1vW','M3W','M1vC','M3C'):
                    for node in range(meta['N']):
                        coef=np.load(trial/f'node_{node+1:03d}_solution.npz')['coefficient_standardized']
                        rawcoef=r.restore_coefficients(coef[None,:],means,scales)[0]
                        assert np.array_equal(rawcoef,C.getrow(node).toarray().ravel())
                        if method in ('M1vW','M3W','M1vC','M3C'):
                            diag=json.loads((trial/f'node_{node+1:03d}.json').read_text())
                            a=np.ones(meta['P']-1) if method in ('M1vW','M1vC') else (np.abs(init[node])+c['adaptive']['eps'])**c['adaptive']['gamma']
                            gradient=Z[:,1:].T@(Z@coef-Y[node,tr])/len(Z)
                            threshold=diag['alpha']/a;nz=coef[1:]!=0
                            violation=np.maximum(np.abs(gradient)-threshold,0.)
                            violation[nz]=np.abs(gradient[nz]+threshold[nz]*np.sign(coef[1:][nz]))
                            scale=max(1.,float(np.max(np.abs(Z[:,1:].T@(Y[node,tr]-Y[node,tr].mean())/len(Z)))))
                            assert float(np.max(violation))/scale<=c['lasso']['kkt_tolerance']+1e-10
                            assert not diag['validation_path'][diag['selected_path_index']]['convergence_warning']
                            assert max(v['full_library_kkt'] for v in diag['validation_path'])<=c['lasso']['kkt_tolerance']+1e-12
                            kkt_checks+=1
                coords=C.tocoo()
                np.savetxt(trial/'coefficient_coordinates.csv',np.column_stack((coords.row+1,coords.col+1,coords.data)),delimiter=',',fmt=['%d','%d','%.17g'])
                trials.append(str(trial.relative_to(out)))
        datasets.append(dict(data=str(folder.relative_to(out)),N=meta['N'],group_size=c[system]['coordinates_per_node'],trials=trials))
    assert len(rng_ids)==len(set(rng_ids))==15360
    write_rows(out/'library_and_expression_audit.csv',expressions)
    write_rows(out/'solver_backend_counts.csv',[dict(system=k[0],method=k[1],status=k[2],backend=k[3],nodes=v) for k,v in sorted(backends.items())])
    write_toml(out/'native_decode_request.toml',dict(datasets=datasets))
    env=dict(os.environ,JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JULIA_DEPOT_PATH=str(out/'julia_depot')+':'+str(ROOT/'.julia-depot-r0'))
    command=['/home/yg/.local/opt/julia-1.9.4/bin/julia','--startup-file=no','--project='+str(ROOT/'environment/r0/julia'),str(ROOT/'scripts/dynamics30_native_decode.jl'),str(out)]
    with (out/'logs/native_decode_validation.log').open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0
    edges=0
    for system,seed in itertools.product(c['systems'],c['seeds']):
        data=r.d30.load_data(out,system,seed)
        for ratio,method in itertools.product(c['noise_ratios'],c['method_order']):
            trial=out/system/f'seed_{seed}'/pilot.slot(ratio)/method
            if json.loads((trial/'status.json').read_text())['status']!='OK':continue
            C=sparse.load_npz(trial/'coefficients.npz').toarray();native=pilot.rows(trial/'native_physical_edges.csv')
            for order in (2,3):
                keys,scores=pilot.physical_scores(C,data['d'],data['meta']['physical_node_of_variable'],order)
                expected={k:float(v) for k,v in zip(keys,scores) if v>c['decode_threshold']}
                actual={(int(a['target']),*sorted([int(a['source_1'])]+([int(a['source_2'])] if order==3 else []))):float(a['score']) for a in native if int(a['order'])==order}
                assert expected==actual,str(trial);edges+=len(expected)
    metrics=pilot.rows(out/'trials.csv');assert len(metrics)==240
    for a in metrics:
        if a['status']=='OK':assert int(a['candidate_count'])==(210 if a['order']=='pairwise' else 1365)
        else:assert a['F1']==a['AUPRC_AP']==''
    check_locks()
    audit=dict(status='PASS',locked_models=120,scored_models=complete['counts']['OK'],
        native_decoded_edges_exact=edges,independent_global_weighted_KKT_checks=kkt_checks,
        process_isolated_node_diagnostics=checked_nodes,independent_rng_names=len(rng_ids),
        truth_or_channel_screening=False,all_true_equation_source_sets_retained=True,
        training_only_standardization_exact=True,response_noise_recomputed_exact=True,
        completed_files_reused_without_change=resumed_files,
        certified_lars_nodes=sum(v for k,v in backends.items() if k[3].startswith('Full weighted LARS')),
        working_set_fallback_nodes=sum(v for k,v in backends.items() if k[3].startswith('Original working-set')),
        original_result_integrity=r.integrity(),postprocess_source_sha256={name:file_hash(ROOT/name) for name in ['scripts/dynamics15_finish.py','scripts/dynamics30_native_decode.jl']})
    pilot.save_json(out/'validation.json',audit);print(json.dumps(audit,ensure_ascii=False),flush=True)


def report(c,out):
    complete=json.loads((out/'completion.json').read_text());audit=json.loads((out/'validation.json').read_text());assert audit['status']=='PASS'
    summary=pilot.rows(out/'summary.csv');sizes=pilot.rows(out/'library_and_expression_audit.csv')
    os.environ['MPLCONFIGDIR']=str(out/'mpl_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('F1','AUPRC_AP'):
        fig,axes=plt.subplots(5,2,figsize=(13,14),sharey=True,layout='constrained')
        for row,system in enumerate(c['systems']):
            for col,ratio in enumerate(c['noise_ratios']):
                ax=axes[row,col]
                for j,order in enumerate(('pairwise','triple')):
                    values=[next(a for a in summary if (a['system'],a['method'],float(a['noise_ratio']),a['order'])==(system,m,ratio,order)) for m in c['method_order']]
                    positions=np.arange(6)+(j-.5)*.36
                    ax.bar(positions,[float(a[metric+'_mean']) if a[metric+'_mean'] else np.nan for a in values],.34,label=order)
                    for x,a in zip(positions,values):
                        if not a[metric+'_mean']:
                            label='FAIL/NA' if int(a['FAILED']) and int(a['NA']) else 'FAILED' if int(a['FAILED']) else 'NA'
                            ax.text(x,.015,label,rotation=90,ha='center',fontsize=6)
                        elif int(a['OK'])<2:ax.text(x,float(a[metric+'_mean'])+.02,'n='+a['OK'],ha='center',fontsize=7)
                    if metric=='AUPRC_AP':ax.axhline(45/210 if order=='pairwise' else 15/1365,color=f'C{j}',ls=':',lw=.7)
                ax.set(title=f'{NAMES[system]} / {ratio:.0%} response noise',ylim=(0,1.08),xticks=range(6),xticklabels=[METHODS[m] for m in c['method_order']])
                ax.tick_params(axis='x',rotation=25,labelsize=7);ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle(f'15 physical nodes; full cubic library; 1024 independent training preparations\n{metric}: means over available seeds (2 attempted), fixed coefficient threshold 1e-8')
        fig.savefig(out/(metric+'_comparison.png'),dpi=160);plt.close(fig)
    failures=Counter()
    for f in out.glob('*/seed_*/noise_*/M*/node_*.json'):
        a=json.loads(f.read_text())
        if a['status']!='OK':failures[(f.parts[-5],f.parent.name,a['status'],a['reason'].split(' Explicitly')[0])]+=1
    write_rows(out/'node_failure_reasons.csv',[dict(system=k[0],method=k[1],status=k[2],reason=k[3],nodes=v) for k,v in sorted(failures.items())])
    fmt=lambda v:f'{float(v):.4f}' if v!='' else 'NA'
    lines=['# 15节点、五类高阶动力学实验','',
        f"已完成120个模型：{complete['counts']['OK']} OK、{complete['counts']['NA']} NA、{complete['counts']['FAILED']} FAILED；独立核验{audit['status']}。每系统2个种子，不作总体优越性或显著性结论。",'',
        '五类模型为Kuramoto、Lorenz、高阶Lotka–Volterra、Stuart–Landau和SIS。每类15物理节点，相同45条有向成对边、15条三体边。三体目标—源二体面均存在。SL每节点2坐标，Lorenz每节点3坐标；其余每节点1变量。GLV为竞争性生态方程，SL为共轭坐标耦合，SIS为确定性节点概率闭合。具体方程、来源和稀疏有向改编边界见[预声明](../../../docs/dynamics15_five_systems_protocol.md)。','',
        '## 数据和函数库','',
        '每seed1024/256/256个训练/验证/测试独立初值，每个只取t=0状态及解析导数。此为局部瞬时向量场辨识，未使用长期轨迹、有限差分或真实观测。每坐标围绕固定非零中心独立均匀取值；GLV为正丰度，SIS概率在(0,1)。噪声为0%/5%导数响应高斯噪声，标准差仅由训练干净导数估计，分割流独立、方法共享。没有状态或传播过程噪声。','',
        '完整原生THIS三阶库，无Pearson、已知耦合通道或真实边筛选。所有真实方程源集合均保留；这由解析真系数的结构读出核对，不将真系数交给拟合。下表Taylor误差是解析系数的诊断结果，不是模型拟合成绩。','',
        '| 系统 | 物理节点 | 状态维度 | 完整库P | 训练K | 解析Taylor测试NMSE范围 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for system in c['systems']:
        v=[a for a in sizes if a['system']==system];errors=[float(a['analytic_Taylor_test_NMSE']) for a in v]
        lines.append(f"| {NAMES[system]} | 15 | {v[0]['state_variables']} | {v[0]['full_features']} | 1024 | {min(errors):.3g}–{max(errors):.3g} |")
    lines+=['','相较30节点轮，本轮同时调整节点数、独立样本量、采样方式并取消Pearson硬筛选；成对/三体每目标入度保持3/1，所以候选边密度也提高。结果改善或退化不能全部归因于节点数。','',
        '## 方法与指标','',
        'STLSQ raw/std为原算法代数的primal/dual线性求解变体，λ=.1、ρ=1、10次；常数惩罚及停止规则保持。普通/Adaptive BIC调用原求解器，要求有效全模型噪声方差；K<=P或零噪声精确多项式使其NA是适用性限制。M1vC/M3C为普通/Adaptive完整库验证选择，所有40个模型统一采用LARS加速、全部12个原alpha点逐一检查全库KKT，不通过即回退原工作集求解。均匀缩放求解坐标和alpha只改善数值范围，不改变原单位目标。最多5000个LARS折点，回退保留原100000次CD上限；无固定删候选。所有方法按训练数据统一标准化，恢复原单位后结构解码。','',
        'STLSQ目前使用固定默认阈值，未获得与LASSO同等的验证参数搜索；raw/std使用同一数值阈值也不代表同一原单位剪枝强度。比较可描述这组具体流程，不足以证明LASSO普遍优于经过充分调参的STLSQ。','',
        '下表只平均OK种子，显示OK/2；NA/FAILED不当零。F1固定abs(coef)>1e-8；AP使用全部210个二体、1365个三体物理候选。预测NMSE对照干净测试导数，低预测误差不等于结构恢复成功。','',
        '15节点下正例比例提高：二体45/210≈0.2143，三体15/1365≈0.0110，也是常数得分的AP基线（图中点线）。30节点轮的对应基线为0.1034/0.00246，跨节点数比较AP应考虑此差异。','',
        '评价范围限定为二体和三体。完整三次多项式库也允许目标以外三个不同物理源的乘积，对应四体候选；这些项可以参与回归，但未计入本表二/三体F1，不能把本表解释为全部阶次结构的准确率。','',
        '| 系统 | 噪声 | 方法 | OK/2 | 二体F1/AP | 三体F1/AP | 测试NMSE |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for system,ratio,method in itertools.product(c['systems'],c['noise_ratios'],c['method_order']):
        get=lambda order:next(a for a in summary if (a['system'],float(a['noise_ratio']),a['method'],a['order'])==(system,ratio,method,order))
        a,b=get('pairwise'),get('triple')
        lines.append(f"| {NAMES[system]} | {ratio:.0%} | {METHODS[method]} | {a['OK']}/2 | {fmt(a['F1_mean'])}/{fmt(a['AUPRC_AP_mean'])} | {fmt(b['F1_mean'])}/{fmt(b['AUPRC_AP_mean'])} | {float(a['clean_test_NMSE_mean']):.3g} |" if a['clean_test_NMSE_mean'] else f"| {NAMES[system]} | {ratio:.0%} | {METHODS[method]} | 0/2 | NA | NA | NA |")
    lines+=['','![F1](F1_comparison.png)','','![AP](AUPRC_AP_comparison.png)','',
        '## 完整性与可复现性','',
        f"40项新Julia方程/契约测试、48项正式数据生成断言及116项Python测试通过（含5项LARS一致性/全库证书测试）。全部{audit['native_decoded_edges_exact']}条接受模型结构边通过Julia原生解码与Python精确比较；另独立检查{audit['independent_global_weighted_KKT_checks']}个验证选参方程的全库KKT。15,360个初值RNG命名不重复，分割不重叠，训练标准化及响应噪声重算精确。原S6交付25368项、R5、小规模试跑及30节点正式轮6427项哈希保护通过。",'',
        '模型状态和系数在真值/测试评分前统一锁定。任一方程FAILED则整个模型FAILED；无FAILED但有NA则整个模型NA，可用节点单独保存。失败诊断见[node_failure_reasons.csv](node_failure_reasons.csv)及各模型目录，无法评分不意味着零F1。','',
        '初始工作集版本因极低alpha路径耗时长，在任何真值/测试评分前停止，其部分结果、失败及日志冻结保存。本轮另建目录、重新生成相同seed数据并重新拟合全部120个模型，40个验证模型统一数值策略，避免混用两种求解精度。未按原轮失败状态选择性重试，也未读取结构真值/测试来调整加速器；[加速记录](execution_accelerator.json)可追溯。','',
        f"验证方程中{audit['certified_lars_nodes']}个直接通过LARS证书，{audit['working_set_fallback_nodes']}个在回退后通过；逐系统/方法计数见[数值后端计数](solver_backend_counts.csv)。两次数据生成的全部10份原始CSV哈希一致。",'',
        '[逐模型指标](trials.csv)、[汇总](summary.csv)、[全候选得分](edge_scores.csv)、[库与表达能力诊断](library_and_expression_audit.csv)、[锁定记录](selection_locked.json)、[独立校验](validation.json)、[完成状态](completion.json)、[文件清单](artifact_manifest.csv)。','',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv-r0/bin/python -B scripts/dynamics15_accelerated.py --label five_systems_certified_reproduce',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv-r0/bin/python -B scripts/dynamics15_finish.py --label five_systems_certified_reproduce',
        '```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    index=ROOT/'reports/dynamics15_five_systems_20261022.md'
    if index.exists():assert out.name in index.read_text()
    index.write_text('# 15节点、五类动力学\n\n[完整报告及图表](../results/dynamics15/'+out.name+'/REPORT.md)。\n\n'
        'Kuramoto、Lorenz、Lotka–Volterra、Stuart–Landau、SIS均为15物理节点，完整三阶库、无Pearson筛选。\n\n'
        f"120个模型：{complete['counts']['OK']} OK，{complete['counts']['NA']} NA，{complete['counts']['FAILED']} FAILED。\n")
    pilot.save_json(out/'delivery.json',dict(status='DELIVERED',report='REPORT.md',index=str(index.relative_to(ROOT)),
        validation_sha256=file_hash(out/'validation.json'),report_source_sha256=file_hash(Path(__file__))))
    files=[f for f in out.rglob('*') if f.is_file() and not {'julia_depot','mpl_cache'}.intersection(f.parts) and f.name!='artifact_manifest.csv']
    write_rows(out/'artifact_manifest.csv',[dict(path=str(f.relative_to(out)),size=f.stat().st_size,sha256=file_hash(f)) for f in sorted(files)])
    print('REPORT',out/'REPORT.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='five_systems_certified_20261022')
    args=parser.parse_args();assert Path(args.label).name==args.label and args.label not in ('.','..')
    out=ROOT/'results/dynamics15'/args.label;c=tomllib.loads((out/'config.toml').read_text())
    assert (out/'completion.json').exists()
    protocol=json.loads((out/'protocol.json').read_text())
    for name,expected in protocol['source_sha256'].items():
        source=ROOT/name;assert file_hash(source)==expected
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert file_hash(target)==expected
        else:shutil.copy2(source,target)
    for name in ['scripts/dynamics15_parallel.py','scripts/dynamics15_finish.py',
                 'scripts/dynamics30_native_decode.jl','tests/test_multisystem_dynamics.jl',
                 'tests/test_lars_validation.py','docs/dynamics15_five_systems_protocol.md']:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert file_hash(target)==file_hash(ROOT/name)
        else:shutil.copy2(ROOT/name,target)
    for name in ['python_tests_20261022.log','python_tests_accelerated_20261022.log',
                 'run_five_systems_20261022.log','run_five_systems_parallel_20261022.log','run_five_systems_certified_20261022.log']:
        source=out.parent/name
        if source.exists():shutil.copy2(source,out/'logs'/name)
    if not (out/'validation.json').exists():validate(c,out)
    report(c,out)


if __name__=='__main__':main()
