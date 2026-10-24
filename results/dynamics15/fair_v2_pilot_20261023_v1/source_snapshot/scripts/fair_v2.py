"""Independent V2 entry point. No legacy fits, reports or hashes are rewritten."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JULIA_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import importlib.util
import itertools
import json
import multiprocessing
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits,threadpool_info
from src.bridge.numeric_io import write_rows,write_toml
from src.experiments.fair_v2 import (SYSTEMS,METHODS,NAMES,sha,save_json,load_csv,make_plan,read_plan,
    instance_config,fit_scaling,noise,solve,restore_coefficients,initial_ridge)
from src.experiments.fair_v2_statistics import export_statistics,METRICS
from src.metrics.structure import binary_metrics
spec=importlib.util.spec_from_file_location('fair_v2_legacy_helpers',ROOT/'scripts/dynamics30.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
pilot=legacy.pilot
JULIA='/home/yg/.local/opt/julia-1.9.4/bin/julia'
DEFAULT_PLAN=ROOT/'configs/fair_comparison_v2'
AUDIT=ROOT/'docs/experiments/fair_comparison_v2'


class ResourceMonitor:
    """Linux /proc process-tree RSS, including workers/Julia; shared pages counted per process."""
    def __init__(self):
        self.stop=threading.Event();self.peak=0;self.max_processes=0;self.samples=0
    def sample(self):
        todo=[os.getpid()];seen=set();rss=0
        while todo:
            pid=todo.pop()
            if pid in seen:continue
            seen.add(pid)
            try:
                todo.extend(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
                for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        self.peak=max(self.peak,rss);self.max_processes=max(self.max_processes,len(seen));self.samples+=1
    def loop(self):
        while not self.stop.is_set():self.sample();self.stop.wait(.25)
    def __enter__(self):
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();self.sample()
    def result(self):
        return dict(peak_process_tree_rss_mib=self.peak/2**20,peak_process_count=self.max_processes,
                    samples=self.samples,interval_seconds=.25,shared_pages_may_be_counted_multiple_times=True)


def source_files():
    return sorted(set([Path(__file__).relative_to(ROOT),Path('scripts/fair_v2_prepare.jl'),
        Path('scripts/dynamics30.py'),Path('scripts/dynamics_pilot.py'),Path('scripts/dynamics30_native_decode.jl')]+
        [f.relative_to(ROOT) for f in (ROOT/'src').rglob('*') if f.suffix in ('.py','.jl')]))


def audit_old(output_name):
    baseline=json.loads((AUDIT/'baseline_sha256.json').read_text())
    for name,record in baseline.items():
        path=ROOT/name
        digest=hashlib.sha256(os.readlink(path).encode()).hexdigest() if record['symlink'] else sha(path)
        assert digest==record['sha256'],name
    upstream={}
    for repo in ['THIS','Su-AdaptiveLASSO']:
        path=ROOT/'upstream'/repo
        assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain'])
        upstream[repo]=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    result=dict(status='PASS',historical_files_unchanged=len(baseline),upstream=upstream,
                existing_R5_and_15_node_results_unchanged=True)
    save_json(AUDIT/output_name,result);return result


def worker(task):
    """Fitting process sees only train/validation arrays, no test arrays or topology."""
    threadpool_limits(limits=1)
    started=time.perf_counter();cpu=time.process_time()
    node,method,paths,solver_config,trial,fingerprint=task
    trial=Path(trial);prefix=trial/f'node_{node+1:03d}'
    try:
        Z=np.load(paths['Z_train'],mmap_mode='r');Zv=np.load(paths['Z_validation'],mmap_mode='r')
        y=np.load(paths['y_train'],mmap_mode='r')[node];yv=np.load(paths['y_validation'],mmap_mode='r')[node]
        initial=np.load(paths['ridge_initial'],mmap_mode='r')[node]
        coef,diag=solve(method,Z,y,Zv,yv,initial,solver_config)
        prediction=Z@coef;vp=Zv@coef
        if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite fit')
        selected=diag['validation_path'][diag['selected_path_index']]
        assert np.isclose(np.mean((vp-yv)**2),selected['MSE'],rtol=1e-10,atol=1e-20)
        assert abs(float(np.mean(prediction-y)))<=1e-10*max(1.,float(np.std(y)))
        if method!='STLSQ_val':
            a=np.ones(len(initial)) if method=='LASSO_val' else np.abs(initial)+1e-6
            g=Z[:,1:].T@(prediction-y)/len(y);nz=coef[1:]!=0;pen=diag['alpha']/a
            v=np.maximum(np.abs(g)-pen,0.);v[nz]=np.abs(g[nz]+pen[nz]*np.sign(coef[1:][nz]))
            scale=max(1.,float(np.max(np.abs(Z[:,1:].T@(y-y.mean())/len(y)))))
            diag['independent_selected_kkt']=float(np.max(v))/scale
            assert diag['independent_selected_kkt']<=solver_config['lasso']['kkt_tolerance']+1e-12
        np.savez_compressed(str(prefix)+'_solution.npz',coefficient_standardized=coef)
        result=dict(status='OK',**diag)
    except Exception as exc:
        result=dict(status='FAILED',method=method,reason=str(exc),traceback=traceback.format_exc(),
                    diagnostics=getattr(exc,'diagnostics',{}))
    result.update(node=node+1,input_fingerprint=fingerprint,wall_seconds=time.perf_counter()-started,
        process_cpu_seconds=time.process_time()-cpu,worker_pid=os.getpid(),
        worker_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        blas_threads=[x['num_threads'] for x in threadpool_info()])
    save_json(str(prefix)+'.json',result)
    return result


def prepare_dataset(out,instance,system,repetition,c):
    data=legacy.load_data(instance,system,repetition['data_seed']);folder=data['folder']
    if not (folder/'feature_audit.json').exists():legacy.features(data)
    phi=np.load(folder/'theta.npy',mmap_mode='r');train=data['splits']=='train'
    means,scales=fit_scaling(phi[train])
    assert np.array_equal(means,np.load(folder/'feature_means.npy'))
    assert np.array_equal(scales,np.load(folder/'feature_scales.npy'))
    for split in ('train','validation','test'):
        assert np.array_equal((phi[data['splits']==split]-means)/scales,np.load(folder/f'Z_{split}.npy',mmap_mode='r'))
    unit,sigma,streams=noise(data['Y'],train,repetition,SYSTEMS.index(system)+1,data['splits'])
    noise_file=folder/'unit_response_noise.npy'
    if noise_file.exists():assert np.array_equal(np.load(noise_file),unit)
    else:np.save(noise_file,unit)
    save_json(folder/'noise_manifest.json',dict(streams=streams,train_sigma=sigma,unit_sha256=sha(noise_file)))
    record=dict(**repetition,system=system,data_folder=str(folder.relative_to(out)),
        state_variables=data['meta']['N'],features=data['meta']['P'],
        train_samples=int(train.sum()),validation_samples=int(np.sum(data['splits']=='validation')),
        test_samples=int(np.sum(data['splits']=='test')),dataset_sha256=sha(folder/'metadata.toml'),
        Z_train_sha256=sha(folder/'Z_train.npy'),Z_validation_sha256=sha(folder/'Z_validation.npy'),
        noise_sha256=sha(noise_file),replay_exact=data['meta']['replay_exact'])
    return data,record


def fit_dataset(out,instance,system,repetition,c,mode,workers,pool):
    data,record=prepare_dataset(out,instance,system,repetition,c);folder=data['folder']
    train=data['splits']=='train';val=data['splits']=='validation'
    Z=np.load(folder/'Z_train.npy',mmap_mode='r')
    means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
    unit=np.load(folder/'unit_response_noise.npy');sigma=np.std(data['Y'][:,train],axis=1)
    nodes=range(1) if mode=='probe' else range(data['meta']['N'])
    configs={k:c[k] for k in ['lasso','stlsq','tie_rule']}
    model_records=[]
    for ratio in ([.05] if mode=='probe' else c['noise_ratios']):
        base=folder.parent/pilot.slot(ratio);base.mkdir(exist_ok=True)
        Y=data['Y']+ratio*sigma[:,None]*unit
        for name,matrix in [('y_train',Y[:,train]),('y_validation',Y[:,val])]:
            path=base/(name+'.npy')
            if path.exists():assert np.array_equal(np.load(path),matrix)
            else:np.save(path,matrix)
        init_path=base/'ridge_initial.npy'
        if not init_path.exists():np.save(init_path,initial_ridge(Z[:,1:],Y[:,train].T,1.))
        paths=dict(Z_train=str(folder/'Z_train.npy'),Z_validation=str(folder/'Z_validation.npy'),
                   y_train=str(base/'y_train.npy'),y_validation=str(base/'y_validation.npy'),ridge_initial=str(init_path))
        inputs={name:sha(path) for name,path in paths.items()}
        fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        save_json(base/'shared_inputs.json',dict(files=inputs,input_fingerprint=fingerprint,methods=METHODS))
        pending=[]
        for method in METHODS:
            trial=base/method;trial.mkdir(exist_ok=True)
            for node in nodes:
                diag=trial/f'node_{node+1:03d}.json'
                if diag.exists():
                    assert json.loads(diag.read_text())['input_fingerprint']==fingerprint
                    continue # Explicit resume never selectively retries FAILED.
                pending.append(pool.submit(worker,(node,method,paths,configs,str(trial),fingerprint)))
        for future in as_completed(pending):
            d=future.result()
            print('V2_NODE',repetition['instance_id'],system,ratio,d['method'],d['node'],d['status'],round(d['wall_seconds'],2),flush=True)
        assert all(sha(paths[k])==v for k,v in inputs.items())
        for method in METHODS:
            trial=base/method;diags=[json.loads((trial/f'node_{j+1:03d}.json').read_text()) for j in nodes]
            state='FAILED' if any(x['status']=='FAILED' for x in diags) else 'NA' if any(x['status']=='NA' for x in diags) else 'OK'
            C=np.zeros((len(nodes),data['meta']['P']))
            for j,d in zip(nodes,diags):
                if d['status']=='OK':
                    coef=np.load(trial/f'node_{j+1:03d}_solution.npz')['coefficient_standardized']
                    raw=restore_coefficients(coef[None,:],means,scales)[0];C[j]=raw
                    phi=np.load(folder/'theta.npy',mmap_mode='r')
                    assert np.allclose(phi[train]@raw,Z@coef,rtol=1e-10,atol=1e-10)
            filename='probe_coefficients.npz' if mode=='probe' else 'coefficients.npz' if state=='OK' else 'available_nodes_only.npz'
            sparse.save_npz(trial/filename,sparse.csr_matrix(C))
            status=dict(status=state,method=method,scope='single_equation_resource_probe' if mode=='probe' else 'complete_network',
                        input_fingerprint=fingerprint,nodes=len(diags),OK=sum(x['status']=='OK' for x in diags),
                        fit_seconds_sum=sum(x['wall_seconds'] for x in diags))
            save_json(trial/'status.json',status)
            model_records.append(dict(**repetition,system=system,noise_ratio=ratio,method=method,status=state,
                trial=str(trial.relative_to(out)),dataset_sha256=record['dataset_sha256'],noise_sha256=record['noise_sha256']))
    return record,model_records


def lock_selection(out,models):
    files={}
    for r in models:
        trial=out/r['trial']
        for f in trial.iterdir():
            if f.is_file():files[str(f.relative_to(out))]=sha(f)
    save_json(out/'selection_locked.json',dict(files=files,test_or_truth_evaluated_before_lock=False))


def verify_locks(out):
    locks=json.loads((out/'selection_locked.json').read_text())
    for name,digest in locks['files'].items():assert sha(out/name)==digest,name


def extract_diagnostics(out,models):
    selections=[];failures=[];resources=[]
    for r in models:
        for f in sorted((out/r['trial']).glob('node_*.json')):
            d=json.loads(f.read_text());base={k:r[k] for k in ['instance_id','network_id','network_seed','data_seed','noise_seed','system','noise_ratio','method']}
            base.update(node=d['node'],input_fingerprint=d['input_fingerprint'])
            resources.append(dict(**base,status=d['status'],wall_seconds=d['wall_seconds'],
                cpu_seconds=d['process_cpu_seconds'],worker_lifetime_peak_rss_mib=d['worker_lifetime_peak_rss_mib'],
                blas_threads=str(d['blas_threads'])))
            if d['status']!='OK':
                failures.append(dict(**base,status=d['status'],reason=d.get('reason',''),diagnostic=str(f.relative_to(out))))
                continue
            for i,row in enumerate(d['validation_path']):
                selections.append(dict(**base,candidate=i,parameter=row['parameter'],relative_parameter=row['relative_parameter'],
                    parameter_reference=d['parameter_reference'],validation_MSE=row['MSE'],nonzero=row['nonzero'],
                    selected=i==d['selected_path_index'],status=row['status'],
                    full_library_kkt=row.get('full_library_kkt',''),iterations=row.get('iterations',''),
                    selected_at_grid_boundary=d['parameter_boundary']))
    write_rows(out/'selection_details.csv',selections,fields=list(selections[0]) if selections else ['status'])
    write_rows(out/'failure_reasons.csv',failures,fields=list(failures[0]) if failures else ['status','reason'])
    write_rows(out/'resources.csv',resources)
    return resources


def evaluate(out,models,datasets,c):
    verify_locks(out);trials=[];native_blocks=[];expected_native={};all_rng=set();all_samples=set();kkt=0
    by_dataset={(r['instance_id'],r['system']):r for r in datasets}
    edge_path=out/'edge_scores.csv'
    fields=['system','instance_id','network_id','network_seed','data_seed','noise_ratio','method','status','order','target','source_1','source_2','truth','score']
    with edge_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for iid,system in by_dataset:
            record=by_dataset[(iid,system)];instance=out/iid
            ic=tomllib.loads((instance/'instance_config.toml').read_text())
            data=legacy.load_data(instance,system,record['data_seed']);folder=data['folder'];test=data['splits']=='test'
            rng_ids=set(data['meta']['trajectory_rng_seeds']);assert not (all_rng&rng_ids);all_rng |= rng_ids
            X=np.loadtxt(folder/'states.csv',delimiter=',',ndmin=2)
            hashes={hashlib.sha256(np.ascontiguousarray(X[:,i]).tobytes()).digest() for i in range(X.shape[1])}
            assert len(hashes)==X.shape[1] and not (hashes&all_samples);all_samples |= hashes
            truth={2:{(int(i),int(j)) for i,j,_ in ic[system]['pair_edges']},
                   3:{(int(i),*sorted([int(j),int(k)])) for i,j,k,_ in ic[system]['triple_edges']}}
            phi=np.load(folder/'theta.npy',mmap_mode='r');native=[]
            for r in [m for m in models if (m['instance_id'],m['system'])==(iid,system)]:
                base={k:r[k] for k in ['system','instance_id','network_id','network_seed','data_seed','noise_seed','noise_ratio','method','status','dataset_sha256','noise_sha256']}
                if r['status']=='OK':
                    trial=out/r['trial'];C=sparse.load_npz(trial/'coefficients.npz')
                    assert C.shape==(data['meta']['N'],data['meta']['P'])
                    residual=C@phi[test].T-data['Y'][:,test]
                    nmse=float(np.sum(residual**2)/np.sum(data['Y'][:,test]**2))
                    coords=C.tocoo();np.savetxt(trial/'coefficient_coordinates.csv',np.c_[coords.row+1,coords.col+1,coords.data],delimiter=',',fmt=['%d','%d','%.17g'])
                    native.append(r['trial']);expected_native[r['trial']]={}
                for order,label in [(2,'pairwise'),(3,'triple')]:
                    row=dict(**base,order=label)
                    if r['status']!='OK':
                        row.update({m:None for m in ['F1','AP','MCC','Jaccard','TP','FP','FN','TN','clean_test_NMSE']})
                        row.update(MCC_status='',AP_status='',candidate_count=210 if order==2 else 1365,positive_count=45 if order==2 else 15)
                    else:
                        keys,scores=pilot.physical_scores(C.toarray(),data['d'],data['meta']['physical_node_of_variable'],order)
                        y=np.array([key in truth[order] for key in keys]);metrics=binary_metrics(y,scores,c['decode_threshold'])
                        assert metrics['candidate_count']==(210 if order==2 else 1365)
                        assert metrics['positive_count']==(45 if order==2 else 15)
                        metrics['AP']=metrics['AUPRC_AP'];metrics['AP_status']=metrics['AUPRC_status']
                        metrics['Jaccard']=metrics['TP']/(metrics['TP']+metrics['FP']+metrics['FN'])
                        row.update(**metrics,clean_test_NMSE=nmse)
                        for key,score,t in zip(keys,scores,y):
                            e={k:row[k] for k in fields if k in row};e.update(target=key[0],source_1=key[1],source_2=key[2] if order==3 else 0,truth=int(t),score=float(score));writer.writerow(e)
                            if score>c['decode_threshold']:expected_native[r['trial']][(order,*key)]=float(score)
                    trials.append(row)
            native_blocks.append(dict(data=str(folder.relative_to(out)),N=data['meta']['N'],
                                      group_size=ic[system]['coordinates_per_node'],trials=native))
    fields=list(dict.fromkeys(k for r in trials for k in r));write_rows(out/'trials.csv',trials,fields)
    write_toml(out/'native_decode_request.toml',dict(datasets=native_blocks))
    env=julia_env(out)
    with (out/'logs/native_decode.log').open('w') as log:
        subprocess.run([JULIA,'--startup-file=no','--project='+str(ROOT/'environment/r0/julia'),
                        str(ROOT/'scripts/dynamics30_native_decode.jl'),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    count=0
    for rel,expected in expected_native.items():
        native=load_csv(out/rel/'native_physical_edges.csv')
        actual={(int(r['order']),int(r['target']),*sorted([int(r['source_1'])]+([int(r['source_2'])] if int(r['order'])==3 else []))):float(r['score']) for r in native}
        assert actual==expected,rel;count+=len(actual)
    verify_locks(out)
    return trials,dict(native_decoded_edges_exact=count,unique_sample_rng_names=len(all_rng),
                      unique_sample_vectors=len(all_samples),candidate_universe_verified=True)


def julia_env(out):
    depot=out/'julia_depot';depot.mkdir(exist_ok=True)
    return dict(os.environ,JULIA_DEPOT_PATH=str(depot)+':'+str(ROOT/'.julia-depot-r0'),JULIA_PKG_PRECOMPILE_AUTO='0',JULIA_PKG_OFFLINE='true')


def plot_results(out,trials):
    os.environ['MPLCONFIGDIR']=str(out/'mpl_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    rows=[r for r in trials if r['order']=='triple']
    groups=list(dict.fromkeys((r['system'],r['noise_ratio']) for r in rows))
    for ax,metric in zip(axes,['F1','AP']):
        for j,method in enumerate(METHODS):
            vals=[]
            for s,n in groups:
                v=[r[metric] for r in rows if (r['system'],r['noise_ratio'],r['method'])==(s,n,method) and r['status']=='OK']
                vals.append(float(np.mean(v)) if v else np.nan)
            ax.bar(np.arange(len(groups))+(j-1)*.25,vals,.25,label=method)
        ax.set_xticks(range(len(groups)),[f'{NAMES[s]} {n:.0%}' for s,n in groups],rotation=20)
        ax.set_ylim(0,1.05);ax.set_title('Triple '+metric);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


def run(args):
    c=read_plan(args.plan)
    if args.mode=='full' and not args.confirm_full:raise ValueError('Formal 900-model run requires explicit user confirmation and --confirm-full')
    if args.mode=='full':repetitions=c['repetitions'];systems=SYSTEMS
    else:repetitions=c['repetitions'][:1];systems=['lorenz'] if args.mode=='probe' else ['kuramoto','sis']
    out=ROOT/'results/dynamics15'/args.label
    assert Path(args.label).name==args.label and args.label not in ('.','..')
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
        shutil.copy2(args.plan/'config_locked.toml',out/'config_locked.toml')
        for name in ['seed_manifest.csv','network_manifest.csv','plan_lock.json']:shutil.copy2(args.plan/name,out/name)
        source={str(p):sha(ROOT/p) for p in source_files()}
        for name in source:
            dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        save_json(out/'run_contract.json',dict(mode=args.mode,systems=systems,repetitions=repetitions,
            source_sha256=source,plan_sha256=sha(args.plan/'config_locked.toml'),workers=args.workers,blas_threads=1))
    contract=json.loads((out/'run_contract.json').read_text())
    assert contract['mode']==args.mode and contract['workers']==args.workers
    assert contract['plan_sha256']==sha(args.plan/'config_locked.toml')==sha(out/'config_locked.toml')
    for name,digest in contract['source_sha256'].items():assert sha(ROOT/name)==digest,name
    if (out/'completion.json').exists():raise ValueError('Completed run is immutable; choose a new label')
    started=time.perf_counter();datasets=[];models=[]
    with ResourceMonitor() as monitor:
        for repetition in repetitions:
            instance=out/repetition['instance_id'];instance.mkdir(exist_ok=True)
            ic=instance_config(c,repetition,systems);icpath=instance/'instance_config.toml'
            if icpath.exists():assert tomllib.loads(icpath.read_text())==ic
            else:write_toml(icpath,ic)
            missing=[s for s in systems if not (instance/s/f'seed_{repetition["data_seed"]}'/'data/metadata.toml').exists()]
            if missing:
                generation=deepcopy_config(ic);generation['generation_systems']=missing
                generation_path=instance/'generation_request.toml';write_toml(generation_path,generation)
                with (out/'logs'/f'{repetition["instance_id"]}_prepare.log').open('a') as log:
                    subprocess.run([JULIA,'--startup-file=no','--project='+str(ROOT/'environment/r0/julia'),
                        str(ROOT/'scripts/fair_v2_prepare.jl'),str(generation_path),str(instance)],env=julia_env(out),stdout=log,stderr=subprocess.STDOUT,check=True)
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for system in systems:
                    record,ms=fit_dataset(out,instance,system,repetition,c,args.mode,args.workers,pool)
                    datasets.append(record);models.extend(ms)
                    write_rows(out/'dataset_manifest.csv',datasets);write_rows(out/'model_manifest.csv',models)
        lock_selection(out,models);resources=extract_diagnostics(out,models)
        if args.mode!='probe':
            trials,checks=evaluate(out,models,datasets,c)
            summary=export_statistics(out,trials,c,'PILOT' if args.mode=='pilot' else 'FORMAL')
            plot_results(out,trials)
        else:checks=dict(scope='single Lorenz equation, 5% noise; no network performance reported')
    states={s:sum(r['status']==s for r in models) for s in ('OK','NA','FAILED')}
    completion=dict(status='COMPLETE' if not states['FAILED'] else 'COMPLETE_WITH_FAILURES',mode=args.mode,
        models=len(models),counts=states,wall_seconds=time.perf_counter()-started,resources=monitor.result(),
        full_experiment_run=args.mode=='full',independent_networks=len({r['network_id'] for r in repetitions}),
        independent_data_repetitions=len(repetitions),selection='validation MSE, not structure F1')
    save_json(out/'resources_summary.json',monitor.result())
    save_json(out/'validation.json',dict(status='PASS' if not states['FAILED'] else 'WITH_MODEL_FAILURES',
        shared_train_validation_inputs=True,original_decoder_preserved=True,selection_locked_before_evaluation=True,
        preprocessing_prediction_parity=True,**checks))
    save_json(out/'completion.json',completion)
    report=['# V2 '+args.mode+' 实验报告','',
        '统一验证集MSE选参是为了公平比较三种回归方法，不保证得到结构恢复F1最优的模型。','',
        f'模型记录：{len(models)}；状态：{states}。运行时间：{completion["wall_seconds"]:.2f}秒。',
        f'进程树RSS峰值（共享页重复计数的保守估计）：{monitor.peak/2**20:.1f} MiB；最多进程数：{monitor.max_processes}。',
        '每个worker的高水位RSS和CPU/墙钟耗时见resources.csv。BLAS=1，并发worker数为'+str(args.workers)+'。','',
        '数据为独立t=0初始状态与解析导数，不是真实观测时间序列；全三阶Taylor库，无结构筛选。',
        '三种算法共享训练均值/RMS标准化、自由截距和训练/验证数据。STLSQ仅搜索12档硬阈值，固定Ridge=1。',
        '噪声仅加在解析导数，尺度由训练响应估计；测试NMSE对照干净导数。',
        '网络实现同时变化拓扑与边权；展开中心、固有参数不变。','',
        '原始指标与状态见trials.csv/model_manifest.csv；逐参数选择见selection_details.csv。',
        '配对统计仅匹配相同网络、数据、噪声；置信区间重采样独立网络，网络数少于2不生成区间。',
        'between_network_mean_sd和within_network_rms_sd是描述性波动，不是分离后的纯拓扑方差分量。',
        '只有10个独立网络时区间仍可能不稳定；不得把30组数据或候选边当成30个独立网络。','']
    if args.mode=='probe':report+=['本目录仅测试Lorenz第1个状态方程的三种求解器和资源，无完整网络结构指标。']
    else:report+=['[三体表](tables/table1_threebody.md) · [二体表](tables/table2_pairwise.md)','', '![三体指标](comparison.png)']
    if args.mode!='full':report+=['','FULL EXPERIMENT NOT RUN：本目录不能代替正式900模型结果。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(completion),flush=True)


def deepcopy_config(c):
    import copy
    return copy.deepcopy(c)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run','audit'])
    parser.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    parser.add_argument('--mode',choices=['pilot','probe','full'],default='pilot')
    parser.add_argument('--label',default='fair_v2_pilot_20261023')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--confirm-full',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=12:parser.error('workers must be between 1 and 12')
    if args.command=='plan':make_plan(args.plan);print('PLAN_LOCKED',args.plan)
    elif args.command=='audit':print(json.dumps(audit_old('final_integrity.json')))
    else:run(args)


if __name__=='__main__':main()
