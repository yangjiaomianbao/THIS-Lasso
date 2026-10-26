"""Locked post-fit support, structure, paths, threshold and redundancy analyses."""
import csv
import gzip
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
import subprocess
import time
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from scipy.linalg import svd
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from src.bridge.numeric_io import file_hash,write_toml,stamp
from src.experiments.support_diagnostics import (SYSTEMS,METHODS,NAMES,THRESHOLDS,support_rows,evaluate_structure)
from src.preprocessing.taylor import restore_coefficients
from support_run import table,save,pilot,d30,LEGACY


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def records(path):
    """Restore scalar types when resuming already committed analysis tables."""
    def scalar(v):
        if v=='':return None
        if v=='True':return True
        if v=='False':return False
        try:return int(v)
        except ValueError:
            try:return float(v)
            except ValueError:return v
    return [{k:scalar(v) for k,v in row.items()} for row in read(path)]


class StreamingRows:
    """Native BIC knot tables can be large; do not retain them all in RAM."""
    def __init__(self,path):
        self.file=gzip.open(path,'wt',newline='',compresslevel=1);self.writer=None;self.count=0
    def append(self,row):
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,list(row));self.writer.writeheader()
        self.writer.writerow(row);self.count+=1
    def close(self):
        if self.writer is None:self.file.write('status\n')
        self.file.close()


def load_truth(folder):
    truth=np.zeros((15,816))
    for r in read(folder/'local_taylor_sparse.csv'):
        truth[int(r['node'])-1,int(r['feature'])-1]=float(r['coefficient'])
    return truth


def standardized(C,means,scales):
    result=C*scales;result[:,0]=C[:,0]+C[:,1:]@means[1:]
    return result


def legacy_support(out,plan):
    if (out/'00_existing_support.csv').exists():return
    rows=[]
    for system,seed,ratio,method in itertools.product(SYSTEMS,[2026102201,2026102202],[0.,.05],METHODS):
        folder=LEGACY/system/f'seed_{seed}'/'data';trial=folder.parent/pilot.slot(ratio)/method
        st=json.loads((trial/'status.json').read_text())
        base=dict(system=system,seed=seed,noise_ratio=ratio,method=method,status=st['status'])
        if st['status']=='OK':
            C=sparse.load_npz(trial/'coefficients.npz').toarray()
            means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
            d=np.loadtxt(folder/'monomial_indices.csv',delimiter=',',dtype=int)
            Y=np.loadtxt(folder/'Y.csv',delimiter=',');tr=np.array([r['split']=='train' for r in read(folder/'sample_manifest.csv')])
            support,_=support_rows(C,standardized(C,means,scales),load_truth(folder),d,scales,Y[:,tr].std(axis=1))
            base.update(support,coefficient_sha256=file_hash(trial/'coefficients.npz'))
        rows.append(base)
    table(out/'00_existing_support.csv',rows)


def per_target_curve_metrics(curves,d,target,config,system):
    """Vectorized AP with exact score-tie handling, evaluated only after locking."""
    result=[]
    for order,label in [(2,'pairwise'),(3,'triple')]:
        ids=[];keys=[]
        for j,row in enumerate(d):
            sources=tuple(sorted(int(v) for v in row if v))
            if len(sources)==order-1 and len(set(sources))==len(sources) and target not in sources:
                ids.append(j);keys.append((target,*sources))
        scores=np.abs(curves[:,ids])
        true_set={(int(r[0]),*sorted(int(v) for v in r[1:-1])) for r in config[system]['pair_edges' if order==2 else 'triple_edges']}
        truth=np.array([k in true_set for k in keys]);positive=int(truth.sum())
        pred=scores>1e-8;tp=(pred&truth).sum(axis=1);fp=(pred&~truth).sum(axis=1);fn=positive-tp
        ranks=np.argsort(-scores,axis=1,kind='stable');sorted_scores=np.take_along_axis(scores,ranks,axis=1)
        tps=np.cumsum(truth[ranks],axis=1)
        endpoints=np.column_stack((sorted_scores[:,:-1]!=sorted_scores[:,1:],np.ones(len(scores),bool)))
        previous=np.column_stack((np.zeros(len(scores)),np.maximum.accumulate(np.where(endpoints,tps,0),axis=1)[:,:-1]))
        ap=(np.where(endpoints,(tps-previous)*tps/np.arange(1,len(ids)+1),0).sum(axis=1)/positive
            if positive else np.full(len(scores),np.nan))
        for i in range(len(curves)):
            result.append(dict(candidate=i,order=label,TP=int(tp[i]),FP=int(fp[i]),FN=int(fn[i]),
                               F1=float(2*tp[i]/(2*tp[i]+fp[i]+fn[i])) if 2*tp[i]+fp[i]+fn[i] else 0.,
                               AP=float(ap[i]),precision=float(tp[i]/(tp[i]+fp[i])) if tp[i]+fp[i] else 0.,
                               recall=float(tp[i]/positive) if positive else None))
    return result


def describe(rows,metric,stage,rng,resamples=5000):
    good=[]
    for r in rows:
        v=r.get(metric)
        if r.get('status')!='OK' or v is None:continue
        try:v=float(v)
        except (ValueError,TypeError):continue
        if np.isfinite(v):good.append((int(r['network_id']),v))
    values=np.array([v for _,v in good])
    if not len(values):return dict(n=0,mean=None,std=None,q05=None,q25=None,median=None,q75=None,q95=None,ci_low=None,ci_high=None,networks=0)
    groups=defaultdict(list)
    for net,v in good:groups[net].append(v)
    units=values if stage=='A' else np.array([np.mean(v) for _,v in sorted(groups.items())])
    bootstrap=units[rng.integers(0,len(units),size=(resamples,len(units)))].mean(axis=1) if len(units)>1 else None
    q=np.quantile(values,[.05,.25,.5,.75,.95])
    within=[np.var(v,ddof=1) for v in groups.values() if len(v)>1]
    return dict(n=len(values),mean=float(units.mean()),std=float(values.std(ddof=1)) if len(values)>1 else None,
                q05=float(q[0]),q25=float(q[1]),median=float(q[2]),q75=float(q[3]),q95=float(q[4]),
                ci_low=float(np.quantile(bootstrap,.025)) if bootstrap is not None else None,
                ci_high=float(np.quantile(bootstrap,.975)) if bootstrap is not None else None,
                networks=len(groups),between_network_mean_sd=float(units.std(ddof=1)) if stage=='B' and len(units)>1 else None,
                within_network_rms_sd=float(np.sqrt(np.mean(within))) if within else None,
                inference_unit='data_seed' if stage=='A' else 'network_mean')


def summaries(stage,seed_rows,relations):
    rng=np.random.default_rng(2026102604);support_summary=[];structure_summary=[];pairs=[];pair_summary=[]
    for system,noise,method in itertools.product(SYSTEMS,[0.,.05,.1],METHODS):
        ss=[r for r in seed_rows if (r['system'],r['noise_ratio'],r['method'])==(system,noise,method)]
        base=dict(stage=stage,system=system,noise_ratio=noise,method=method,attempted=len(ss),**Counter(r['status'] for r in ss))
        for metric in ['absolute_nonzero','exact_nonzero','relative_nonzero','normalized_nonzero','support_rate','truth_nonzero','distinct_nonzero_basis']:
            support_summary.append(dict(**base,metric=metric,**describe(ss,metric,stage,rng)))
        for order in ['pairwise','triple']:
            rr=[r for r in relations if (r['system'],r['noise_ratio'],r['method'],r['order'])==(system,noise,method,order)]
            for metric in ['precision','recall','F1','AUPRC_AP','MCC','clean_test_NMSE','FP','FN','predicted_edges']:
                structure_summary.append(dict(**base,order=order,metric=metric,**describe(rr,metric,stage,rng)))
    index={(r['system'],r['noise_ratio'],r['order'],r['instance'],r['method']):r for r in relations}
    for system,noise,order,comparison in itertools.product(SYSTEMS,[0.,.05,.1],['pairwise','triple'],[('M2B','M1B'),('M3C','M1vC')]):
        instances=sorted({r['instance'] for r in relations if r['system']==system})
        for metric in ['F1','AUPRC_AP','precision','recall','MCC','clean_test_NMSE','absolute_nonzero']:
            group=[]
            for instance in instances:
                a=index[(system,noise,order,instance,comparison[0])];b=index[(system,noise,order,instance,comparison[1])]
                valid=a['status']==b['status']=='OK' and all(np.isfinite(float(r.get(metric,np.nan))) for r in [a,b])
                p=dict(stage=stage,system=system,noise_ratio=noise,order=order,comparison=' - '.join(comparison),
                       metric=metric,instance=instance,network_id=a['network_id'],seed=a['seed'],
                       status='OK' if valid else 'NA',difference=float(a[metric])-float(b[metric]) if valid else None,
                       reason='' if valid else f'{a["status"]}/{b["status"]} or undefined metric')
                group.append(p);pairs.append(p)
            desc=describe(group,'difference',stage,rng);valid=[r['difference'] for r in group if r['status']=='OK']
            pair_summary.append(dict(stage=stage,system=system,noise_ratio=noise,order=order,comparison=' - '.join(comparison),
                                     metric=metric,attempted=len(group),missing=len(group)-len(valid),
                                     positive_fraction=float(np.mean(np.array(valid)>0)) if valid else None,
                                     negative_fraction=float(np.mean(np.array(valid)<0)) if valid else None,**desc))
    return support_summary,structure_summary,pairs,pair_summary


def redundancy(folder,trial,C,grid_records,base,cfg,data):
    """Bounded LOCAL projection/refit diagnostics; no threshold or model changes."""
    if base['sample_id']!=1 or base['method'] not in ('M1vC','M3C'):return []
    triple=evaluate_structure(C,data['d'],cfg,base['system'])[1]
    if triple['FP']<15 or max(r['F1'] for r in grid_records if r['order']=='triple')>=.9:return []
    Z=np.load(folder/'Z_train.npy');Zv=np.load(folder/'Z_validation.npy')
    Y=np.loadtxt(trial.parent/'Y_noisy.csv',delimiter=',');tr=data['splits']=='train';val=data['splits']=='validation'
    init=np.load(trial.parent/'Ridge_initial_standardized.npy')
    means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
    Cs=standardized(C,means,scales);d=data['d'];chosen=[]
    # Rank predicted groups before looking at true labels.
    for target in range(15):
        for sources in itertools.combinations([j for j in range(1,16) if j!=target+1],2):
            group=np.array([j for j,row in enumerate(d) if set(int(v) for v in row if v)-{target+1}==set(sources)],int)
            score=float(np.max(np.abs(C[target,group]),initial=0.))
            if score>1e-8:chosen.append((score,target,sources,group))
    chosen=sorted(chosen,key=lambda r:(-r[0],r[1],r[2]))[:10];rows=[]
    for score,target,sources,group in chosen:
        diag=json.loads((trial/f'node_{target+1:03d}.json').read_text());alpha=float(diag['alpha'])
        own=[j for j,row in enumerate(d) if j and set(int(v) for v in row if v)<={target+1}]
        ranked=np.argsort(-np.abs(Cs[target,1:]),kind='stable')+1
        competitors=list(dict.fromkeys(own+[int(j) for j in ranked if j not in group]))[:128]
        G=Z[:,group];V=Z[:,competitors]
        U,s,_=svd(V,full_matrices=False,check_finite=False);tol=np.finfo(float).eps*max(V.shape)*(s[0] if len(s) else 0)
        rank=int(np.sum(s>tol));Q=U[:,:rank];residual=G-Q@(Q.T@G)
        sr=np.linalg.svd(residual,compute_uv=False);sg=np.linalg.svd(G,compute_uv=False)
        a=np.ones(815) if base['method']=='M1vC' else (np.abs(init[target])+cfg['adaptive']['eps'])**cfg['adaptive']['gamma']
        train_y=Y[target,tr];yv=Y[target,val];ym=float(train_y.mean())
        def refit(ids):
            weights=a[np.array(ids)-1]
            model=Lasso(alpha=alpha,fit_intercept=False,max_iter=100000,tol=1e-10).fit(np.asfortranarray(Z[:,ids]*weights),train_y-ym)
            beta=model.coef_*weights
            gradient=Z[:,ids].T@(Z[:,ids]@beta-(train_y-ym))/len(train_y)
            penalty=alpha/weights;nz=beta!=0;v=np.maximum(np.abs(gradient)-penalty,0.)
            v[nz]=np.abs(gradient[nz]+penalty[nz]*np.sign(beta[nz]))
            err=(ym+Zv[:,ids]@beta-yv)**2
            return float(err.mean()),float(v.max(initial=0.))
        local=list(dict.fromkeys(competitors+group.tolist()));full,k1=refit(local);removed,k2=refit(competitors)
        truth={(int(r[0]),*sorted(int(v) for v in r[1:-1])) for r in cfg[base['system']]['triple_edges']}
        normalized=V/np.maximum(np.linalg.norm(V,axis=0),np.finfo(float).tiny)
        cross=np.abs(normalized.T@(G/np.maximum(np.linalg.norm(G,axis=0),np.finfo(float).tiny)))
        rows.append(dict(**base,target=target+1,source_1=sources[0],source_2=sources[1],predicted_score=score,
                         true_group=int((target+1,*sources) in truth),group_columns=len(group),competitor_columns=len(competitors),
                         competitor_rank=rank,group_rank=int(np.linalg.matrix_rank(G)),residual_rank=int(np.linalg.matrix_rank(residual)),
                         residual_norm_ratio=float(np.linalg.norm(residual)/max(np.linalg.norm(G),np.finfo(float).tiny)),
                         max_group_competitor_correlation=float(cross.max(initial=0.)),
                         group_singular_values=json.dumps(sg.tolist()),residual_singular_values=json.dumps(sr.tolist()),
                         local_validation_MSE=full,local_deleted_validation_MSE=removed,local_deletion_delta_MSE=removed-full,
                         full_local_KKT_abs=k1,deleted_local_KKT_abs=k2,
                         scope='local competing set; fixed original alpha/weights; not full-library causal evidence'))
    return rows


def analyse_stage(out,plan,stage):
    out=Path(out);dest=out/('A_fixed' if stage=='A' else 'B_networks');dest.mkdir(exist_ok=True)
    checkpoint=out/f'analysis_tables_{stage}.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        for name,digest in saved['files'].items():assert file_hash(out/name)==digest,name
        return finalise_stage(out,plan,stage)
    lock=json.loads((out/f'selection_locked_{stage}.json').read_text())['models']
    # All fitting must be locked before either truth or clean test scoring.
    assert len(lock)==(540 if stage=='A' else 2700)
    save(out/'selection_locked.json',dict(stage=stage,models=lock,locked=stamp()))
    support=[];node_support=[];relations=[];paths=[];selections=[];decoders=[];extras=[];red=[];native_requests=[]
    native_nodes=StreamingRows(dest/'04_alpha_path_nodes.csv.gz')
    for inst in plan['instances']:
        if inst['stage']!=stage:continue
        cfg=tomllib.loads((out/inst['config']).read_text());instance_root=out/inst['folder'];seed=inst['data_seed']
        for system in SYSTEMS:
            data=d30.load_data(instance_root,system,seed);folder=data['folder'];truth=load_truth(folder);d=data['d']
            means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
            phi=np.load(folder/'theta.npy',mmap_mode='r');tr=data['splits']=='train';val=data['splits']=='validation';test=data['splits']=='test'
            raw_train=np.asarray(phi[tr]);raw_val=np.asarray(phi[val]);raw_test=np.asarray(phi[test]);sigma=data['Y'][:,tr].std(axis=1)
            native_trials=[]
            for ratio,method in itertools.product(plan['noise_ratios'],METHODS):
                trial=folder.parent/pilot.slot(ratio)/method;st=json.loads((trial/'status.json').read_text())
                key=str(trial.relative_to(out));assert file_hash(trial/'status.json')==lock[key]['status_sha256']
                base=dict(stage=stage,network_id=inst['network_id'],sample_id=inst['sample_id'],instance=inst['folder'],
                          system=system,seed=seed,noise_ratio=ratio,method=method,status=st['status'])
                if st['status']!='OK':
                    support.append(dict(**base,reason=st.get('reason',json.dumps(st.get('node_counts',{})))))
                    relations.extend(dict(**base,order=o) for o in ['pairwise','triple']);continue
                assert file_hash(trial/'coefficients.npz')==lock[key]['coefficients_sha256']
                C=sparse.load_npz(trial/'coefficients.npz').toarray();Cs=standardized(C,means,scales)
                counts,eq=support_rows(C,Cs,truth,d,scales,sigma);support.append(dict(**base,**counts))
                node_support.extend(dict(**base,**r) for r in eq)
                mask=np.abs(C)>1e-8;mask[:,0]=False
                np.savez_compressed(trial/'support_indices.npz',node=np.nonzero(mask)[0]+1,feature=np.nonzero(mask)[1]+1)
                nmse=float(np.sum((C@raw_test.T-data['Y'][:,test])**2)/np.sum(data['Y'][:,test]**2))
                metrics=evaluate_structure(C,d,cfg,system)
                for r in metrics:relations.append(dict(**base,**counts,clean_test_NMSE=nmse,**r))
                coords=np.argwhere(C!=0)
                table(trial/'coefficient_coordinates.csv',({'node':int(i+1),'feature':int(j+1),'coefficient':float(C[i,j])} for i,j in coords))
                native_trials.append(str(trial.relative_to(out)))
                for threshold in THRESHOLDS:
                    for r in evaluate_structure(C,d,cfg,system,threshold):decoders.append(dict(**base,**r))
                if method not in ('M1B','M2B','M1vC','M3C'):continue
                Y=np.loadtxt(trial.parent/'Y_noisy.csv',delimiter=',');grid=np.load(trial/'aligned_path_standardized.npy')
                raw=restore_coefficients(grid.reshape(-1,816),means,scales).reshape(grid.shape)
                train_errors=((raw.reshape(-1,816)@raw_train.T).reshape(12,15,-1)-Y[None,:,tr])**2
                val_errors=((raw.reshape(-1,816)@raw_val.T).reshape(12,15,-1)-Y[None,:,val])**2
                test_errors=((raw.reshape(-1,816)@raw_test.T).reshape(12,15,-1)-data['Y'][None,:,test])**2
                model_grid=[]
                for candidate,fraction in enumerate(cfg['lasso']['alpha_fractions']):
                    info=dict(candidate=candidate,alpha_fraction=fraction,alignment='common fraction of each equation alpha_max',
                              absolute_nonzero=int(np.count_nonzero(np.abs(raw[candidate,:,1:])>1e-8)),
                              training_MSE=float(train_errors[candidate].mean()),validation_MSE=float(val_errors[candidate].mean()),
                              clean_test_NMSE=float(test_errors[candidate].sum()/np.sum(data['Y'][:,test]**2)))
                    for r in evaluate_structure(raw[candidate],d,cfg,system):
                        p=dict(**base,**info,**r);paths.append(p);model_grid.append(p)
                for node in range(15):
                    diag=json.loads((trial/f'node_{node+1:03d}.json').read_text())
                    # NPZ indexing decompresses a member on every access. Load
                    # each member once before traversing hundreds of BIC knots.
                    with np.load(trial/f'node_{node+1:03d}_path.npz') as compressed:
                        bundle={name:compressed[name] for name in compressed.files}
                    selected=int(diag['selected_path_index']);curve=restore_coefficients(bundle['candidates'],means,scales)
                    target_metrics=per_target_curve_metrics(curve,d,node+1,cfg,system)
                    alphas=bundle['alphas'];criteria=diag.get('bic_path',[None]*len(alphas))
                    for r in target_metrics:
                        i=r['candidate'];cert=float(bundle['full_library_KKT'][i])
                        native_nodes.append(dict(**base,node=node+1,alpha=float(alphas[i]),selected=i==selected,
                                                 alpha_fraction=float(alphas[i]/max(alphas[0],np.finfo(float).eps)),
                                                 absolute_nonzero=int(np.count_nonzero(np.abs(curve[i,1:])>1e-8)),
                                                 standardized_nonzero=int(np.count_nonzero(bundle['candidates'][i,1:])),
                                                 training_MSE=float(bundle['train_MSE'][i]),validation_MSE=float(bundle['validation_MSE'][i]),
                                                 BIC=criteria[i],full_library_KKT=cert,candidate_status='OK' if cert<=1e-6 else 'KKT_NOT_CERTIFIED',**r))
                    best=int(np.argmin(bundle['validation_MSE']))
                    selections.append(dict(**base,node=node+1,selection='BIC' if method in ('M1B','M2B') else 'validation',
                                           selected_alpha=float(diag['alpha']),selected_candidate=selected,candidates=len(alphas),
                                           selected_at_path_boundary=selected in (0,len(alphas)-1),
                                           validation_MSE_selected=float(bundle['validation_MSE'][selected]),
                                           holdout_1se_candidate=diag.get('holdout_1se_index'),
                                           holdout_best_MSE_SE=diag.get('holdout_best_MSE_SE'),
                                           selected_nonzero=int(np.count_nonzero(np.abs(curve[selected,1:])>1e-8))))
                if method in ('M1vC','M3C'):
                    conservative=sparse.load_npz(trial/'holdout_1se_coefficients.npz').toarray()
                    for r in evaluate_structure(conservative,d,cfg,system):
                        extras.append(dict(**dict(base,method=method+'_holdout1se'),
                                           absolute_nonzero=int(np.count_nonzero(np.abs(conservative[:,1:])>1e-8)),
                                           clean_test_NMSE=float(np.sum((conservative@raw_test.T-data['Y'][:,test])**2)/np.sum(data['Y'][:,test]**2)),**r))
                if stage=='A':red.extend(redundancy(folder,trial,C,model_grid,base,cfg,data))
            native_requests.append(dict(data=str(folder.relative_to(out)),N=15,group_size=1,trials=native_trials))
            print('ANALYSED',stage,inst['folder'],system,flush=True)
    native_nodes.close()
    ss,struct,pairs,paired=summaries(stage,support,relations)
    for name,rows in [('01_coefficient_support_seed.csv',support),('01_coefficient_support_nodes.csv',node_support),
                      ('02_coefficient_support_summary.csv',ss),('03_support_structure_relation.csv',relations),
                      ('structure_summary.csv',struct),('04_alpha_path.csv',paths),
                      ('05_parameter_selection_diagnostics.csv',selections),('06_decoder_threshold_diagnostics.csv',decoders),
                      ('07_redundancy_diagnostics.csv',red),('paired_differences.csv',pairs),('paired_summary.csv',paired),
                      ('holdout_1se_structure.csv',extras)]:table(dest/name,rows)
    write_toml(dest/'native_decode_request.toml',dict(datasets=native_requests))
    outputs=list(dest.glob('*.csv'))+[dest/'04_alpha_path_nodes.csv.gz',dest/'native_decode_request.toml']
    save(checkpoint,dict(status='TABLES_COMMITTED',stage=stage,created=stamp(),
                         files={str(f.relative_to(out)):file_hash(f) for f in outputs}))
    return finalise_stage(out,plan,stage)


def finalise_stage(out,plan,stage):
    """Resume native decoding, figures and report without replaying model paths."""
    dest=out/('A_fixed' if stage=='A' else 'B_networks')
    locks=json.loads((out/f'selection_locked_{stage}.json').read_text())['models']
    for name,expected in locks.items():
        assert file_hash(out/name/'status.json')==expected['status_sha256'],name
        if expected['coefficients_sha256']:
            assert file_hash(out/name/'coefficients.npz')==expected['coefficients_sha256'],name
    support=records(dest/'01_coefficient_support_seed.csv')
    relations=records(dest/'03_support_structure_relation.csv')
    paths=records(dest/'04_alpha_path.csv');pairs=records(dest/'paired_differences.csv')
    paired=records(dest/'paired_summary.csv');extras=records(dest/'holdout_1se_structure.csv')
    red=records(dest/'07_redundancy_diagnostics.csv')
    request=dest/'native_decode_request.toml'
    native_requests=tomllib.loads(request.read_text())['datasets']
    (out/'native_decode_request.toml').write_bytes(request.read_bytes())
    runtime=['/home/yg/.local/opt/julia-1.9.4/bin/julia','--startup-file=no','--project='+str(ROOT/'environment/r0/julia')]
    env=dict(os.environ,JULIA_NUM_THREADS='1',
             JULIA_DEPOT_PATH=str(out/'julia_depot')+':'+str(ROOT/'.julia-depot-r0'))
    with (out/f'logs/native_decode_{stage}.log').open('w') as log:
        subprocess.run(runtime+[str(ROOT/'scripts/dynamics30_native_decode.jl'),str(out)],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    checked=0
    from src.metrics.structure import edge_scores
    for block in native_requests:
        d=np.loadtxt(out/block['data']/'monomial_indices.csv',delimiter=',',dtype=int)
        for name in block['trials']:
            trial=out/name;C=sparse.load_npz(trial/'coefficients.npz').toarray()
            native={(int(r['order']),int(r['target']),int(r['source_1']),int(r['source_2'])):float(r['score']) for r in read(trial/'native_physical_edges.csv')}
            for order in [2,3]:
                keys,scores=edge_scores(C,d,order)
                for key,v in zip(keys,scores):
                    k=(order,key[0],key[1],key[2] if order==3 else 0)
                    assert native.get(k,0.)==float(v),(name,k,v,native.get(k))
                    checked+=1
    figures(dest,support,relations,paths,pairs,red)
    report(dest,stage,support,relations,paths,paired,extras,red)
    save(out/f'analysed_{stage}.json',dict(status='PASS',models=len(support),counts=dict(Counter(r['status'] for r in support)),
                                         native_candidate_checks=checked,red_groups=len(red),finished=stamp()))


def figures(dest,support,relations,paths,pairs,red):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    noises=[0.,.05,.1];colors=plt.get_cmap('tab10')
    def panels():return plt.subplots(3,3,figsize=(16,11),squeeze=False,constrained_layout=True)
    def finish(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=150);fig.savefig(dest/(name+'.pdf'));plt.close(fig)
    fig,axes=panels()
    for i,s in enumerate(SYSTEMS):
        for j,n in enumerate(noises):
            ax=axes[i,j]
            for k,m in enumerate(METHODS):
                vals=[r['absolute_nonzero'] for r in support if (r['system'],r['noise_ratio'],r['method'],r['status'])==(s,n,m,'OK')]
                if vals:ax.boxplot([vals],positions=[k],widths=.6,showfliers=True)
            ax.set(title=f'{s} / noise={n:.0%}',xticks=range(6),xticklabels=METHODS,ylabel='Nonzero coefficients (bias excluded)')
    finish(fig,'support_distributions')
    fig,axes=panels()
    for i,s in enumerate(SYSTEMS):
        for j,n in enumerate(noises):
            ax=axes[i,j]
            for k,comparison in enumerate(['M2B - M1B','M3C - M1vC']):
                vals=[r['difference'] for r in pairs if (r['system'],r['noise_ratio'],r['order'],r['comparison'],r['metric'],r['status'])==(s,n,'triple',comparison,'absolute_nonzero','OK')]
                if vals:ax.boxplot([vals],positions=[k],widths=.6)
            ax.axhline(0,color='gray');ax.set(title=f'{s} / {n:.0%}',xticks=[0,1],xticklabels=['BIC','Validation'],ylabel='Adaptive - ordinary support')
    finish(fig,'paired_support_difference')
    for kind in ['support_vs_FP','AP_vs_F1']:
        fig,axes=panels()
        for i,s in enumerate(SYSTEMS):
            for j,n in enumerate(noises):
                ax=axes[i,j]
                for k,m in enumerate(METHODS):
                    rs=[r for r in relations if (r['system'],r['noise_ratio'],r['method'],r['order'],r['status'])==(s,n,m,'triple','OK')]
                    x=[r['absolute_nonzero' if kind=='support_vs_FP' else 'AUPRC_AP'] for r in rs]
                    y=[r['FP' if kind=='support_vs_FP' else 'F1'] for r in rs]
                    ax.scatter(x,y,s=14,alpha=.6,color=colors(k),label=NAMES[m])
                ax.set(title=f'{s} / {n:.0%}',xlabel='Coefficient support' if kind=='support_vs_FP' else 'AP',ylabel='Triple FP' if kind=='support_vs_FP' else 'Triple F1')
        axes[0,0].legend(fontsize=7);finish(fig,kind)
    for key,name,ylabel in [('absolute_nonzero','alpha_support','Nonzero coefficient count'),('validation_MSE','alpha_validation_error','Validation MSE'),('predicted_edges','alpha_predicted_triples','Predicted triple edges')]:
        fig,axes=panels()
        for i,s in enumerate(SYSTEMS):
            for j,n in enumerate(noises):
                ax=axes[i,j]
                for k,m in enumerate(['M1vC','M3C']):
                    rs=[r for r in paths if (r['system'],r['noise_ratio'],r['method'],r['order'])==(s,n,m,'triple')]
                    xs=sorted({r['alpha_fraction'] for r in rs})
                    ys=[np.mean([r[key] for r in rs if r['alpha_fraction']==x]) for x in xs]
                    ax.plot(xs,ys,marker='.',label=NAMES[m])
                ax.set(xscale='log',title=f'{s} / {n:.0%}',xlabel='Per-equation alpha / alpha_max',ylabel=ylabel)
        axes[0,0].legend();finish(fig,name)
    if red:
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for label in [0,1]:
            vals=[r['residual_norm_ratio'] for r in red if r['true_group']==label]
            if vals:ax.hist(vals,bins=15,alpha=.5,label='True' if label else 'False')
        ax.set(xlabel='Conditional residual norm / group norm',ylabel='Groups');ax.legend();finish(fig,'redundancy_residuals')


def report(dest,stage,support,relations,paths,paired,extras,red):
    states=Counter(r['status'] for r in support)
    lines=[f'# 三类动力学支持集与结构误检诊断：阶段 {stage}','',
           f'更新时间：{stamp()}。模型状态：{dict(states)}。',
           'A 为固定原网络、10 个数据种子；B 为 5 个额外网络、每网络10个数据种子。三个噪声为0/5/10%，六方法保留原含义。',
           '数据为独立 t=0 初值和解析导数，没有积分、同步预热或真实轨迹。',
           '原默认 STLSQ 的阈值固定；与验证选参 LASSO 的对比是具体流程比较，不能据此证明调参后 STLSQ 普遍较差。','',
           '## Q1：函数支持集是否膨胀？','',
           '| 系统 | 噪声 | 方法 | OK | 非零系数均值 | 解析参考均值 | 支持率 | 三体FP均值 | 三体F1/AP |',
           '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for s,n,m in itertools.product(SYSTEMS,[0.,.05,.1],METHODS):
        rr=[r for r in relations if (r['system'],r['noise_ratio'],r['method'],r['order'],r['status'])==(s,n,m,'triple','OK')]
        if not rr:
            lines.append(f'| {s} | {n:.0%} | {NAMES[m]} | 0 | NA | NA | NA | NA | NA |');continue
        avg=lambda key:float(np.mean([r[key] for r in rr]))
        lines.append(f'| {s} | {n:.0%} | {NAMES[m]} | {len(rr)} | {avg("absolute_nonzero"):.1f} | {avg("truth_nonzero"):.1f} | {avg("support_rate"):.4f} | {avg("FP"):.1f} | {avg("F1"):.4f}/{avg("AUPRC_AP"):.4f} |')
    lines+=['','主要支持数排除常数/截距；另存精确非零、原尺度1e-8、逐方程相对1e-6、标准化与响应归一化统计。解析参考只作事后诊断；Kuramoto有三阶截断误差，额外项不能自动认定为虚假物理作用。','',
            '## Q2：支持数与误检是否对应？','',
            '| 系统 | 噪声 | 支持数—三体FP Spearman（合并流程，描述性） |','|---|---:|---:|']
    for s,n in itertools.product(SYSTEMS,[0.,.05,.1]):
        rr=[r for r in relations if (r['system'],r['noise_ratio'],r['order'],r['status'])==(s,n,'triple','OK')]
        rho=spearmanr([r['absolute_nonzero'] for r in rr],[r['FP'] for r in rr]).statistic if len(rr)>1 and len({r['absolute_nonzero'] for r in rr})>1 and len({r['FP'] for r in rr})>1 else np.nan
        lines.append(f'| {s} | {n:.0%} | {rho:.4f} |')
    # Four descriptive examples; criteria fixed in output, not used to fit models.
    examples=[]
    for r in relations:
        if r['status']!='OK' or r['order']!='triple':continue
        high=r['absolute_nonzero']>2*max(r['truth_nonzero'],1)
        category='A_many_functions_many_FP' if high and r['FP']>=15 else 'B_many_functions_accurate_structure' if high and r['F1']>=.9 else 'C_fewer_functions_many_FP' if not high and r['FP']>=15 else 'D_few_functions_many_FN' if not high and r['recall']<=.5 else ''
        if category:examples.append(dict(category=category,**r))
    table(dest/'case_examples.csv',examples)
    lines+=['','相关不代表因果；合并流程的相关也受选参机制影响。A–D案例采用“支持数>解析参考2倍”“FP>=15”“F1>=.9”“Recall<=.5”的预定义描述标准，见case_examples.csv。没有找到的类型应如实保留为空。','',
            '## Q3：正则化或结构阈值能否改善？','',
            '| 系统 | 噪声 | 方法 | 原F1 | 留出1-SE式F1 | 原FP | 1-SE式FP | 原Recall | 1-SE式Recall |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for s,n,m in itertools.product(SYSTEMS,[0.,.05,.1],['M1vC','M3C']):
        rr=[r for r in relations if (r['system'],r['noise_ratio'],r['method'],r['order'],r['status'])==(s,n,m,'triple','OK')]
        ee=[r for r in extras if (r['system'],r['noise_ratio'],r['method'],r['order'])==(s,n,m+'_holdout1se','triple')]
        if rr and ee:
            av=lambda a,k:np.mean([r[k] for r in a])
            lines.append(f'| {s} | {n:.0%} | {NAMES[m]} | {av(rr,"F1"):.4f} | {av(ee,"F1"):.4f} | {av(rr,"FP"):.1f} | {av(ee,"FP"):.1f} | {av(rr,"recall"):.4f} | {av(ee,"recall"):.4f} |')
    lines+=['','1-SE式方法只使用独立验证样本的平方误差标准误，不是交叉验证。alpha路径以共同相对网格组合逐方程系数；BIC原始折点保存在节点路径中，模型级12点插值仅用于诊断，不取代原BIC选择。',
            '04_alpha_path.csv固定解码阈值1e-8；06_decoder_threshold_diagnostics.csv固定最终系数。后者的未截断连续AP不随判定阈值变化。真值最优路径/阈值属于事后oracle，不能作为实际部署选择或用测试分数倒推规则。','',
            '## Q4：是否支持研究投影二次筛选？','']
    if red:
        true=[r['residual_norm_ratio'] for r in red if r['true_group']];false=[r['residual_norm_ratio'] for r in red if not r['true_group']]
        lines.append(f'按预声明触发规则诊断 {len(red)} 组。真组 {len(true)}、假组 {len(false)}；投影残差比均值分别为 {np.mean(true) if true else float("nan"):.4g}、{np.mean(false) if false else float("nan"):.4g}。')
        if true and false and max(min(true),min(false))<=min(max(true),max(false)):
            lines.append('本次真/假组的残差比分布范围重叠，简单按残差删除不可靠；存在冗余不等于已经证明投影筛选有效。')
        else:lines.append('样本和竞争空间有限，尚不足以验证新的筛选算法；需要独立预声明数据检验可区分性。')
        lines.append('组按预测分数选择，真值仅事后标注。使用最多128竞争列的QR/SVD等价残差和固定alpha/权重的局部组删除重拟合；没有进行全库重选参或改变原方法输出。')
    elif stage=='A':lines.append('没有模型满足预声明触发规则，本阶段未执行投影组诊断；当前应先依据Q3的选参/阈值结果，不增加新投影模块。')
    else:lines.append('B用于跨网络验证Q1–Q3；有限Q4机制诊断单列在A中，未扩展为全网络全组投影。')
    lines+=['','## 统计与局限','',
            '固定网络阶段以数据种子配对bootstrap；多网络阶段以网络均值为重采样单位，不把每网络10个样本当作独立网络。5000次重采样；区间逐点、未作多重比较校正。NA/FAILED不填零。',
            'BIC无噪声GLV/SIS的原残差方差条件不满足，保留NA。只有10个固定网络数据重复、5个额外网络，区间可能不稳定。',
            '候选支持的物理源分组与原生THIS读出分别统计：次数不等于交互阶次，部分自身/混合/三次项参与回归但不进入原二/三体读出。',
            '节点级alpha、原生BIC路径、完整系数与逐样本验证损失均保存，支持复查；本轮没有真实拓扑预筛选、Pearson筛选或新增投影筛选算法。','',
            '主要文件：01/02 支持统计，03结构关系，04模型/节点路径，05选参，06阈值，07冗余，paired_summary.csv配对差与区间。图表同时保存PNG/PDF。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')


def finish(out,plan,wall_seconds):
    stages={s:json.loads((out/f'analysed_{s}.json').read_text()) for s in ['A','B']}
    counts=Counter()
    for v in stages.values():counts.update(v['counts'])
    names=['01_coefficient_support_seed.csv','02_coefficient_support_summary.csv','03_support_structure_relation.csv',
           '04_alpha_path.csv','05_parameter_selection_diagnostics.csv','06_decoder_threshold_diagnostics.csv',
           '07_redundancy_diagnostics.csv','paired_summary.csv']
    for name in names:
        table(out/name,itertools.chain(read(out/'A_fixed'/name),read(out/'B_networks'/name)))
    # Check original sources and all frozen old data/results after the full run.
    for name,digest in plan['source_sha256'].items():assert file_hash(ROOT/name)==digest,name
    for name,digest in plan['legacy_hashes'].items():assert file_hash(LEGACY/name)==digest,name
    result=dict(status='COMPLETE_WITH_FAILURES' if counts['FAILED'] else 'COMPLETE',models=sum(counts.values()),
                counts=dict(counts),stages=stages,wall_seconds=wall_seconds,finished=stamp(),
                old_files_verified=len(plan['legacy_hashes']),source_hashes_verified=len(plan['source_sha256']),
                systems=SYSTEMS,noise_ratios=plan['noise_ratios'],seeds_per_network=10,
                original_fixed_networks=1,additional_networks=5,independent_decode='PASS')
    save(out/'completion.json',result);save(out/'progress.json',result)
    (out/'REPORT.md').write_text('# 三类动力学、10种子支持集与误检诊断 V2\n\n'
        f'完成时间：{stamp()}。总计{result["models"]}记录，状态{dict(counts)}。\n\n'
        '本实验由用户将五类动力学缩减为Kuramoto、Lotka–Volterra、SIS，固定网络数据重复减为10。噪声0/5/10%，原六方法。\n\n'
        '- [A：固定原网络、10个数据种子](A_fixed/REPORT.md)\n'
        '- [B：5个额外网络、每网络10个数据种子](B_networks/REPORT.md)\n\n'
        'A/B的支持集、结构指标、路径、阈值及配对表分别汇总，根目录合并表保留stage字段，不把A/B混为同一统计样本。\n'
        '独立Julia/Python结构读出核对通过；原记录哈希和求解器源码核对通过。\n'
        '先查看A/B报告中的Q1–Q4结果及限制，再结合paired_summary.csv判断方法差异。\n')
    manifest=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and 'julia_depot' not in p.parts and p.name!='artifact_manifest.csv':
            manifest.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=file_hash(p)))
    table(out/'artifact_manifest.csv',manifest)
    print('COMPLETE',json.dumps(result),flush=True)
