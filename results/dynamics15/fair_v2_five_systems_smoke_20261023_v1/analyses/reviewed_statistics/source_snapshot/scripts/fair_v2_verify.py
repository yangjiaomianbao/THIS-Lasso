"""Independent read-only verification of V2 scores, paired statistics and locked choices."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.experiments.fair_v2 import sha,load_csv,save_json,METHODS


def ap(truth,scores):
    ix=np.argsort(-scores,kind='stable');s=scores[ix];y=truth[ix]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0.,tp/y.sum()])*tp/(ends+1)))


def verify(out):
    out=Path(out);contract=json.loads((out/'run_contract.json').read_text())
    assert sha(out/'config_locked.toml')==contract['plan_sha256']
    for name,digest in contract['source_sha256'].items():
        assert sha(out/'source_snapshot'/name)==digest,name
    for name,digest in json.loads((out/'selection_locked.json').read_text())['files'].items():
        assert sha(out/name)==digest,name
    models=load_csv(out/'model_manifest.csv');trials=load_csv(out/'trials.csv')
    assert len(trials)==2*len(models)
    model_keys={(r['instance_id'],r['system'],float(r['noise_ratio']),r['method']) for r in models}
    assert len(model_keys)==len(models)
    for iid,s,ratio in {k[:3] for k in model_keys}:
        assert {k[3] for k in model_keys if k[:3]==(iid,s,ratio)}==set(METHODS)
    def key(r):return r['instance_id'],r['system'],float(r['noise_ratio']),r['method'],r['order']
    edges=defaultdict(list)
    for r in load_csv(out/'edge_scores.csv'):edges[key(r)].append(r)
    checked=0;undefined=0
    for r in trials:
        if r['status']!='OK':
            assert key(r) not in edges;continue
        es=edges[key(r)];y=np.array([int(x['truth']) for x in es]);s=np.array([float(x['score']) for x in es])
        pair=r['order']=='pairwise';assert len(es)==(210 if pair else 1365) and y.sum()==(45 if pair else 15)
        ids={(int(x['target']),int(x['source_1']),int(x['source_2'])) for x in es}
        assert len(ids)==len(es)
        assert all(i!=j and (k==0 if pair else k!=i and k!=j and j<k) for i,j,k in ids)
        pred=s>1e-8;tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)))
        fn=int(np.sum(~pred&(y==1)));tn=int(np.sum(~pred&(y==0)))
        assert [tp,fp,fn,tn]==[int(r[x]) for x in ['TP','FP','FN','TN']]
        for field,x in [('F1',2*tp/(2*tp+fp+fn)),('Jaccard',tp/(tp+fp+fn)),('AP',ap(y,s))]:
            assert math.isclose(float(r[field]),x,rel_tol=1e-12,abs_tol=1e-12),(key(r),field)
        den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        if den:assert r['MCC_status']=='OK' and math.isclose(float(r['MCC']),(tp*tn-fp*fn)/den,abs_tol=1e-12)
        else:assert r['MCC_status']=='NA';undefined+=1
        checked+=1
    for r in load_csv(out/'summary.csv'):
        rs=[x for x in trials if all(str(x[k])==str(r[k]) for k in ['system','noise_ratio','method','order'])]
        assert len(rs)==int(r['attempted'])
        for m in ['F1','AP','MCC','Jaccard']:
            vs=[float(x[m]) for x in rs if x['status']=='OK' and x[m] and np.isfinite(float(x[m]))]
            assert len(vs)==int(r[m+'_n'])
            if vs:assert math.isclose(float(r[m+'_mean']),np.mean(vs),abs_tol=1e-12)
            if len(vs)>1:assert math.isclose(float(r[m+'_std']),np.std(vs,ddof=1),abs_tol=1e-12)
            else:assert not r[m+'_std']
    selected_checks=0
    for r in models:
        for file in (out/r['trial']).glob('node_*.json'):
            d=json.loads(file.read_text())
            if d['status']!='OK':continue
            trial=file.parent;base=trial.parent
            yv=np.load(base/'y_validation.npy',mmap_mode='r')[d['node']-1]
            y=np.load(base/'y_train.npy',mmap_mode='r')[d['node']-1]
            path=d['validation_path'];assert len(path)==12
            floor=1e-12*max(float(np.mean((yv-y.mean())**2)),np.finfo(float).tiny)
            best=min(x['MSE'] for x in path)
            selected=next(i for i,x in enumerate(path) if x['MSE']<=best+floor)
            assert selected==d['selected_path_index']
            shared=json.loads((base/'shared_inputs.json').read_text())
            assert d['input_fingerprint']==shared['input_fingerprint']
            selected_checks+=1
    if len(contract['repetitions'])==1:
        assert all(not r['ci_low'] and not r['ci_high'] for r in load_csv(out/'paired_summary.csv'))
    return dict(status='PASS',models=len(models),network_metric_records=checked,
        undefined_MCC=undefined,selection_checks=selected_checks,
        scores_recomputed_independently=True,summary_sample_sd_verified=True,
        all_original_run_source_snapshots_and_selection_locks_verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.run)
    if args.output.exists():raise FileExistsError(args.output)
    save_json(args.output,result);print(json.dumps(result))
