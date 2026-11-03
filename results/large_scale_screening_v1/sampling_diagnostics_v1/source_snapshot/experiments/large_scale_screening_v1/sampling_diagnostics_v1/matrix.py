"""B4 fixed-candidate nested libraries and B1 row-permutation negative control."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from ..common import save,table
from ..data import load_data
from ..library import terms_and_groups
from .sampling import coordinates
from .diagnostics import design_diagnostics,spectrum,lag1


def run(out,cfg):
    out=out;rows=[];permutations=[];spectra={};distributions=[]
    folders=[]
    for folder in sorted((out/'jobs').iterdir()):
        job=json.loads((folder/'job.json').read_text())
        if job['seed_index']<2 and job['N']==30:folders.append((folder,job))
    if not folders:raise ValueError('Run B1 and pilot first')
    with threadpool_limits(limits=1):
        for seed in [0,1]:
            baseline=next(f for f,j in folders if j['kind']=='B1' and j['seed_index']==seed and j['train_trajectories']==4)
            data=load_data(baseline/'data');train=data['splits']=='train';x,_=coordinates(data)
            with np.load(baseline/'arrays.npz') as a:
                selected={target:a[f'r0.05_fwl_i{target}_k16_selected'] for target in [0,1,2]}
            terms,_=terms_and_groups(30,0,selected[0]);info,s,matrix=design_diagnostics(x[train],terms)
            permutation=np.random.default_rng(2026110300+seed).permutation(len(matrix))
            pi,ps=spectrum(matrix[permutation]);relative=float(np.max(abs(s-ps))/s[0])
            if relative>1e-12:raise AssertionError('Permutation changed singular spectrum beyond roundoff')
            rec=dict(seed_index=seed,target=0,rows=len(matrix),columns=matrix.shape[1],rank_before=info['rank'],rank_after=pi['rank'],
                spectrum_max_abs_difference=float(np.max(abs(s-ps))),spectrum_difference_relative_smax=relative,
                lag1_original_row_order=lag1(data['derivative'][train]),lag1_shuffled_row_order=lag1(data['derivative'][train][permutation]),
                lag1_original_within_trajectory=data['diagnostics']['derivative_lag1_median'],
                note='same columns and samples, no re-screening; numerical ranks near the default cutoff may fluctuate at roundoff')
            permutations.append(rec);spectra[f'shuffle_s{seed}_original']=s;spectra[f'shuffle_s{seed}_permuted']=ps
            np.save(out/f'row_permutation_seed{seed}.npy',permutation)
            iid=next(f for f,j in folders if j['kind']=='A' and j['seed_index']==seed)
            iid_data=load_data(iid/'data');ix,_=coordinates(iid_data);ix=ix[iid_data['splits']=='train']
            from scipy.stats import wasserstein_distance,ks_2samp
            distributions.append(dict(seed_index=seed,comparison='trajectory4_T0.5 versus local_iid_delta0.6',
                mean_node_wasserstein=float(np.mean([wasserstein_distance(x[train,i],ix[:,i]) for i in range(30)])),
                mean_node_KS_distance=float(np.mean([ks_2samp(x[train,i],ix[:,i]).statistic for i in range(30)])),
                iid_mean_abs_node_mean=float(abs(ix.mean(axis=0)).mean()),trajectory_mean_abs_node_mean=float(abs(x[train].mean(axis=0)).mean()),
                iid_mean_node_std=float(ix.std(axis=0).mean()),trajectory_mean_node_std=float(x[train].std(axis=0).mean())))
            # Include every nonfactorial diagnostic dataset, not just visually favorable examples.
            for folder,job in folders:
                if job['seed_index']!=seed:continue
                d=load_data(folder/'data');train=d['splits']=='train';dy=d['derivative'][train]
                for target in [0,1,2]:
                    full,_=terms_and_groups(30,target,selected[target])
                    for center_mode,scale in [('fixed','standard'),('fixed','raw'),('circular','standard'),('arithmetic','standard')]:
                        xx,_=coordinates(d,center_mode);xt=xx[train];b=np.column_stack([np.ones(len(xt)),xt])
                        q,bs,_=np.linalg.svd(b,full_matrices=False);brank=int(np.sum(bs>max(b.shape)*np.finfo(float).eps*bs[0]));q=q[:,:brank]
                        yr=dy[:,target]-q@(q.T@dy[:,target]);r2=1-float(yr@yr)/float(np.sum((dy[:,target]-dy[:,target].mean())**2))
                        for degree in [1,2,3]:
                            terms=[t for t in full if len(t)<=degree];info,s,m=design_diagnostics(xt,terms,scale_mode=scale)
                            nonlin=[i-1 for i,t in enumerate(terms) if len(t)>1]
                            residual=m[:,nonlin]-q@(q.T@m[:,nonlin]) if nonlin else None
                            total=float(np.sum(m[:,nonlin]**2)) if nonlin else 0.
                            info.update(job=job['name'],seed_index=seed,target=target,degree=degree,center_mode=center_mode,scale_mode=scale,
                                fixed_candidates_from='original trajectory4 FWL K16',sampling_mode=job['sampling_mode'],
                                trajectories=job['train_trajectories'] if job['sampling_mode']=='trajectory' else 0,
                                delta=job['delta'] if job['sampling_mode']=='local_iid' else None,duration=job['duration'] if job['sampling_mode']=='trajectory' else None,
                                low_order_response_R2=r2,nonlinear_terms_low_order_R2=1-float(np.sum(residual**2))/total if total else None)
                            rows.append(info);spectra[f'{job["name"]}_i{target}_{center_mode}_{scale}_degree{degree}']=s
    table(out/'matrix_spectrum.csv',rows);table(out/'row_shuffle_control.csv',permutations)
    table(out/'state_distribution_comparison.csv',distributions);np.savez_compressed(out/'matrix_singular_values.npz',**spectra)
    save(out/'matrix_completion.json',dict(designs=len(rows),shuffle_controls=len(permutations),fixed_candidates=True))
    print('MATRIX',len(rows),'designs;',len(permutations),'shuffle controls',flush=True)
