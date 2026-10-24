import copy
import inspect
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.experiments.fair_v2 import (ROOT,FRACTIONS,METHODS,SYSTEMS,seed_design,network,validate_network,
    make_plan,read_plan,instance_config,fit_scaling,standardize,noise,solve)
from src.experiments.fair_v2_statistics import summarise,clustered_bootstrap
from src.preprocessing.taylor import restore_coefficients
from src.solvers.large_library import initial_ridge,stlsq_dual
from src.solvers.lars_validation import validation_lasso_accelerated
from src.solvers.stlsq_validation import stlsq_validation,select_validation_path
from src.metrics.structure import edge_scores,binary_metrics


def options():
    return dict(lasso=dict(alpha_fractions=FRACTIONS,max_iter=100000,tol=1e-10,kkt_tolerance=1e-6,
        validation_tolerance=1e-12,working_set_rounds=60,working_set_add=64),
        stlsq=dict(rho=1.,niter=10,alpha_fractions=FRACTIONS,validation_tolerance=1e-12),
        tie_rule='first descending relative penalty within tolerance * validation response energy')


class FairSolverTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(304)
        self.raw=np.c_[np.ones(180),rng.normal(size=(180,12))*np.arange(1,13)+2]
        self.rawv=np.c_[np.ones(70),rng.normal(size=(70,12))*np.arange(1,13)+2]
        self.means,self.scales=fit_scaling(self.raw)
        self.Z=standardize(self.raw,self.means,self.scales);self.Zv=standardize(self.rawv,self.means,self.scales)
        self.beta=np.r_[7.3,.8,-.5,np.zeros(10)]
        self.y=self.Z@self.beta+.01*rng.normal(size=180);self.yv=self.Zv@self.beta+.01*rng.normal(size=70)
        self.init=initial_ridge(self.Z[:,1:],self.y[:,None],1.)[0];self.c=options()

    def test_free_intercept_translation_invariance(self):
        a,ad=stlsq_validation(self.Z[:,1:],self.y,self.Zv[:,1:],self.yv,self.c['stlsq'])
        b,bd=stlsq_validation(self.Z[:,1:],self.y+100,self.Zv[:,1:],self.yv+100,self.c['stlsq'])
        np.testing.assert_allclose(a[1:],b[1:],atol=1e-12)
        self.assertAlmostEqual(b[0]-a[0],100)
        self.assertEqual(ad['selected_path_index'],bd['selected_path_index'])
        self.assertAlmostEqual(a[0],self.y.mean())

    def test_constant_response_and_zero_intercept(self):
        for level in (0.,42.):
            coef,_=stlsq_validation(self.Z[:,1:],np.full(180,level),self.Zv[:,1:],np.full(70,level),self.c['stlsq'])
            np.testing.assert_array_equal(coef,np.r_[level,np.zeros(12)])

    def test_no_double_centering_or_penalized_constant(self):
        with self.assertRaises(ValueError):stlsq_validation(self.Z,self.y,self.Zv,self.yv,self.c['stlsq'])
        with self.assertRaises(ValueError):stlsq_validation(self.Z[:,1:]+1,self.y,self.Zv[:,1:],self.yv,self.c['stlsq'])

    def test_all_three_predict_in_original_coordinates(self):
        for m in METHODS:
            coef,d=solve(m,self.Z,self.y,self.Zv,self.yv,self.init,self.c)
            raw=restore_coefficients(coef[None,:],self.means,self.scales)[0]
            np.testing.assert_allclose(self.raw@raw,self.Z@coef,atol=1e-11)
            np.testing.assert_allclose(self.rawv@raw,self.Zv@coef,atol=1e-11)
            self.assertFalse(d['intercept_penalized']);self.assertFalse(d['internally_standardized'])
            self.assertEqual(len(d['validation_path']),12)

    def test_lasso_adapter_preserves_old_objective_and_weights(self):
        for method in ['LASSO_val','Adaptive_val']:
            a=np.ones(12) if method=='LASSO_val' else np.abs(self.init)+1e-6
            old,od=validation_lasso_accelerated(self.Z[:,1:],self.y,self.Zv[:,1:],self.yv,a,self.c['lasso'])
            new,nd=solve(method,self.Z,self.y,self.Zv,self.yv,self.init,self.c)
            np.testing.assert_array_equal(old,new);self.assertEqual(od['alpha'],nd['alpha'])
            self.assertLessEqual(max(x['full_library_kkt'] for x in nd['validation_path']),1e-6)

    def test_stlsq_path_matches_pinned_nonconstant_mechanism(self):
        coef,diag=stlsq_validation(self.Z[:,1:],self.y,self.Zv[:,1:],self.yv,self.c['stlsq'])
        for row in diag['validation_path']:
            beta,_=stlsq_dual(self.Z[:,1:],(self.y-self.y.mean())[:,None],row['parameter'],1.,10)
            mse=np.mean((self.y.mean()+self.Zv[:,1:]@beta[0]-self.yv)**2)
            self.assertAlmostEqual(mse,row['MSE'],places=12)

    def test_training_initial_ridge_and_grid_ignore_validation(self):
        _,a=stlsq_validation(self.Z[:,1:],self.y,self.Zv[:,1:],self.yv,self.c['stlsq'])
        _,b=stlsq_validation(self.Z[:,1:],self.y,self.Zv[:,1:],self.yv*3,self.c['stlsq'])
        self.assertEqual(a['parameter_reference'],b['parameter_reference'])
        self.assertEqual([x['nonzero'] for x in a['validation_path']],[x['nonzero'] for x in b['validation_path']])
        self.assertFalse({'truth','test','topology'}&set(inspect.signature(solve).parameters))

    def test_common_tie_rule(self):
        path=[dict(MSE=1.+1e-13),dict(MSE=1.)]
        self.assertEqual(select_validation_path(path,np.array([2.,2.]),0.,1e-12),0)
        self.assertEqual(select_validation_path(path,np.array([2.,2.]),0.,1e-15),1)

    def test_intercept_does_not_change_edge_decode(self):
        d=np.array([[0,0,0],[0,0,2],[0,2,3]])
        C=np.array([[0.,.5,.2],[0.,0.,0.],[0.,0.,0.]])
        for order in (2,3):
            keys,scores=edge_scores(C,d,order);C[:,0]+=10000
            k2,s2=edge_scores(C,d,order);self.assertEqual(keys,k2);np.testing.assert_array_equal(scores,s2)

    def test_scaling_is_training_only(self):
        m,s=fit_scaling(self.raw)
        self.rawv[:]=999
        m2,s2=fit_scaling(self.raw)
        np.testing.assert_array_equal(m,m2);np.testing.assert_array_equal(s,s2)

    def test_repeat_solve_is_deterministic(self):
        for method in METHODS:
            a,ad=solve(method,self.Z,self.y,self.Zv,self.yv,self.init,self.c)
            b,bd=solve(method,self.Z,self.y,self.Zv,self.yv,self.init,self.c)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(ad['selected_path_index'],bd['selected_path_index'])

    def test_underdetermined_stlsq_free_intercept(self):
        rng=np.random.default_rng(21)
        X=rng.normal(size=(25,40));X-=X.mean(axis=0);Xv=rng.normal(size=(20,40))
        y=3+X[:,2];yv=3+Xv[:,2]
        a,d=stlsq_validation(X,y,Xv,yv,self.c['stlsq'])
        expected=np.linalg.solve(X.T@X+np.eye(40),X.T@(y-y.mean()))
        self.assertAlmostEqual(d['parameter_reference'],np.max(np.abs(expected)),places=12)
        self.assertAlmostEqual(a[0],3)


class SeedDesignTests(unittest.TestCase):
    def test_reproducible_independent_10_by_3(self):
        rows=seed_design();self.assertEqual(rows,seed_design());self.assertEqual(len(rows),30)
        self.assertEqual(len({r['network_seed'] for r in rows}),10)
        self.assertEqual(len({r['data_seed'] for r in rows}),30)
        domains=[{r[k] for r in rows} for k in ['network_seed','topology_seed','weight_seed','data_seed','noise_seed']]
        self.assertEqual(sum(map(len,domains)),len(set.union(*domains)))
        names=set()
        for r in rows:
            for code in range(1,6):
                ids={r['data_seed']*1000000+code*100000+i for i in range(1,1537)}
                self.assertFalse(ids&names);names|=ids
        self.assertEqual(len(names),30*5*1536)

    def test_network_constraints_and_topology_weight_separation(self):
        topologies=set()
        for r in seed_design()[::3]:
            p,t=network(r['topology_seed'],r['weight_seed']);validate_network(p,t)
            self.assertEqual((p,t),network(r['topology_seed'],r['weight_seed']))
            p2,t2=network(r['topology_seed'],r['weight_seed']+1)
            self.assertEqual([x[:-1] for x in p+t],[x[:-1] for x in p2+t2]);self.assertNotEqual(p,p2)
            topologies.add(str([x[:-1] for x in p+t]))
        self.assertEqual(len(topologies),10)

    def test_plan_lock_common_topology_and_fixed_intrinsics(self):
        with tempfile.TemporaryDirectory() as d:
            c=make_plan(Path(d)/'plan');self.assertEqual(c,read_plan(Path(d)/'plan'))
            a=instance_config(c,c['repetitions'][0],['kuramoto','sis'])
            b=instance_config(c,c['repetitions'][3],SYSTEMS)
            for s in SYSTEMS:
                self.assertEqual(a[s]['center'],b[s]['center'])
                self.assertEqual([[*x[:-1],abs(x[-1])] for x in a[s]['pair_edges']],a['kuramoto']['pair_edges'])
            self.assertEqual(a['systems'],SYSTEMS)
            with (Path(d)/'plan/seed_manifest.csv').open('a') as f:f.write('tamper')
            with self.assertRaises(AssertionError):read_plan(Path(d)/'plan')

    def test_noise_replay_shared_methods_and_stream_disjointness(self):
        rows=seed_design();Y=np.arange(3*18,dtype=float).reshape(3,18);split=np.array(['train']*10+['validation']*4+['test']*4)
        u,s,streams=noise(Y,split=='train',rows[0],1,split)
        u2,s2,streams2=noise(Y,split=='train',rows[0],1,split)
        np.testing.assert_array_equal(u,u2);np.testing.assert_array_equal(s,s2)
        self.assertEqual(streams,streams2);self.assertEqual(len(set(map(tuple,streams))),9)
        u3,_,ss=noise(Y,split=='train',rows[1],1,split)
        self.assertFalse(np.array_equal(u,u3));self.assertFalse(set(map(tuple,streams))&set(map(tuple,ss)))
        Y[:,10:]+=1e6;_,s3,_=noise(Y,split=='train',rows[0],1,split)
        np.testing.assert_array_equal(s,s3)


class StatisticsTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for n in range(3):
            for data in range(3):
                for mi,m in enumerate(METHODS):
                    x=(n+data+mi)/10
                    rows.append(dict(system='sis',noise_ratio=.05,order='triple',method=m,
                        instance_id=f'{n}_{data}',network_id=n,network_seed=n,data_seed=10*n+data,
                        noise_seed=100*n+data,dataset_sha256='data',noise_sha256='noise',status='OK',
                        F1=x,AP=x,MCC=x,Jaccard=x,TP=1,FP=2,FN=3,TN=4,MCC_status='OK'))
        return rows

    def test_sample_sd_and_exclude_failures(self):
        rows=self.rows();rows[0]['status']='FAILED'
        summary,_,pairs,_=summarise(rows,resamples=50)
        r=next(r for r in summary if r['method']=='STLSQ_val')
        vals=[x['F1'] for x in rows if x['method']=='STLSQ_val' and x['status']=='OK']
        self.assertEqual(r['F1_n'],8);self.assertEqual(r['FAILED'],1)
        self.assertAlmostEqual(r['F1_std'],np.std(vals,ddof=1))
        self.assertTrue(any(x['reason'] for x in pairs))

    def test_mcc_undefined_preserved(self):
        rows=self.rows();rows[0].update(MCC=float('nan'),MCC_status='NA')
        s,_,_,_=summarise(rows,resamples=30)
        r=next(x for x in s if x['method']=='STLSQ_val')
        self.assertEqual(r['MCC_n'],8);self.assertEqual(r['F1_n'],9);self.assertEqual(r['OK'],9)

    def test_pair_alignment_and_differences(self):
        rows=self.rows();_,_,pairs,summary=summarise(rows,resamples=50)
        for r in pairs:self.assertAlmostEqual(r['difference'],.2 if r['comparison']=='Adaptive_val - STLSQ_val' else .1)
        self.assertTrue(all(r['networks']==3 and r['valid_pairs']==9 for r in summary))
        rows[1]['data_seed']=-1
        with self.assertRaises(AssertionError):summarise(rows,resamples=30)

    def test_cluster_bootstrap_not_individual_repeat_bootstrap(self):
        a=clustered_bootstrap({0:[0.],1:[1.],2:[2.]},123,500)
        b=clustered_bootstrap({0:[0.]*100,1:[1.]*100,2:[2.]*100},123,500)
        self.assertEqual(a,b)
        self.assertIsNone(clustered_bootstrap({0:[0.,1.,2.]},1)['ci_low'])

    def test_duplicate_instances_rejected(self):
        rows=self.rows()
        with self.assertRaises(ValueError):summarise(rows+[rows[0]])

    def test_metrics_from_scores_and_jaccard_per_seed(self):
        a=binary_metrics(np.array([1,1,0,0],bool),np.array([.9,.8,.7,.0]),.5)
        self.assertAlmostEqual(a['F1'],.8);self.assertEqual(a['AUPRC_AP'],1.)
        j=a['TP']/(a['TP']+a['FP']+a['FN']);self.assertAlmostEqual(j,2/3)
        b=binary_metrics(np.array([1,1,0,0],bool),np.zeros(4))
        self.assertEqual(b['MCC_status'],'NA');self.assertTrue(np.isnan(b['MCC']))
        self.assertNotAlmostEqual((j+0)/2,((a['F1']+0)/2)/(2-(a['F1']+0)/2))

    def test_formal_900_model_statistics_shape_without_fitting(self):
        rows=[]
        for system in SYSTEMS:
            for ratio in (0.,.05):
                for order in ('pairwise','triple'):
                    for net in range(10):
                        for data in range(3):
                            for mi,method in enumerate(METHODS):
                                x=(net+data+mi)/30
                                rows.append(dict(system=system,noise_ratio=ratio,order=order,method=method,
                                    instance_id=f'{net}_{data}',network_id=net,network_seed=net,data_seed=net*3+data,
                                    noise_seed=100+net*3+data,dataset_sha256='same',noise_sha256='same',status='OK',
                                    F1=x,AP=x,MCC=x,Jaccard=x,TP=1,FP=2,FN=3,TN=4,MCC_status='OK'))
        summary,stability,pairs,ps=summarise(rows,resamples=20)
        self.assertEqual(len(rows),1800);self.assertEqual(len(summary),60)
        self.assertEqual(len(ps),240);self.assertEqual(len(pairs),7200)
        self.assertTrue(all(r['OK']==30 and r['F1_n']==30 for r in summary))
        self.assertTrue(all(r['networks']==10 and r['valid_pairs']==30 for r in ps))

    def test_model_NA_is_not_zero(self):
        rows=self.rows();rows[0]['status']='NA';rows[0]['F1']=0.
        summary,_,pairs,_=summarise(rows,resamples=20)
        s=next(x for x in summary if x['method']=='STLSQ_val')
        self.assertEqual(s['NA'],1);self.assertEqual(s['F1_n'],8)
        self.assertTrue(any(r['status']=='NA' for r in pairs))


if __name__=='__main__':unittest.main()
