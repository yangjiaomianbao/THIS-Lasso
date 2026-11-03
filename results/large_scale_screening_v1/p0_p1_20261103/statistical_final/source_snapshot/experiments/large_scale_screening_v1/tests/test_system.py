"""Scientific identities and failure boundaries, not just interface smoke tests."""
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from experiments.large_scale_screening_v1.common import ROOT,load_config
from experiments.large_scale_screening_v1.screeners import Context,Screener,STAT_METHODS,group_stat
from experiments.large_scale_screening_v1.candidates import blocks,shared_pool,count
from experiments.large_scale_screening_v1.library import terms_and_groups,evaluate,native_indices,decode
from experiments.large_scale_screening_v1.metrics import sparse_metrics,screening_metrics
from experiments.large_scale_screening_v1.data import network,rhs,generate,noisy_derivative,coordinates
from experiments.large_scale_screening_v1.backend import fit,standardized
from src.solvers.methods import fit_node
from src.preprocessing.taylor import restore_coefficients
from src.data.contracts import validate_splits


class ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(limits=1);cls.cfg=load_config()
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def fixture(self,mode='product',samples=4000):
        rng=np.random.default_rng(91);x=rng.normal(size=(samples,7));y=rng.normal(scale=.1,size=x.shape)
        y[:,0]=x[:,1]*x[:,2]+.01*rng.normal(size=samples) if mode=='product' else 2*x[:,1]-.7*x[:,3]
        return x,y,np.concatenate(list(blocks(7,0,64)))
    def score(self,method,x,y,p,**parameters):
        return Screener(method,dict(self.cfg['screening'],**parameters)).score(x,y,0,p)
    def test_product_residual_screens(self):
        x,y,p=self.fixture()
        for method in ['fwl','scc','scc_ols']:
            r=self.score(method,x,y,p);self.assertEqual(tuple(p[r['ranking'][0]]),(1,2))
    def test_linear_negative_control_fwl(self):
        x,y,p=self.fixture('linear');r=self.score('fwl',x,y,p)
        self.assertLess(max(r['scores']),1e-20)
    def test_correlated_proxy_is_not_causal_identification(self):
        x,y,p=self.fixture();x[:,3]=x[:,1]+.03*np.random.default_rng(1).normal(size=len(x))
        scores=self.score('fwl',x,y,p)['scores'];lookup=dict(zip(map(tuple,p),scores))
        self.assertGreater(lookup[(2,3)],.9*lookup[(1,2)])
    def test_affine_scale_invariance(self):
        x,y,p=self.fixture(samples=700);scaled=x*np.arange(1,8)*3+np.arange(7)
        for m in STAT_METHODS:
            np.testing.assert_allclose(self.score(m,x,y,p)['scores'],self.score(m,scaled,y,p)['scores'],atol=1e-10,rtol=1e-9)
    def test_constant_and_small_groups(self):
        x,y,p=self.fixture(samples=8);x[:,1]=1
        for m in STAT_METHODS:
            r=self.score(m,x,y,p,min_group=5);self.assertTrue(np.isfinite(r['scores']).all())
            if m in ['scc','scc_ols']:np.testing.assert_array_equal(r['scores'],0)
    def test_scc_slope_is_intercept_ols_not_correlation(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(100,2));y=3+2*x[:,0,None]+rng.normal(size=(100,1));g=np.ones_like(x,bool)
        slopes,_=group_stat(x,y,g,'slope',3)
        for j in range(2):
            expected=np.linalg.lstsq(np.column_stack([np.ones(100),x[:,j]]),y[:,0],rcond=None)[0][1]
            self.assertAlmostEqual(slopes[j],expected,places=12)
        corr,_=group_stat(x,y,g,'corr',3);self.assertGreater(max(abs(slopes-corr)),.3)
    def test_fwl_exact_added_feature_ols(self):
        x,y,p=self.fixture(samples=700);ctx=Context(x,y);r=Screener('fwl',self.cfg['screening'],ctx).score(x,y,0,p)
        j,k=1,2;z=ctx.z[:,j]*ctx.z[:,k];b=np.column_stack([np.ones(len(x)),ctx.z]);res=y[:,0]-b@np.linalg.lstsq(b,y[:,0],rcond=None)[0]
        aug=np.column_stack([b,z]);res2=y[:,0]-aug@np.linalg.lstsq(aug,y[:,0],rcond=None)[0]
        self.assertAlmostEqual(dict(zip(map(tuple,p),r['scores']))[(j,k)],res@res-res2@res2,places=9)
    def test_fwl_nearly_projected_product_avoids_norm_cancellation(self):
        rng=np.random.default_rng(72);x=rng.normal(size=(700,5));u=rng.normal(size=700)
        x[:,3]=x[:,1]*x[:,2]+1e-7*u;y=np.zeros_like(x);y[:,0]=x[:,1]*x[:,2]+.1*u
        p=np.array([[1,2]]);ctx=Context(x,y);score=Screener('fwl',self.cfg['screening'],ctx).score(x,y,0,p)['scores'][0]
        z=ctx.z[:,1]*ctx.z[:,2];b=np.column_stack([np.ones(len(x)),ctx.z]);aug=np.column_stack([b,z])
        residual=y[:,0]-aug@np.linalg.lstsq(aug,y[:,0],rcond=None)[0]
        expected=ctx.residual[:,0]@ctx.residual[:,0]-residual@residual
        self.assertGreater(score,1);self.assertAlmostEqual(score,expected,places=6)
    def test_gc_product_failure_boundary(self):
        # Exact sign-balanced quadrants remove all Gaussian-copula pair correlations.
        rng=np.random.default_rng(3);a=rng.uniform(.1,2,1000);b=rng.uniform(.1,2,1000)
        x=np.concatenate([np.column_stack([np.zeros(1000),sj*a,sk*b]) for sj,sk in itertools.product([-1,1],repeat=2)])
        y=np.zeros_like(x);y[:,0]=x[:,1]*x[:,2]
        r=self.score('gc_cmi',x,y,np.array([[1,2]]));self.assertLess(r['scores'][0],1e-12)
        self.assertTrue(r['experimental'])
    def test_gc_additive_nonlinear_and_proxy_finite(self):
        x,y,p=self.fixture(samples=900);x[:,3]=x[:,1]+.1*np.random.default_rng(3).normal(size=len(x));y[:,0]=x[:,1]**2+np.sin(x[:,2])
        r=self.score('gc_cmi',x,y,p);self.assertTrue(np.isfinite(r['scores']).all());self.assertTrue((r['scores']>=0).all())
    def test_gc_additive_nonlinear_false_positive_boundary(self):
        rng=np.random.default_rng(88);x=rng.normal(size=(4000,5));x[:,2]=.8*x[:,1]+.6*x[:,2]
        y=np.zeros_like(x);y[:,0]=np.sin(2*x[:,1])+x[:,2]**2
        # No joint term, yet the Gaussian-copula synergy approximation is nonzero.
        score=self.score('gc_cmi',x,y,np.array([[1,2]]))['scores'][0]
        self.assertGreater(score,5e-4)
    def test_periodic_coordinates_ignore_two_pi_shifts(self):
        rng=np.random.default_rng(2);phase=rng.normal(size=(120,5));mask=np.arange(120)<80
        x,_,_=coordinates(phase,mask);z,_,_=coordinates(phase+2*np.pi*np.arange(5),mask)
        np.testing.assert_allclose(x,z,atol=1e-14)
    def test_pair_symmetry(self):
        x,y,p=self.fixture(samples=300)
        for m in STAT_METHODS:np.testing.assert_allclose(self.score(m,x,y,p)['scores'],self.score(m,x,y,p[:,::-1])['scores'],atol=1e-12)
    def test_blocked_matches_single_block(self):
        x,y,p=self.fixture(samples=300)
        for m in STAT_METHODS:np.testing.assert_allclose(self.score(m,x,y,p,block_size=2)['scores'],self.score(m,x,y,p,block_size=100)['scores'],atol=1e-11)
    def test_candidate_counts_pool_and_k(self):
        for n in [6,30,50,1000]:
            for target in [0,n-1]:
                pool=shared_pool(n,target,50,9);self.assertEqual(len(pool),min(count(n),50));self.assertFalse(np.any(pool==target));self.assertEqual(len(np.unique(pool,axis=0)),len(pool))
                np.testing.assert_array_equal(pool,shared_pool(n,target,50,9))
            for k in [16,32,64,128]:
                p=shared_pool(n,0,150,9);metric=screening_metrics({tuple(p[0])},p,np.linspace(1,.1,len(p)),np.arange(len(p)),k,count(n))
                self.assertEqual(metric['selected_count'],min(k,len(p)));self.assertAlmostEqual(metric['Recall_at_K'],1)
    def test_library_groups_and_no_duplicate_monomials(self):
        for n in [6,30,1000]:
            p=shared_pool(n,0,16,7);ts,g=terms_and_groups(n,0,p)
            self.assertEqual(len(ts),6*n-2+4*len(p));self.assertEqual(len(set(ts)),len(ts))
            for (j,k),indices in g.items():
                self.assertEqual(len(indices),4);self.assertIn((j,k),[ts[i] for i in indices]);self.assertTrue(all(set(ts[i])<={0,j,k} for i in indices))
    def test_decode_excludes_repeated_and_target_terms(self):
        ts=[(),(0,),(1,),(1,2),(1,1,2),(0,1,2),(1,1)];c=np.array([1,1,.2,.3,5,7,9.])
        s=decode(c,ts,0);self.assertEqual(s[2],{(0,1):.2});self.assertEqual(s[3],{(0,1,2):.3})
    def test_backend_numerical_regression_to_public_M2(self):
        rng=np.random.default_rng(18);x=rng.uniform(-1,1,(400,5));ts,_=terms_and_groups(5,0,np.array([[1,2],[3,4]]))
        theta=evaluate(x,ts);y=.3+x[:,1]+.5*x[:,1]*x[:,2]+rng.normal(scale=.05,size=len(x))
        result=fit(x,y,ts,self.cfg['backend']);self.assertEqual(result['status'],'OK')
        z,mean,scale=standardized(theta);config=dict(self.cfg['backend']);config.pop('selection')
        c,d=fit_node('M2',z,y,dict(lasso={},adaptive=config));expected=restore_coefficients(c[None,:],mean,scale)[0]
        np.testing.assert_array_equal(result['coefficient'],expected);self.assertEqual(result['diagnostics']['alpha'],d['alpha'])
        np.testing.assert_array_equal(result['diagnostics']['weights'],d['weights'])
    def test_backend_NA_not_fallback(self):
        x=np.random.default_rng(1).normal(size=(20,5));ts,_=terms_and_groups(5,0,np.array([[1,2]]))
        self.assertEqual(fit(x,x[:,0],ts,self.cfg['backend'])['status'],'NA')
        x=np.tile(np.linspace(-1,1,400)[:,None],(1,5));self.assertEqual(fit(x,np.arange(400),ts,self.cfg['backend'])['status'],'NA')
    def test_sparse_ap_exact_including_zero_ties(self):
        rng=np.random.default_rng(2);truth={1,3,10};scores={1:.8,2:.7,3:.4,7:.1};m=sparse_metrics(truth,scores,1000)
        y=np.array([i in truth for i in range(1000)]);s=np.array([scores.get(i,0) for i in range(1000)])
        self.assertAlmostEqual(m['AP'],average_precision_score(y,s),places=15)
        self.assertNotEqual(m['AP'],average_precision_score(y,s>1e-8))
    def test_sparse_rhs_matches_explicit_edges_and_zero_degree(self):
        c=self.cfg['data'];net=network(8,12,c);theta=np.random.default_rng(8).normal(size=8)
        expected=net['omega'].copy()
        for i,j in net['pairs']:expected[i]+=c['K2']/net['degree2'][i]*np.sin(theta[j]-theta[i])
        for i,j,k in net['triples']:expected[i]+=c['K3']/net['degree3'][i]*np.sin(theta[j]+theta[k]-2*theta[i])
        np.testing.assert_allclose(rhs(theta,net,c),expected)
        z=network(8,12,dict(c,mean_pair_degree=0,mean_triad_degree=0));np.testing.assert_array_equal(rhs(theta,z,c),z['omega'])
    def test_whole_trajectory_split_and_seed_separation(self):
        d=generate(6,110,220,self.cfg['data'],[256,256,256]);self.assertEqual(validate_splits(d['groups'],d['splits']),dict(train=1,validation=1,test=1))
        train=d['splits']=='train';x,center,_=coordinates(d['phase'],train)
        changed=d['phase'].copy();changed[~train]+=1
        _,center2,_=coordinates(changed,train);np.testing.assert_array_equal(center,center2)
        broken=d['splits'].copy();broken[0]='test'
        with self.assertRaises(ValueError):validate_splits(d['groups'],broken)
        d2=generate(6,110,221,self.cfg['data'],[256,256,256]);np.testing.assert_array_equal(d['net']['omega'],d2['net']['omega']);self.assertFalse(np.array_equal(d['phase'],d2['phase']))
    def test_noise_same_unit_stream_across_ratios(self):
        d=generate(6,110,220,self.cfg['data'],[256,256,256]);a=noisy_derivative(d,.05,1,2);b=noisy_derivative(d,.1,1,2)
        np.testing.assert_allclose(a-d['derivative'],.5*(b-d['derivative']),atol=1e-15)
    def test_neural_symmetry_both_architectures(self):
        import torch
        from experiments.large_scale_screening_v1.neural import TriadicModel
        torch.set_num_threads(1);torch.manual_seed(3);x=torch.randn(3,12,3);y=torch.randn(3,12)
        for architecture in ['light','transformer']:
            model=TriadicModel(architecture).eval()
            with torch.no_grad():np.testing.assert_allclose(model(x,y).numpy(),model(x[:,:,[0,2,1]],y).numpy(),atol=1e-7)
    @unittest.skipUnless(os.environ.get('THIS_JULIA') or shutil.which('julia'),'Set THIS_JULIA to verify actual native Julia parity')
    def test_actual_native_julia_library_and_decode(self):
        julia=os.environ.get('THIS_JULIA') or shutil.which('julia')
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);x=np.random.default_rng(9).normal(size=(15,4));np.savetxt(folder/'x.csv',x.T,delimiter=',')
            program='''using DelimitedFiles
include(joinpath(ARGS[1],"src","this_adapter.jl"))
using .THISAdapter
p=ARGS[2];X=Matrix{Float64}(readdlm(joinpath(p,"x.csv"),',',Float64));theta,d=get_θd(X,3)
writedlm(joinpath(p,"theta.csv"),theta,',');writedlm(joinpath(p,"d.csv"),d,',')
C=zeros(4,size(d,1));C[1,:]=collect(1:size(d,1))./100
A=decode_coefficients(C,d,4,3)
for order in (2,3);writedlm(joinpath(p,"edges$(order).csv"),A[order],',');end
'''
            (folder/'check.jl').write_text(program)
            env=dict(os.environ,JULIA_DEPOT_PATH=str(folder/'depot')+':'+str(ROOT/'.julia-depot-r0'),JULIA_NUM_PRECOMPILE_TASKS='1')
            subprocess.run([julia,'--startup-file=no','--project='+str(ROOT/'environment/r0/julia'),str(folder/'check.jl'),str(ROOT),str(folder)],check=True,env=env,capture_output=True,timeout=120)
            d=np.loadtxt(folder/'d.csv',delimiter=',',dtype=int);ts=[tuple(sorted(j-1 for j in row if j)) for row in d]
            np.testing.assert_allclose(evaluate(x,ts),np.loadtxt(folder/'theta.csv',delimiter=',').T,atol=1e-13)
            np.testing.assert_array_equal(native_indices(ts),d)
            coef=np.arange(1,len(ts)+1)/100;ours=decode(coef,ts,0)
            for order in [2,3]:
                rows=np.loadtxt(folder/f'edges{order}.csv',delimiter=',',ndmin=2)
                native={tuple(int(j)-1 for j in row[:-1]):float(row[-1]) for row in rows}
                self.assertEqual(ours[order],native)


if __name__=='__main__':unittest.main()
