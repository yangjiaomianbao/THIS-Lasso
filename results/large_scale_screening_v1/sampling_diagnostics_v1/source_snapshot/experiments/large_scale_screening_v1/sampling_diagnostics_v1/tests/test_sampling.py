import unittest
from unittest.mock import patch
import numpy as np
from experiments.large_scale_screening_v1.common import load_config
from experiments.large_scale_screening_v1.data import generate as old_generate,rhs,noisy_derivative
from experiments.large_scale_screening_v1.library import terms_and_groups,native_indices,evaluate
from experiments.large_scale_screening_v1.sampling_diagnostics_v1.sampling import generate,coordinates,triadic_coefficients,taylor_rhs
from experiments.large_scale_screening_v1.sampling_diagnostics_v1.diagnostics import design_diagnostics,lag1,spectrum
from src.data.contracts import validate_splits


class SamplingTests(unittest.TestCase):
    def setUp(self):self.cfg=load_config()['data']
    def iid(self,**kw):return generate(8,110,220,self.cfg,[128,32,32],**kw)
    def test_iid_analytic_no_integration_and_no_time(self):
        with patch('experiments.large_scale_screening_v1.data.solve_ivp',side_effect=AssertionError('IID integrated')):d=self.iid()
        np.testing.assert_array_equal(d['derivative'],rhs(d['phase'],d['net'],self.cfg))
        self.assertTrue(np.isnan(d['time']).all());self.assertEqual(len(set(d['groups'])),192)
        self.assertEqual(d['diagnostics']['trajectory_count'],0)
    def test_split_independence_and_train_count_invariance(self):
        a=self.iid();b=generate(8,110,220,self.cfg,[128,40,48])
        np.testing.assert_array_equal(a['phase'][:128],b['phase'][:128])
        self.assertEqual(validate_splits(a['groups'],a['splits']),dict(train=128,validation=32,test=32))
        self.assertEqual(len(np.unique(a['phase'],axis=0)),192)
        self.assertFalse(np.array_equal(a['phase'][:32],a['phase'][128:160]))
    def test_fixed_center_and_nested_delta_realizations(self):
        a=self.iid(delta=.2);b=self.iid(delta=.6);x,c=coordinates(a);z,c2=coordinates(b)
        np.testing.assert_array_equal(c,a['net']['center']);np.testing.assert_array_equal(c,c2)
        np.testing.assert_allclose(z,3*x,atol=2e-15)
        self.assertGreater(abs(c).max(),.1);self.assertLessEqual(abs(x).max(),.1)
    def test_network_frequency_center_independent_of_data_mode(self):
        a=self.iid();b=generate(8,110,220,self.cfg,[128,32,32],sampling_mode='trajectory',train_trajectories=4)
        for key in ['pair_edges','triple_edges','omega','center']:np.testing.assert_array_equal(a['net'][key],b['net'][key])
    def test_trajectory_matches_legacy_exactly(self):
        a=generate(8,110,220,self.cfg,[256,256,256],sampling_mode='trajectory',train_trajectories=1)
        b=old_generate(8,110,220,self.cfg,[256,256,256])
        for key in ['phase','derivative','groups','splits','time']:np.testing.assert_array_equal(a[key],b[key])
    def test_noise_unit_realizations_reused_across_delta(self):
        a=self.iid(delta=.2);b=self.iid(delta=.6)
        for d in [a,b]:
            noisy=noisy_derivative(d,.05,456,220)
            d['unit_noise']=(noisy-d['derivative'])/(.05*d['derivative'][:128].std(axis=0))
        np.testing.assert_allclose(a['unit_noise'],b['unit_noise'],atol=1e-12)
    def test_quadratic_cross_coefficient_finite_difference(self):
        d=self.iid();net=d['net'];c=net['center'];h=1e-4
        for row in triadic_coefficients(net,self.cfg)[:8]:
            i,j,k=[row[key] for key in ['target','j','k']];value=0.
            for sj in [-1,1]:
                for sk in [-1,1]:
                    t=c.copy();t[j]+=sj*h;t[k]+=sk*h;value+=sj*sk*rhs(t,net,self.cfg)[i]
            self.assertAlmostEqual(value/(4*h*h),row['quadratic_cross'],places=7)
    def test_analytic_taylor_fourth_order_remainder(self):
        d=self.iid(delta=.2);net=d['net'];dx=d['phase']-net['center']
        errors=[]
        for scale in [1.,.5,.25]:
            p=net['center']+scale*dx;errors.append(np.linalg.norm(rhs(p,net,self.cfg)-taylor_rhs(p,net,self.cfg)))
        self.assertGreater(errors[0]/errors[1],14);self.assertGreater(errors[1]/errors[2],14)
    def test_library_native_indices_and_iid_full_rank(self):
        d=self.iid();x,_=coordinates(d);p=np.array([[1,2],[3,4],[5,6]])
        terms,groups=terms_and_groups(8,0,p);indices=native_indices(terms)
        self.assertEqual(len(terms),6*8-2+12)
        self.assertEqual([tuple(sorted(v-1 for v in row if v)) for row in indices],terms)
        info,_,_=design_diagnostics(x[:128],terms);self.assertEqual(info['rank'],len(terms)-1)
        self.assertEqual(len(groups),3)
    def test_row_shuffle_preserves_spectrum_but_not_lag(self):
        t=np.linspace(0,1,1024);x=np.column_stack([np.sin(t),np.cos(t),t*t])
        p=np.random.default_rng(2).permutation(len(x));_,s=spectrum(x);_,ps=spectrum(x[p])
        np.testing.assert_allclose(s,ps,atol=1e-13,rtol=1e-12)
        self.assertGreater(lag1(x),.99);self.assertLess(abs(lag1(x[p])),.1)
    def test_invalid_modes_and_counts(self):
        for kw in [dict(sampling_mode='bogus'),dict(delta=0),dict(sampling_mode='trajectory',train_trajectories=3)]:
            with self.assertRaises(ValueError):self.iid(**kw)


if __name__=='__main__':unittest.main()
