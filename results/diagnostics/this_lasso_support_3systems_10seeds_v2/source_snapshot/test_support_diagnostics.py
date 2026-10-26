"""Numerical and mapping checks for full support-path export."""
import sys
import unittest
import numpy as np
from sklearn.metrics import average_precision_score
from src.experiments.support_diagnostics import (capture_fit,validation_path,bic_path,align_bic,support_rows)
from src.solvers.lars_validation import validation_lasso_accelerated
from src.solvers.methods import fit_node


class SupportDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(24026)
        X=rng.normal(size=(120,10));means=X.mean(0);scales=X.std(0)
        self.Z=np.column_stack((np.ones(120),(X-means)/scales))
        self.Zv=np.column_stack((np.ones(60),(rng.normal(size=(60,10))-means)/scales))
        b=np.array([.4,.8,0,0,.2,0,0,0,0,0,0])
        self.y=self.Z@b+.06*rng.normal(size=120)
        self.yv=self.Zv@b+.06*rng.normal(size=60)
        self.c=dict(lasso=dict(max_iter=100000,tol=1e-10,kkt_tolerance=1e-6,
                    validation_tolerance=1e-12,alpha_fractions=[1.,.3,.1,.03,.01,.003,.001,.0003,.0001,1e-5,1e-6,1e-8],
                    working_set_rounds=60,working_set_add=64),
                    adaptive=dict(initialization='ridge',standardize=False,fit_intercept=True,ridge_rho=1.,gamma=1.,eps=1e-6,
                                  bic_noise_floor=1e-12,bic_condition_limit=1e8))

    def test_validation_capture_preserves_exact_solution(self):
        for a in [np.ones(10),np.linspace(.2,2,10)]:
            baseline,expected=validation_lasso_accelerated(self.Z[:,1:],self.y,self.Zv[:,1:],self.yv,a,self.c['lasso'])
            coef,diag,path=validation_path(self.Z,self.y,self.Zv,self.yv,a,self.c['lasso'])
            np.testing.assert_array_equal(coef,baseline)
            np.testing.assert_array_equal(path[diag['selected_path_index']],coef)
            self.assertEqual(path.shape,(12,11))
            mse=((self.Zv@path.T-self.yv[:,None])**2).mean(0)
            np.testing.assert_allclose(mse,[r['MSE'] for r in diag['validation_path']],atol=1e-14)
            self.assertLessEqual(diag['holdout_1se_index'],int(np.argmin(mse)))

    def test_bic_capture_preserves_original_selection(self):
        clean=dict(self.c,lasso={k:v for k,v in self.c['lasso'].items() if not k.startswith('working_set_')})
        for method,original in [('M1B','M1'),('M2B','M2')]:
            baseline,_=fit_node(original,self.Z,self.y,clean)
            coef,diag,path=bic_path(method,self.Z,self.y,self.c)
            np.testing.assert_array_equal(coef,baseline)
            np.testing.assert_allclose(path[diag['selected_path_index']],coef,atol=1e-12)
            aligned,base=align_bic(path,diag['alpha_path'],self.c['lasso']['alpha_fractions'])
            self.assertEqual(aligned.shape,(12,11));self.assertGreater(base,0)

    def test_capture_restores_profiler_on_failure(self):
        before=sys.getprofile()
        def fail():raise ValueError('expected')
        with self.assertRaises(ValueError):capture_fit(fail)
        self.assertIs(sys.getprofile(),before)

    def test_group_count_does_not_equate_degree_with_order(self):
        d=np.array([[0,0,0],[2,2,0],[2,3,0],[1,1,1]])
        C=np.array([[5.,1.,2.,3.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        counts,rows=support_rows(C,C,C,d,np.ones(4),np.ones(3))
        self.assertEqual(rows[0]['pair_related'],1)
        self.assertEqual(rows[0]['triple_related'],1)
        self.assertEqual(rows[0]['self_only'],1)
        self.assertEqual(rows[0]['native_readout_support'],1)
        self.assertEqual(counts['absolute_nonzero'],3)
        self.assertEqual(counts['including_bias_nonzero'],4)

    def test_native_path_AP_ties_match_sklearn(self):
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from support_analyse import per_target_curve_metrics
        # 15-variable linear and distinct quadratic features plus bias.
        d=np.array([[0,0,0]]+[[j,0,0] for j in range(1,16)]+[[j,k,0] for j in range(1,16) for k in range(j+1,16)])
        cfg={'kuramoto':{'pair_edges':[[1,2,1.]],'triple_edges':[[1,2,3,1.]]}}
        rng=np.random.default_rng(1);curves=rng.integers(0,3,size=(4,len(d))).astype(float)
        rows=per_target_curve_metrics(curves,d,1,cfg,'kuramoto')
        for row in rows:
            order=2 if row['order']=='pairwise' else 3
            ids=[i for i,r in enumerate(d) if len([v for v in r if v])==order-1 and 1 not in r]
            truth=[set(v for v in d[i] if v)==({2} if order==2 else {2,3}) for i in ids]
            self.assertAlmostEqual(row['AP'],average_precision_score(truth,np.abs(curves[row['candidate'],ids])))

    def test_resume_table_restores_types(self):
        from pathlib import Path
        import tempfile
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from support_analyse import records
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'table.csv'
            path.write_text('status,seed,noise_ratio,selected,missing\nOK,2026102201,0.05,False,\n')
            row=records(path)[0]
        self.assertEqual(row,dict(status='OK',seed=2026102201,noise_ratio=.05,selected=False,missing=None))

    def test_native_coordinates_are_headerless_and_exact(self):
        from pathlib import Path
        import tempfile
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from support_analyse import write_coordinates
        C=np.array([[.12345678901234567,0.,-2.],[0.,3.,0.]])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'coordinates.csv';write_coordinates(path,C)
            rows=np.loadtxt(path,delimiter=',',ndmin=2)
        replay=np.zeros_like(C)
        for i,j,v in rows:replay[int(i)-1,int(j)-1]=v
        np.testing.assert_array_equal(C,replay)

    def test_native_threshold_and_raw_score_semantics(self):
        from src.experiments.support_diagnostics import verify_score_exports
        # target 1 receives source 2; strict cutoff excludes equality as well.
        d=np.array([[0,0,0],[2,0,0]])
        for value in [0.,.5e-8,1e-8,2e-8]:
            C=np.zeros((2,2));C[0,1]=value
            key=(2,1,2,0)
            raw={key:value} if value else {}
            native={key:value} if value>1e-8 else {}
            count,tiny=verify_score_exports(C,d,native,raw)
            self.assertEqual(count,2)
            self.assertEqual(tiny,int(0<value<=1e-8))
            if value>0:
                with self.assertRaises(AssertionError):verify_score_exports(C,d,native,{})


if __name__=='__main__':unittest.main()
