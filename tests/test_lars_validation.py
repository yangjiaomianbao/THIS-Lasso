import unittest
import numpy as np
from src.solvers.large_library import initial_ridge,validation_lasso_working_set
from src.solvers.lars_validation import _certified_path,validation_lasso_accelerated


class CertifiedLarsTests(unittest.TestCase):
    def fixture(self,n,p):
        rng=np.random.default_rng(20261022)
        X=rng.normal(size=(n,p));X-=X.mean(axis=0);X/=X.std(axis=0)
        Xv=rng.normal(size=(70,p));beta=np.zeros(p);beta[[1,8,20]]=[.8,-.5,.4]
        y=.3+X@beta+.02*rng.normal(size=n);yv=.3+Xv@beta+.02*rng.normal(size=70)
        c=dict(alpha_fractions=[1.,.1,.01,.001,.0001,1e-6,1e-8],max_iter=100000,tol=1e-10,
            kkt_tolerance=1e-6,validation_tolerance=1e-12,working_set_rounds=60,working_set_add=16)
        return X,y,Xv,yv,c

    def compare(self,n,p,adaptive):
        X,y,Xv,yv,c=self.fixture(n,p)
        a=np.abs(initial_ridge(X,y[:,None])[0])+1e-6 if adaptive else np.ones(p)
        old,old_diag=validation_lasso_working_set(X,y,Xv,yv,a,c)
        actual,diag=validation_lasso_accelerated(X,y,Xv,yv,a,c)
        self.assertAlmostEqual(diag['alpha'],old_diag['alpha'],places=12)
        np.testing.assert_allclose(actual,old,atol=3e-6,rtol=3e-6)
        self.assertLessEqual(max(row['full_library_kkt'] for row in diag['validation_path']),1e-6)

    def test_plain_overdetermined(self):self.compare(160,35,False)
    def test_adaptive_overdetermined(self):self.compare(160,35,True)
    def test_plain_underdetermined(self):self.compare(60,150,False)
    def test_adaptive_underdetermined(self):self.compare(60,150,True)

    def test_independent_weighted_KKT(self):
        X,y,Xv,yv,c=self.fixture(140,35);a=np.abs(initial_ridge(X,y[:,None])[0])+1e-6
        coef,diag=_certified_path(X,y,Xv,yv,a,c)
        g=X.T@(coef[0]+X@coef[1:]-y)/len(y);threshold=diag['alpha']/a;nz=coef[1:]!=0
        v=np.maximum(np.abs(g)-threshold,0.);v[nz]=np.abs(g[nz]+threshold[nz]*np.sign(coef[1:][nz]))
        scale=max(1.,float(np.max(np.abs(X.T@(y-y.mean())/len(y)))))
        self.assertLessEqual(float(np.max(v))/scale,c['kkt_tolerance'])


if __name__=='__main__':unittest.main()
