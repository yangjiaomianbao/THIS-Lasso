import unittest
import numpy as np
from src.solvers.large_library import ridge_coefficients,stlsq_dual,initial_ridge,validation_lasso_working_set
from src.solvers.plain_lasso import fit_plain_lasso
from src.solvers.adaptive_lasso import fit_adaptive_lasso


class LargeLibraryTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261021)
        X=rng.normal(size=(180,35));X-=X.mean(axis=0);X/=X.std(axis=0)
        self.X=X;self.Xv=rng.normal(size=(80,35))
        beta=np.zeros(35);beta[[0,7,20]]=[.8,-1.1,.3]
        self.y=.4+X@beta+.03*rng.normal(size=180)
        self.yv=.4+self.Xv@beta+.03*rng.normal(size=80)
        self.c=dict(alpha_fractions=[1.,.1,.01,.001,.0001],max_iter=100000,tol=1e-10,
                    kkt_tolerance=1e-6,validation_tolerance=1e-12)

    def test_dual_ridge_matches_primal(self):
        X=self.X[:12];Y=self.y[:12,None]
        actual=ridge_coefficients(X,Y,1.)
        expected=np.linalg.solve(X.T@X+np.eye(X.shape[1]),X.T@Y)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_free_intercept_ridge_initializer(self):
        B=initial_ridge(self.X,self.y[:,None],1.)[0]
        expected=np.linalg.solve(self.X.T@self.X+np.eye(35),self.X.T@(self.y-self.y.mean()))
        np.testing.assert_allclose(B,expected,atol=1e-12,rtol=1e-12)

    def test_stlsq_matches_pinned_inverse_algorithm(self):
        X=np.column_stack([np.ones(len(self.X)),self.X[:,:12]]);Y=np.column_stack([self.y,2*self.y])
        C=(Y.T@X)@np.linalg.inv(X.T@X+np.eye(X.shape[1]))
        previous=1e6;k=1
        while k<10 and np.count_nonzero(np.abs(C)>1e-6)!=previous:
            k+=1;previous=np.count_nonzero(np.abs(C)>1e-6)
            small=np.abs(C)<.1;C[small]=0.
            for i in range(2):
                active=~small[i];A=X[:,active]
                C[i,active]=(Y[:,i]@A)@np.linalg.inv(A.T@A+np.eye(active.sum()))
        actual,_=stlsq_dual(X,Y)
        np.testing.assert_allclose(actual,C,atol=1e-11,rtol=1e-11)
        np.testing.assert_array_equal(actual!=0,C!=0)

    def test_working_set_plain_matches_existing_solver(self):
        X=np.column_stack([np.ones(len(self.X)),self.X]);Xv=np.column_stack([np.ones(len(self.Xv)),self.Xv])
        expected,diag=fit_plain_lasso(X,self.y,dict(self.c,selection='validation'),dict(X_validation=Xv,y_validation=self.yv))
        actual,new=validation_lasso_working_set(self.X,self.y,self.Xv,self.yv,np.ones(35),dict(self.c,working_set_add=4))
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
        self.assertAlmostEqual(new['alpha'],diag['alpha'],places=12)
        self.assertLessEqual(new['kkt_scaled_violation'],1e-6)

    def test_working_set_adaptive_matches_existing_solver(self):
        X=np.column_stack([np.ones(len(self.X)),self.X]);Xv=np.column_stack([np.ones(len(self.Xv)),self.Xv])
        expected,diag=fit_adaptive_lasso(X,self.y,dict(self.c,selection='validation',standardize=False),
            X_validation=Xv,y_validation=self.yv,train_groups=np.arange(180),validation_groups=np.arange(180,260))
        a=np.abs(initial_ridge(self.X,self.y[:,None])[0])+1e-6
        actual,new=validation_lasso_working_set(self.X,self.y,self.Xv,self.yv,a,dict(self.c,working_set_add=4))
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
        self.assertAlmostEqual(new['alpha'],diag['alpha'],places=12)
        self.assertLessEqual(new['kkt_scaled_violation'],1e-6)

    def test_invalid_ridge_or_weights_rejected(self):
        with self.assertRaises(ValueError):ridge_coefficients(self.X,self.y[:,None],0.)
        with self.assertRaises(ValueError):validation_lasso_working_set(self.X,self.y,self.Xv,self.yv,np.zeros(35),self.c)

    def test_underdetermined_full_library_matches_existing_plain_path(self):
        rng=np.random.default_rng(128)
        X=rng.normal(size=(60,150));X-=X.mean(axis=0);X/=X.std(axis=0)
        Xv=rng.normal(size=(80,150));beta=np.zeros(150);beta[[2,50,120]]=[.7,-1.,.4]
        y=.2+X@beta+.02*rng.normal(size=60);yv=.2+Xv@beta+.02*rng.normal(size=80)
        expected,diag=fit_plain_lasso(np.column_stack([np.ones(60),X]),y,dict(self.c,selection='validation'),
            dict(X_validation=np.column_stack([np.ones(80),Xv]),y_validation=yv))
        actual,new=validation_lasso_working_set(X,y,Xv,yv,np.ones(150),dict(self.c,working_set_add=8))
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
        self.assertAlmostEqual(new['alpha'],diag['alpha'],places=12)

    def test_independent_global_weighted_kkt_certificate(self):
        a=np.abs(initial_ridge(self.X,self.y[:,None])[0])+1e-6
        coef,diag=validation_lasso_working_set(self.X,self.y,self.Xv,self.yv,a,dict(self.c,working_set_add=4))
        gradient=self.X.T@(coef[0]+self.X@coef[1:]-self.y)/len(self.y)
        nz=coef[1:]!=0;threshold=diag['alpha']/a
        violation=np.maximum(np.abs(gradient)-threshold,0.)
        violation[nz]=np.abs(gradient[nz]+threshold[nz]*np.sign(coef[1:][nz]))
        scale=max(1.,np.max(np.abs(self.X.T@(self.y-self.y.mean())/len(self.y))))
        self.assertLessEqual(float(np.max(violation))/scale,1e-6)


if __name__=='__main__':unittest.main()
