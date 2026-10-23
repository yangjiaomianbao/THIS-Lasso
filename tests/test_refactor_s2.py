"""Behavioral invariants for S2; frozen source and original algorithms stay unchanged."""
import ast
from dataclasses import asdict
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from src.solvers import AdaptiveLassoConfig, fit_adaptive_lasso
from src.solvers.adaptive_lasso import _prepare, _select, _validation

ROOT=Path(__file__).resolve().parents[1]


class SolverExtractionTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(20261019)
        X=np.column_stack([np.ones(320),rng.normal(size=(320,6))])
        y=X@np.array([.3,1.2,-.4,0.,.15,0.,0.])+rng.normal(0.,.05,320)
        return X,y

    def zero_exponent(self,selection):
        from src.solvers.plain_lasso import fit_plain_lasso,fit_plain_bic
        X,y=self.fixture();Xv,yv=X[240:],y[240:];X,y=X[:240],y[:240]
        c=AdaptiveLassoConfig(gamma=0.,standardize=False,selection=selection,alpha=.002)
        p=_prepare(X,y,c)
        initial=np.linspace(-.7,.9,len(p['active']))
        a=(np.abs(initial)+c.eps)**c.gamma;weights=1./a
        np.testing.assert_array_equal(weights,np.ones_like(weights))
        U=p['z']*(p['scales'][p['active']]*a)
        validation=None
        if selection=='validation':
            validation=_validation(Xv,yv,X,p,np.arange(240),np.arange(240,320),c)
        theta,alpha,diag=_select(X,p,U,a,c,validation)
        coef=np.zeros(X.shape[1]);coef[p['active']]=a*theta
        coef[0]=p['response_mean']-p['means'][p['active']]@coef[p['active']]
        if selection=='bic':other,ordinary=fit_plain_bic(X,y,asdict(c))
        else:
            options=dict(selection=selection,alpha=.002)
            kwargs=dict(X_validation=Xv,y_validation=yv) if selection=='validation' else None
            other,ordinary=fit_plain_lasso(X,y,options,kwargs)
        np.testing.assert_array_equal(coef,other)
        self.assertEqual(alpha,ordinary['alpha'])
        if selection=='bic':
            np.testing.assert_array_equal(diag['alpha_path'],ordinary['alpha_path'])
            np.testing.assert_array_equal(diag['bic_path'],ordinary['bic_path'])
        if selection=='validation':
            self.assertEqual(diag['selected_path_index'],ordinary['selected_path_index'])
            self.assertEqual([r['alpha'] for r in diag['validation_path']],
                             [r['alpha'] for r in ordinary['validation_path']])
            # The frozen paths evaluate predictions in different floating-point orders.
            # Compare ordinary-path MSEs to that same frozen path, not to a different
            # formula's bit pattern. Coefficients, alpha path and selected index above
            # must still be exactly equal across the gamma=0 and ordinary problems.
            with tarfile.open(ROOT/'docs/refactor/S1_source_snapshot.tar.gz','r:gz') as frozen:
                reference={'__name__':'frozen_r3_bridge'}
                exec(compile(frozen.extractfile('src/data_bridge.py').read(),
                             'S1_snapshot/src/data_bridge.py','exec'),reference)
            old_coef,old=reference['fit_plain_lasso'](X,y,dict(selection=selection,alpha=.002),kwargs)
            np.testing.assert_array_equal(coef,old_coef)
            np.testing.assert_array_equal([r['MSE'] for r in ordinary['validation_path']],
                                          [r['MSE'] for r in old['validation_path']])
            self.assertEqual(float(np.mean((Xv@coef-yv)**2)),old['validation_path'][old['selected_path_index']]['MSE'])

    def test_zero_exponent_fixed_is_ordinary_lasso(self):self.zero_exponent('fixed')
    def test_zero_exponent_bic_path_is_ordinary_lasso(self):self.zero_exponent('bic')
    def test_zero_exponent_validation_path_is_ordinary_lasso(self):self.zero_exponent('validation')

    def test_public_gamma_validation_is_preserved(self):
        X,y=self.fixture()
        with self.assertRaisesRegex(ValueError,'gamma'):
            fit_adaptive_lasso(X,y,dict(gamma=0.,selection='fixed',alpha=.01))

    def test_legacy_imports_use_the_same_functions(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        from src import data_bridge
        from src.bridge.numeric_io import read_matrix,write_toml
        from src.solvers.plain_lasso import fit_plain_lasso,fit_plain_bic
        from src.solvers.methods import fit_node
        import r4_methods
        self.assertIs(data_bridge.fit_plain_lasso,fit_plain_lasso)
        self.assertIs(data_bridge.read_matrix,read_matrix)
        self.assertIs(data_bridge.write_toml,write_toml)
        self.assertIs(r4_methods.fit_plain_bic,fit_plain_bic)
        self.assertIs(r4_methods.fit_node,fit_node)

    def test_extracted_functions_match_frozen_ast(self):
        mapping={
            'src/bridge/numeric_io.py':{'src/data_bridge.py':['file_hash','_toml','write_toml','read_matrix'],
                                      'scripts/r4_experiment.py':['stamp','write_rows']},
            'src/solvers/plain_lasso.py':{'src/data_bridge.py':['fit_plain_lasso'],'scripts/r4_methods.py':['fit_plain_bic']},
            'src/solvers/methods.py':{'scripts/r4_methods.py':['fit_node']},
            'src/data/contracts.py':{'scripts/r4_metrics.py':['validate_splits'],
                                     'scripts/r4_experiment.py':['load_dataset']},
            'src/preprocessing/taylor.py':{'scripts/r4_metrics.py':['restore_coefficients']},
            'src/metrics/structure.py':{'scripts/r4_metrics.py':['binary_metrics','edge_scores']},
        }
        def functions(text):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        with tarfile.open(ROOT/'docs/refactor/S1_source_snapshot.tar.gz','r:gz') as frozen:
            for current,sources in mapping.items():
                new=functions((ROOT/current).read_text())
                for old,names in sources.items():
                    reference=functions(frozen.extractfile(old).read().decode())
                    for name in names:self.assertEqual(new[name],reference[name],(current,name))


class DataAndScoreExtractionTests(unittest.TestCase):
    def test_legacy_metric_and_data_exports_are_identical(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        import r4_metrics,r4_experiment
        from src.data.contracts import validate_splits,load_dataset
        from src.preprocessing.taylor import restore_coefficients
        from src.metrics.structure import binary_metrics,edge_scores
        from src.bridge.numeric_io import stamp,write_rows
        for old,new in [(r4_metrics.validate_splits,validate_splits),(r4_experiment.load_dataset,load_dataset),
                        (r4_metrics.restore_coefficients,restore_coefficients),(r4_metrics.binary_metrics,binary_metrics),
                        (r4_metrics.edge_scores,edge_scores),(r4_experiment.stamp,stamp),(r4_experiment.write_rows,write_rows)]:
            self.assertIs(old,new)

    def test_full_taylor_contract_and_whole_trajectory_splits(self):
        from src.data.contracts import load_dataset,validate_splits
        data=load_dataset(ROOT/'results/r5/e1_full_independent/clean',22261035)
        self.assertEqual(data['theta'].shape,(165,480))
        self.assertEqual(data['Y'].shape,(8,480));self.assertEqual(data['d'].shape,(165,3))
        np.testing.assert_array_equal(data['theta'][0],np.ones(480))
        self.assertEqual(validate_splits(data['groups'],data['splits']),dict(train=100,validation=30,test=30))
        with self.assertRaisesRegex(ValueError,'crosses'):
            validate_splits([1,1,2],['train','validation','test'])

    def test_one_target_exclusions_max_aggregation_and_boundary(self):
        from src.metrics.structure import edge_scores,binary_metrics
        d=np.array([[0,0,0],[0,0,1],[0,1,2],[0,1,1],[0,2,1]])
        C=np.zeros((4,5));C[2,2]=-.2;C[2,4]=.7;C[2,3]=9.;C[0,2]=8.;C[3,1]=1e-8
        keys,scores=edge_scores(C,d,3)
        self.assertEqual(dict(zip(keys,scores))[(3,1,2)],.7)
        self.assertEqual(np.count_nonzero(scores),1)
        pk,ps=edge_scores(C,d,2)
        boundary=dict(zip(pk,ps))[(4,1)]
        m=binary_metrics([True],[boundary]);self.assertEqual((m['TP'],m['FN']),(0,1))

    def test_prediction_restore_and_free_bias(self):
        from src.preprocessing.taylor import restore_coefficients
        rng=np.random.default_rng(19)
        raw=np.column_stack([np.ones(120),rng.normal(size=(120,4))*[.01,100.,3.,8.]+[7.,-3.,20.,1.]])
        mu=raw[:80].mean(axis=0);mu[0]=0.
        s=np.sqrt(np.mean((raw[:80]-mu)**2,axis=0));s[0]=1.
        z=(raw-mu)/s;C=rng.normal(size=(3,5));restored=restore_coefficients(C,mu,s)
        np.testing.assert_allclose(restored@raw.T,C@z.T,atol=1e-11,rtol=1e-11)


if __name__=='__main__':unittest.main()
