"""Physical-node scoring regressions for multivariate dynamical systems."""
import importlib.util
from pathlib import Path
import sys
import tomllib
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('dynamics_pilot', ROOT/'scripts/dynamics_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PhysicalScoringTests(unittest.TestCase):
    def setUp(self):
        self.ids = [1,1,1,2,2,2,3,3,3]
        self.d = np.array([[0,0,0],[0,0,4],[0,4,7],[0,4,5],[0,4,4],[0,0,2]])
        self.C = np.zeros((9,6))

    def score(self, order):
        keys, scores = pilot.physical_scores(self.C,self.d,self.ids,order)
        return dict(zip(keys,scores))

    def test_max_aggregation_across_target_coordinates(self):
        self.C[0,1] = .2;self.C[1,1] = -.7
        self.assertEqual(self.score(2)[(1,2)], .7)

    def test_internal_coupling_is_not_physical_network_edge(self):
        self.C[0,5] = 10.;self.C[4,1] = 3.
        self.assertTrue(all(value==0 for value in self.score(2).values()))

    def test_distinct_variable_sources_same_physical_node_not_triple(self):
        self.C[0,3] = 4.
        self.assertTrue(all(value==0 for value in self.score(3).values()))

    def test_repeated_variable_source_keeps_native_exclusion(self):
        self.C[0,4] = 5.
        self.assertTrue(all(value==0 for value in self.score(3).values()))

    def test_valid_physical_triple_and_all_negative_candidates(self):
        self.C[1,2] = -.35
        scores = self.score(3)
        self.assertEqual(len(scores),3)
        self.assertEqual(scores[(1,2,3)],.35)
        self.assertEqual(scores[(2,1,3)],0.)

    def test_candidate_counts(self):
        self.assertEqual(len(self.score(2)),6)
        self.assertEqual(len(self.score(3)),3)

    def test_selected_topologies_include_target_faces_of_triples(self):
        c=tomllib.loads((ROOT/'configs/dynamics_pilot.toml').read_text())
        for system in c['systems']:
            pairs={(int(i),int(j)) for i,j,_ in c[system]['pair_edges']}
            for i,j,k,_ in c[system]['triple_edges']:
                self.assertIn((int(i),int(j)),pairs)
                self.assertIn((int(i),int(k)),pairs)

    def test_multitrial_initial_rng_namespaces_do_not_overlap(self):
        c=tomllib.loads((ROOT/'configs/dynamics_pilot.toml').read_text())
        nt=sum(c[k+'_trajectories'] for k in ('train','validation','test'))
        seeds=[s*10000+code*1000+t for s in c['seeds']
               for code in range(1,len(c['systems'])+1) for t in range(1,nt+1)]
        self.assertEqual(len(seeds),len(set(seeds)))


if __name__ == '__main__':unittest.main()
