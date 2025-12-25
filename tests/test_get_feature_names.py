import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dspca import DSPCA

class TestDSPCAGetFeatureNames(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(10, 5)
        self.feature_names = [f"f_{i}" for i in range(5)]
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.dspca = DSPCA(n_components=2, sparsity_levels=[3, 2], max_sensors=4)
        self.dspca.fit(self.df)

    def test_get_feature_names_all(self):
        all_names = self.dspca.get_feature_names()
        self.assertEqual(len(all_names), 2)
        self.assertEqual(len(all_names[0]), 3)
        self.assertEqual(len(all_names[1]), 2)

    def test_get_feature_names_indexed(self):
        names_0 = self.dspca.get_feature_names(0)
        self.assertEqual(len(names_0), 3)

    def test_get_feature_names_errors(self):
        unfitted = DSPCA(n_components=1, sparsity_levels=[1], max_sensors=1)
        with self.assertRaises(ValueError):
            unfitted.get_feature_names()
        with self.assertRaises(ValueError):
            self.dspca.get_feature_names(99)

if __name__ == '__main__':
    unittest.main()
