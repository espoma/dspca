import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dspca import DSPCA

class TestDSPCAReprStr(unittest.TestCase):
    def test_repr(self):
        dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        repr_str = repr(dspca)
        self.assertIn("DSPCA", repr_str)
        self.assertIn("n_components=2", repr_str)

    def test_str_unfitted(self):
        dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        self.assertIn("not fitted", str(dspca))

    def test_str_fitted(self):
        X = np.random.randn(10, 10)
        dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        dspca.fit(X)
        self.assertIn("fitted", str(dspca))
        self.assertIn("Explained variance", str(dspca))

if __name__ == '__main__':
    unittest.main()
