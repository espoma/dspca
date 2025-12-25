import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dspca import DSPCA

class TestDSPCAFitTransform(unittest.TestCase):
    def test_fit_transform(self):
        X = np.random.randn(10, 10)
        dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=8)
        X_transformed = dspca.fit_transform(X)
        
        self.assertEqual(X_transformed.shape, (10, 2))
        self.assertIsNotNone(dspca.components_)
        
        # Consistency check
        dspca2 = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=8)
        dspca2.fit(X)
        X_manual = dspca2.transform(X)
        np.testing.assert_array_almost_equal(X_transformed, X_manual)

if __name__ == '__main__':
    unittest.main()
