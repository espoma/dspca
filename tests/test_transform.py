import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dspca import DSPCA

class TestDSPCATransform(unittest.TestCase):
    """Test the transform method of the DSPCA class."""
    def setUp(self):
        """Set up a fitted DSPCA instance."""
        self.X = np.random.randn(10, 10)
        self.dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=8)
        self.dspca.fit(self.X)

    def test_transform_shape(self):
        """Test that the transformed data has the correct shape."""
        X_transformed = self.dspca.transform(self.X)
        self.assertEqual(X_transformed.shape, (10, 2))

    def test_transform_errors(self):
        """Test that transform raises appropriate errors for unfitted models or invalid input shapes."""
        unfitted = DSPCA(n_components=1, sparsity_levels=[1], max_sensors=1)
        with self.assertRaises(ValueError):
            unfitted.transform(self.X)
        
        X_wrong = np.random.randn(10, 5)
        with self.assertRaises(ValueError):
            self.dspca.transform(X_wrong)

if __name__ == '__main__':
    unittest.main()
