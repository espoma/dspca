import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dspca import DSPCA

class TestDSPCAReprStr(unittest.TestCase):
    """Test the __repr__ and __str__ methods of the DSPCA class."""
    def test_repr(self):
        """Test the string representation (repr) of the DSPCA object."""
        dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        repr_str = repr(dspca)
        self.assertIn("DSPCA", repr_str)
        self.assertIn("n_components=2", repr_str)

    def test_str_unfitted(self):
        """Test the user-friendly string representation (str) before fitting."""
        dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        self.assertIn("not fitted", str(dspca))

    def test_str_fitted(self):
        """Test the user-friendly string representation (str) after fitting."""
        X = np.random.randn(10, 10)
        dspca = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        dspca.fit(X)
        self.assertIn("fitted", str(dspca))
        self.assertIn("Explained variance", str(dspca))

if __name__ == '__main__':
    unittest.main()
