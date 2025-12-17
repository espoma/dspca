import unittest
import os
import sys
import numpy as np
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dspca import DSPCA

class TestBackwardVariableElimination(unittest.TestCase):
    
    def setUp(self):
        # Initialize with valid sparsity_levels to avoid ValueError in __init__
        self.dspca = DSPCA(n_components=2, sparsity_levels=[2, 1], max_sensors=5)
        self.dspca._compute_max_variance = MagicMock()

    def test_basic_elimination(self):
        X = np.random.rand(100, 50)
        V = [0, 1, 2]
        current_var = 35.0
        threshold_var = 25.0
        k = 3

        # Variances when removing index 0, 1, and 2 respectively
        # We want one of them to beat threshold_var (25.0)
        self.dspca._compute_max_variance.side_effect = [10.0, 20.0, 30.0]

        # Correct argument order: X, V, k, current_variance, threshold_variance
        new_k, new_V, new_Var = self.dspca._backward_variable_elimination(X, V, k, current_var, threshold_var)

        # Should remove the feature that resulted in variance 30 (index 2)
        # because 30 > 25
        self.assertEqual(new_k, 2)
        self.assertEqual(new_V, [0, 1])
        self.assertEqual(new_Var, 30.0)

        self.assertEqual(self.dspca._compute_max_variance.call_count, 3)

    def test_elimination_with_no_features(self):
        X = np.random.rand(100, 50)
        V = []
        current_var = 5.0
        threshold_var = 4.0
        k = 0

        new_k, new_V, new_Var = self.dspca._backward_variable_elimination(X, V, k, current_var, threshold_var)

        self.assertEqual(new_k, 0)
        self.assertEqual(new_V, [])
        self.assertEqual(new_Var, 5.0)

    def test_elimination_with_one_feature(self):
        X = np.random.rand(100, 50)
        V = [0]
        current_var = 5.0
        threshold_var = 4.0
        k = 1

        new_k, new_V, new_Var = self.dspca._backward_variable_elimination(X, V, k, current_var, threshold_var)

        self.assertEqual(new_k, k)
        self.assertEqual(new_V, V)
        self.assertEqual(new_Var, 5.0)

    def test_no_elimination(self):
        X = np.random.rand(100, 50)
        V = [0, 1, 2]
        current_var = 28.0
        threshold_var = 25.0
        k = 3

        # All subsets have variance <= threshold_var
        self.dspca._compute_max_variance.side_effect = [10.0, 20.0, 25.0]

        new_k, new_V, new_Var = self.dspca._backward_variable_elimination(X, V, k, current_var, threshold_var)

        self.assertEqual(new_k, 3)
        self.assertEqual(new_V, [0, 1, 2])
        self.assertEqual(new_Var, 28.0)

    def test_real_data_integration(self):
        """Test with real data (no mocking) to ensure integration."""
        # Create new instance to avoid using the mocked method
        dspca = DSPCA(n_components=1, sparsity_levels=[2], max_sensors=5)
        
        # Create data:
        # Feature 0: High variance (Signal)
        # Feature 1: High variance (Signal)
        # Feature 2: Low variance (Noise)
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 0] *= 10  # Var ~ 100
        X[:, 1] *= 10  # Var ~ 100
        X[:, 2] *= 0.1 # Var ~ 0.01
        
        # Start with all features selected
        V = [0, 1, 2]
        k = 3
        
        # Calculate variances
        # Var([0, 1, 2]) approx 100
        current_var = dspca._compute_max_variance(X[:, V])
        
        # Threshold: Variance of a "previous" set of size 2.
        # Let's pretend the previous set was [0, 2] (Signal + Noise).
        # Var([0, 2]) approx 100 (dominated by 0).
        # But [0, 1] (Signal + Signal) should be better if they are correlated or if we look at total variance.
        # Actually, if 0 and 1 are independent, Var([0, 1]) ~ 100 (first PC aligns with one of them).
        
        # To force elimination, we need a threshold that is LOWER than the best subset variance.
        # Best subset is [0, 1]. Var([0, 1]) ~ 100.
        # Let's set threshold to 50.
        threshold_var = 50.0
        
        new_k, new_V, new_Var = dspca._backward_variable_elimination(X, V, k, current_var, threshold_var)
        
        # It should find that removing 2 gives [0, 1] with variance ~100 > 50.
        # So it should remove 2.
        
        self.assertNotIn(2, new_V)
        self.assertIn(0, new_V)
        self.assertIn(1, new_V)
        self.assertEqual(new_k, 2)

if __name__ == '__main__':
    unittest.main()