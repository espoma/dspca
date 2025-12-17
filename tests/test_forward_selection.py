import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dspca import DSPCA

class TestForwardVariableSelection(unittest.TestCase):
    
    def setUp(self):
        self.dspca = DSPCA(n_components=2, sparsity_levels=[2, 1], max_sensors=5)
        # Mock _compute_max_variance to avoid actual PCA computation and control results
        self.dspca._compute_max_variance = MagicMock()

    def test_basic_selection(self):
        """Test that the feature with the highest variance is selected."""
        X = np.zeros((10, 5)) # Dummy data, won't be used for calculation due to mock
        V = []
        candidates = [0, 1, 2]
        k = 0
        
        # Setup mock returns: feature 0 -> 10, feature 1 -> 20, feature 2 -> 5
        # The method calls _compute_max_variance for V + [candidate]
        # So it will be called with subsets.
        # We can side_effect based on the shape or content of the argument, 
        # but simpler is to just return values in order if we know the order of iteration.
        # Iteration order is candidates list: 0, 1, 2.
        self.dspca._compute_max_variance.side_effect = [10.0, 20.0, 5.0]
        
        new_k, new_V, max_var = self.dspca._forward_variable_selection(X, V, candidates, k)
        
        self.assertEqual(new_k, 1)
        self.assertEqual(new_V, [1]) # Should select index 1 (variance 20)
        self.assertEqual(max_var, 20.0)
        
        # Verify calls
        self.assertEqual(self.dspca._compute_max_variance.call_count, 3)

    def test_selection_with_existing_V(self):
        """Test selection when V is not empty."""
        X = np.zeros((10, 5))
        V = [3]
        candidates = [0, 1]
        k = 1
        
        # Candidates: 0, 1.
        # Calls for: [3, 0], [3, 1]
        self.dspca._compute_max_variance.side_effect = [15.0, 12.0]
        
        new_k, new_V, max_var = self.dspca._forward_variable_selection(X, V, candidates, k)
        
        self.assertEqual(new_k, 2)
        self.assertEqual(new_V, [3, 0]) # Should select index 0
        self.assertEqual(max_var, 15.0)

    def test_empty_candidates(self):
        """Test that empty candidates list raises ValueError."""
        X = np.zeros((10, 5))
        V = []
        candidates = []
        k = 0
        
        with self.assertRaises(ValueError):
            self.dspca._forward_variable_selection(X, V, candidates, k)

    def test_tuple_return_from_variance(self):
        """Test handling when _compute_max_variance returns a tuple (variance, weights)."""
        X = np.zeros((10, 5))
        V = []
        candidates = [0, 1]
        k = 0
        
        # Mock returns: (10.0, weights), (20.0, weights)
        self.dspca._compute_max_variance.side_effect = [
            (10.0, np.array([1])), 
            (20.0, np.array([1]))
        ]
        
        new_k, new_V, max_var = self.dspca._forward_variable_selection(X, V, candidates, k)
        
        self.assertEqual(new_k, 1)
        self.assertEqual(new_V, [1])
        self.assertEqual(max_var, (20.0, np.array([1])))

    def test_real_data_integration(self):
        """Test with real data (no mocking) to ensure integration."""
        # Create new instance to avoid using the mocked method
        dspca = DSPCA(n_components=1, sparsity_levels=[1], max_sensors=5)
        
        # Create data where feature 1 has much higher variance than feature 0
        # Feature 0: variance ~ 1
        # Feature 1: variance ~ 100
        np.random.seed(42)
        X = np.random.randn(100, 2)
        X[:, 1] *= 10 
        
        V = []
        candidates = [0, 1]
        k = 0
        
        new_k, new_V, max_var = dspca._forward_variable_selection(X, V, candidates, k)
        
        self.assertEqual(new_V, [1])
        # Variance of feature 1 should be around 100
        self.assertGreater(max_var, 50) 

if __name__ == '__main__':
    unittest.main()
