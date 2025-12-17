import unittest
import numpy as np
import sys
import os
from sklearn.decomposition import PCA

# Add the parent directory to sys.path to allow importing dspca if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dspca import DSPCA

class TestDSPCAVariance(unittest.TestCase):
    """Test the variance computation methods of the DSPCA class."""

    def test_total_variance_basic_correctness(self):
        """Test that _total_variance returns the correct total variance."""
        X = np.array([[1, 2], [3, 4]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 2.0)

    def test_total_variance_zero_variance(self):
        """Test that _total_variance returns 0 when the input data has zero variance."""
        X = np.array([[1, 1], [1, 1]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 0)

    def test_total_variance_axis_verification(self):
        """Test that _total_variance calculates the variance across the right axis"""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 6.75)
        
    def test_total_variance_single_feature(self):
        """Test that _total_variance returns the correct total variance in case the dataframe only contains a single feature"""
        X = np.array([[1], [2], [3], [4]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, np.var(X))

    def test_total_variance_single_row(self):
        """Test that _total_variance returns zero if the input dataframe only contains one row"""
        X = np.array([[1, 2, 3, 4]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 0)

    def test_total_variance_categorical_input(self):
        """Test that _total_variance raises TypeError for categorical input"""
        X = np.array([['a', 'b'], ['c', 'd']])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(TypeError):
            dspca._total_variance(X)

    def test_total_variance_dataframe_with_nans(self):
        """Test that _total_variance raises ValueError for dataframe with NaNs"""
        X = np.array([[1, 2, 3, 4], [5, np.nan, 7, 8]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(ValueError):
            dspca._total_variance(X)

    def test_total_variance_dataframe_with_infs(self):
        """Test that _total_variance raises ValueError for dataframe with infinities"""
        X = np.array([[1, 2, 3, 4], [5, np.inf, 7, 8]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(ValueError):
            dspca._total_variance(X)

    def test_compute_max_variance_single_feature(self):
        """Test _compute_max_variance with a single feature."""
        X = np.array([[1], [2], [3], [4]])
        dspca = DSPCA(n_components=1, sparsity_levels=[1], max_sensors=10)
        var = dspca._compute_max_variance(X)
        self.assertIsInstance(var, float)
        self.assertAlmostEqual(var, np.var(X))

    def test_compute_max_variance_multiple_features(self):
        """Test _compute_max_variance with multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        dspca = DSPCA(n_components=1, sparsity_levels=[1], max_sensors=10)
        result = dspca._compute_max_variance(X)
        
        # Should return tuple (variance, weights)
        self.assertIsInstance(result, tuple)
        var, weights = result
        
        # Check variance against sklearn PCA
        pca = PCA(n_components=1)
        pca.fit(X)
        expected_var = pca.explained_variance_[0]
        
        self.assertAlmostEqual(var, expected_var)
        
        # Check weights properties
        self.assertEqual(len(weights), 2)
        # Weights might be flipped (sign ambiguity), but norm is 1
        self.assertAlmostEqual(np.linalg.norm(weights), 1.0)

    def test_compute_max_variance_empty(self):
        """Test _compute_max_variance with empty feature set raises ValueError (via _validate_data)."""
        X = np.zeros((5, 0))
        dspca = DSPCA(n_components=1, sparsity_levels=[1], max_sensors=10)
        with self.assertRaises(ValueError):
            dspca._compute_max_variance(X)

if __name__ == '__main__':
    unittest.main()
