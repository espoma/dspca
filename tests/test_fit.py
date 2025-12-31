import unittest
import numpy as np
import pandas as pd
import sys
import os
import warnings

# Add the parent directory to sys.path to allow importing dspca if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dspca import DSPCA

class TestDSPCAFit(unittest.TestCase):
    """Test the fit method of the DSPCA class."""

    def setUp(self):
        """Set up a basic DSPCA instance and synthetic data."""
        self.n_components = 2
        self.sparsity_levels = [5, 3]
        self.max_sensors = 10
        self.dspca = DSPCA(
            n_components=self.n_components,
            sparsity_levels=self.sparsity_levels,
            max_sensors=self.max_sensors
        )
        # 10 samples, 20 features
        self.X = np.random.randn(10, 20)

    def test_fit_max_sensors_none_vs_fixed(self):
        """Test that max_sensors=None allows more features than a fixed max_sensors."""
        X = np.random.randn(10, 20)
        
        # Case 1: max_sensors=5
        dspca_fixed = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=5)
        dspca_fixed.fit(X)
        all_used_fixed = set().union(*dspca_fixed.components_)
        self.assertLessEqual(len(all_used_fixed), 5)
        
        # Case 2: max_sensors=None
        # Sparsity levels [10, 5] would normally use 10-15 features
        dspca_none = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=None)
        dspca_none.fit(X)
        all_used_none = set().union(*dspca_none.components_)
        self.assertGreaterEqual(len(all_used_none), 10)

    def test_fit_valid_numpy(self):
        """Test fit with valid numpy array."""
        self.dspca.fit(self.X)
        self.assertIsNotNone(self.dspca.components_)
        self.assertEqual(len(self.dspca.components_), self.n_components)
        self.assertIsNotNone(self.dspca.explained_variance_)
        self.assertEqual(len(self.dspca.explained_variance_), self.n_components)
        self.assertIsNotNone(self.dspca.total_variance_)

    def test_fit_valid_pandas(self):
        """Test fit with valid pandas DataFrame."""
        df = pd.DataFrame(
            self.X, 
            columns=[f"col_{i}" for i in range(self.X.shape[1])]
        )
        self.dspca.fit(df)
        self.assertIsNotNone(self.dspca.feature_names_)
        self.assertEqual(self.dspca.feature_names_[0], "col_0")
        # Check that get_feature_names works
        names = self.dspca.get_feature_names(0)
        self.assertTrue(all(name.startswith("col_") for name in names))

    def test_fit_invalid_type(self):
        """Test fit raises TypeError for invalid input type."""
        with self.assertRaises(TypeError):
            self.dspca.fit("not an array")

    def test_fit_invalid_dimensions(self):
        """Test fit raises ValueError for non-2D input."""
        # 1D array
        with self.assertRaises(ValueError):
            self.dspca.fit(np.array([1, 2, 3]))
        # 3D array
        with self.assertRaises(ValueError):
            self.dspca.fit(np.random.randn(2, 2, 2))

    def test_fit_insufficient_samples(self):
        """Test fit raises ValueError for fewer than 2 samples."""
        X_small = np.random.randn(1, 10)
        with self.assertRaises(ValueError) as cm:
            self.dspca.fit(X_small)
        self.assertIn("n_samples must be at least 2", str(cm.exception))

    def test_fit_insufficient_features(self):
        """Test fit raises ValueError when n_features < n_components."""
        # Model needs 2 components, but data only has 1 feature
        X_narrow = np.random.randn(10, 1)
        with self.assertRaises(ValueError) as cm:
            self.dspca.fit(X_narrow)
        self.assertIn("n_features (1) must be >= n_components (2)", str(cm.exception))

    def test_fit_sparsity_exceeds_features(self):
        """Test fit raises ValueError when sparsity level > n_features."""
        # Sparsity level is [5, 3], but data only has 4 features
        X_small_features = np.random.randn(10, 4)
        with self.assertRaises(ValueError) as cm:
            self.dspca.fit(X_small_features)
        self.assertIn("cannot exceed n_features", str(cm.exception))

    def test_fit_sparsity_warning(self):
        """Test fit issues a warning for already sparse data."""
        # Create a very sparse matrix (mostly zeros)
        # 200 elements, only 2 non-zero -> 99% sparse
        X_sparse = np.zeros((10, 20))
        X_sparse[0, 0] = 1.0
        X_sparse[1, 1] = 1.0
        
        with self.assertWarns(UserWarning) as cm:
            self.dspca.fit(X_sparse)
        self.assertIn("already sparse", str(cm.warning))

    def test_fit_float_sparsity_conversion(self):
        """Test fit correctly converts float sparsity levels."""
        # max_sensors = 10
        # 0.8 * 10 = 8
        # 0.5 * 8 = 4
        dspca_float = DSPCA(n_components=2, sparsity_levels=[0.8, 0.5], max_sensors=10)
        dspca_float.fit(self.X)
        self.assertEqual(len(dspca_float.components_[0]), 8)
        self.assertEqual(len(dspca_float.components_[1]), 4)

    def test_fit_max_sensors_constraint(self):
        """Test that the total number of unique sensors used does not exceed max_sensors."""
        X = np.random.randn(10, 10)
        # Component 0 will take 4 features.
        # Component 1 will then see len(total_used_sensors) == 4.
        # If max_sensors is 4, it MUST pick from those 4.
        dspca_constrained = DSPCA(n_components=2, sparsity_levels=[4, 2], max_sensors=4)
        dspca_constrained.fit(X)
        
        all_features = set()
        for comp in dspca_constrained.components_:
            all_features.update(comp)
        
        self.assertLessEqual(len(all_features), 4)

    def test_fit_deterministic(self):
        """Test that fitting twice on the same data yields the same result."""
        X = np.random.RandomState(42).randn(10, 20)
        dspca1 = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        dspca2 = DSPCA(n_components=2, sparsity_levels=[5, 3], max_sensors=10)
        
        dspca1.fit(X)
        dspca2.fit(X)
        
        self.assertEqual(dspca1.components_, dspca2.components_)
        np.testing.assert_array_almost_equal(
            dspca1.explained_variance_, 
            dspca2.explained_variance_
        )

    def test_fit_max_sensors_warning(self):
        """Test fit issues a warning when max_sensors < n_components."""
        # n_components=3, max_sensors=2
        # Sparsity levels must be strictly decreasing: [3, 2, 1]
        dspca = DSPCA(n_components=3, sparsity_levels=[3, 2, 1], max_sensors=3)
        # Manually trigger the warning condition by setting max_sensors low
        dspca.max_sensors = 2 
        with self.assertWarns(UserWarning) as cm:
            dspca.fit(self.X)
        self.assertIn("max_sensors (2) is less than n_components (3)", str(cm.warning))

    def test_fit_max_sensors_none(self):
        """Test that fit works correctly when max_sensors is None."""
        # With max_sensors=None, all features are available
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=None)
        dspca.fit(self.X)
        
        self.assertEqual(len(dspca.components_), 2)
        self.assertEqual(len(dspca.components_[0]), 10)
        self.assertEqual(len(dspca.components_[1]), 5)
        # Check that all selected features are within range
        for component in dspca.components_:
            for idx in component:
                self.assertTrue(0 <= idx < self.X.shape[1])

    def test_fit_max_sensors_none_proportions(self):
        """Test that proportions work correctly when max_sensors is None."""
        # 0.5 of 20 features = 10 features for first component
        # 0.5 of 10 features = 5 features for second component
        dspca = DSPCA(n_components=2, sparsity_levels=[0.5, 0.5], max_sensors=None)
        dspca.fit(self.X)
        
        self.assertEqual(len(dspca.components_[0]), 10)
        self.assertEqual(len(dspca.components_[1]), 5)

if __name__ == '__main__':
    unittest.main()
