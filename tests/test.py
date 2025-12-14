import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow importing dspca if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dspca import DSPCA

class TestDSPCA(unittest.TestCase):

    """Test the initialization of the DSPCA class."""
    def test_init_negative_n_components(self):
        """Test that initializing DSPCA with negative n_components raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=-5, sparsity_levels=[5], max_sensors=10)

    def test_init_negative_sparsity_levels(self):
        """Test that initializing DSPCA with negative sparsity_levels raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=5, sparsity_levels=[-5], max_sensors=10)

    def test_init_sparsity_levels_not_list(self):
        """Test that initializing DSPCA with non-list sparsity_levels raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=5, sparsity_levels="not a list", max_sensors=10)

    def test_init_sparsity_levels_mixed_types(self):
        """Test that initializing DSPCA with mixed types of sparsity_levels raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=3, sparsity_levels=[0.2, 4, 9], max_sensors=20)

    def test_init_sparsity_levels_different_size_n_components(self):
        """Test that initializing DSPCA with different size of sparsity_levels and n_components raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=3, sparsity_levels=[10, 2], max_sensors=20)

    def test_init_sparsity_levels_not_decreasing_int(self):
        """Test that initializing DSPCA with non-decreasing sparsity_levels raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=3, sparsity_levels=[3, 4, 2], max_sensors=20)

    def test_init_negative_max_sensors(self):
        """Test that initializing DSPCA with negative max_sensors raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=3, sparsity_levels=[10, 5, 3], max_sensors=-1)

    def test_init_non_integer_max_sensors(self):
        """Test that initializing DSPCA with non-integer max_sensors raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=3, sparsity_levels=[10, 5, 3], max_sensors=12.2)

    def test_init_sparsity_levels_larger_max_sensors(self):
        """Test that initializing DSPCA with sparsity_levels larger than max_sensors raises ValueError."""
        with self.assertRaises(ValueError):
            DSPCA(n_components=3, sparsity_levels=[10, 5, 3], max_sensors=8)
            

    def test_init_valid_inputs_int_sparsity_levels(self):
        """Test that initializing DSPCA with valid inputs works correctly."""
        # Test with integer sparsity levels
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        self.assertEqual(dspca.n_components, 2)
        self.assertEqual(dspca.sparsity_levels, [10, 5])
        self.assertEqual(dspca.max_sensors, 20)

    def test_init_valid_inputs_float_sparsity_levels(self):
        """Test that initializing DSPCA with valid inputs works correctly."""
        # Test with float sparsity levels
        dspca_float = DSPCA(n_components=2, sparsity_levels=[0.5, 0.2], max_sensors=20)
        self.assertEqual(dspca_float.n_components, 2)
        self.assertEqual(dspca_float.sparsity_levels, [0.5, 0.2])
        self.assertEqual(dspca_float.max_sensors, 20)

    """Test the _validate_data method"""
    def test_validate_data_valid(self):
        """Test that _validate_data returns the array for valid input."""
        X = np.array([[1, 2], [3, 4]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        X_valid = dspca._validate_data(X)
        np.testing.assert_array_equal(X, X_valid)

    def test_validate_data_invalid_type(self):
        """Test that _validate_data raises TypeError for invalid input type."""
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(TypeError):
            dspca._validate_data("not an array")

    def test_validate_data_empty(self):
        """Test that _validate_data raises ValueError for empty input."""
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(ValueError):
            dspca._validate_data(np.array([]))

    def test_validate_data_non_numeric(self):
        """Test that _validate_data raises TypeError for non-numeric input."""
        X = np.array([['a', 'b'], ['c', 'd']])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(TypeError):
            dspca._validate_data(X)

    def test_validate_data_nan(self):
        """Test that _validate_data raises ValueError for input with NaNs."""
        X = np.array([[1, 2], [3, np.nan]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(ValueError):
            dspca._validate_data(X)

    def test_validate_data_inf(self):
        """Test that _validate_data raises ValueError for input with Infs."""
        X = np.array([[1, 2], [3, np.inf]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(ValueError):
            dspca._validate_data(X)

    def test_validate_data_wrong_dim(self):
        """Test that _validate_data raises ValueError for non-2D input."""
        X = np.array([1, 2, 3])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        with self.assertRaises(ValueError):
            dspca._validate_data(X)

    """Test the _total_variance method"""
    def test_basic_correctness(self):
        """Test that _total_variance returns the correct total variance."""
        X = np.array([[1, 2], [3, 4]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 2.0)

    def test_zero_variance(self):
        """Test that _total_variance returns 0 when the input data has zero variance."""
        X = np.array([[1, 1], [1, 1]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 0)

    def test_axis_verification(self):
        """Test that _total_variance calculates the variance across the right axis"""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 6.75)
        
    def test_single_feature(self):
        """Test that _total_variance returns the correct total variance in case the dataframe only contains a single feature"""
        X = np.array([[1], [2], [3], [4]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, np.var(X))

    def test_single_row(self):
        """Test that _total_variance returns zero if the input dataframe only contains one row"""
        X = np.array([[1, 2, 3, 4]])
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=20)
        total_variance = dspca._total_variance(X)
        self.assertEqual(total_variance, 0)

if __name__ == '__main__':
    unittest.main()
