import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow importing dspca if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dspca import DSPCA

class TestDSPCAValidation(unittest.TestCase):
    """Test the _validate_data method of the DSPCA class."""

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

if __name__ == '__main__':
    unittest.main()
