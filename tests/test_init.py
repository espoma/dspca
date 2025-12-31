import unittest
import sys
import os

# Add the parent directory to sys.path to allow importing dspca if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dspca import DSPCA

class TestDSPCAInit(unittest.TestCase):
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

    def test_init_max_sensors_none(self):
        """Test that initializing DSPCA with max_sensors=None works correctly."""
        dspca = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=None)
        self.assertEqual(dspca.n_components, 2)
        self.assertEqual(dspca.sparsity_levels, [10, 5])
        self.assertIsNone(dspca.max_sensors)

if __name__ == '__main__':
    unittest.main()
