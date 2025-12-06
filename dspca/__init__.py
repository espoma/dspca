"""
DSPCA - Dynamic Sparse Principal Component Analysis

A Python implementation of sparse PCA with nested sparsity constraints.
"""

from .dspca import DSPCA
from .version import __version__

__all__ = ['DSPCA']

# Optional: Package-level configuration
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)