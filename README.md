# DSPCA

DSPCA is a Python package for dimensionality reduction using the Dynamic Sparse Principal Component Analysis (DSPCA) algorithm. The package is based on the original DSPCA algorithm by [Wang et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0957417424008613).

## Background and Motivation
PCA is a well-oiled machine for dimensionality reduction, but it is not sparse. As such, every principal component is a linear combination of all features, which translates to low interpretability of the PCs based on the original features.
DSPCA addresses this issue by using a dynamic sparsity approach to select the most relevant features for each principal component. This allows for a more interpretable PCA, as the PCs are based on a subset of the original features. 

DSPCA is particularly helpful when dimensionality reduction needs to be performed on a large number of features as it is often the case with sensor data, and it can be paired with a feature selection method to further improve the interpretability of the PCs.

## Algorithm
DSPCA fixes a budget $M$ of a maximum cumulative number of sensors to use for **all** the principal components, and a maximum number of components to calculate, $q$. Each PC will contain a decreasing number of non-zero features $ K_j $, $ K_1 \geq K_2 \geq \ldots \geq K_q $. This is obtained by building the PCs iteratively, first adding one feature at a time using a greedy algorithm that, given a set of candidate variables CV, selects the feature that maximizes the explained variance (Forward Variable Selection, FVS). CV at step $ j $ is chosen either as the set of all the features, or as the set of features selected for the previous PCs, depending on whether the number of non-zero features used in all the previous components, that is $ \mathcal{L}_{j-1} \leq M $. 

Then, using Backward Variable Elimination (BVE), features are removed one by one (till a minimum of two features for a given PC) to check if the explained variance increases, which means that the system was previously in a local minima. BVE is helpful to avoid nesting effects due to the greedy algorithm and explores the space of possible solutions, allowing to find a near-optimal minimum. Importantly, because of BVE, the principal components found by DSPCA will **not be linearly independent**, unlike PCA or sparse PCA.
 
## Installation

### For Users
You can install the package directly from GitHub using pip:

```bash
pip install git+https://github.com/espoma/dspca.git
```

### For Developers
If you want to contribute or modify the code:

1. Clone the repository:
```bash
git clone https://github.com/espoma/dspca.git
cd dspca
```

2. Install in editable mode:
```bash
pip install -e .
```

## Usage

Here is a simple example of how to use DSPCA:

```python
import numpy as np
from dspca import DSPCA

# Generate dummy data
X = np.random.rand(100, 50)

# Initialize DSPCA
# n_components: number of PCs to compute
# sparsity_levels: number of features to keep for each PC (must be decreasing)
# max_sensors: maximum total features to use across all components (optional, default=None)
model = DSPCA(n_components=2, sparsity_levels=[10, 5], max_sensors=None)

# Fit the model
model.fit(X)

# Transform data
X_transformed = model.transform(X)

# Access results
print(f"Explained variance: {model.explained_variance_}")
print(f"Selected features for PC1: {model.components_[0]}")
```

## Roadmap

Future releases will focus on the following improvements:

- **Visualization Tools**: Add built-in plotting utilities for explained variance and feature selection paths.
- **Scikit-learn Compatibility**: Ensure full compatibility with `Pipeline` and `GridSearchCV`.
- **Performance Optimization**: Further optimize Forward Variable Selection (FVS) and Backward Variable Elimination (BVE) for very large datasets (e.g. Bayesian Optimization, Genetic Algorithm).