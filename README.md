# DSPCA

DSPCA is a Python package for dimensionality reduction using the Dynamic Sparse Principal Component Analysis (DSPCA) algorithm. The package is based on the original DSPCA algorithm by [Wang et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0957417424008613).

## General Idea
The general idea is that PCA is a good way to reduce the dimensionality of a dataset, but it is not sparse. As such, every principal component is a linear combination of all features, which translates to low interpreatability of the PCs based on the original features. 
DSPCA addresses this issue by using a dynamic sparsity approach to select the most relevant features for each principal component. This allows for a more interpretable PCA, as the PCs are based on a subset of the original features.

The way DSPCA works is by fixing a budget of a maximum cumulative number of sensors M to use for **all** the principal components. Each PC will contain a decreasing number of non-zero features. This is obtained by building the PCs iteratively, first adding one feature at a time using a greedy algorithm that looks for the feature that maximizes the explained variance (Forward Variable Selection, FVS), then using Backward Variable Elimination (BVE), that removes features one by one (till a minimum of two features for a given PC) to check if the explained variance increases, which means that the system was in a local minima. BVE is helpful to avoid nesting effects due to the greedy algorithm and explores the space of possible solutions, allowing to find a near-optimal minimum. This process is repeated for each principal component. 

