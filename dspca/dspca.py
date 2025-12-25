"""
Dynamic Sparse Principal Component Analysis (DSPCA) implementation.

This module provides a sparse PCA implementation using Forward Variable Selection (FVS)
and Backward Variable Elimination (BVE) with nested sparsity constraints.
"""

from typing import Optional, Union, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class DSPCA:
    """
    Dynamic Sparse Principal Component Analysis (DSPCA).
    
    DSPCA performs dimensionality reduction while enforcing sparsity constraints
    on the principal components. It uses Forward Variable Selection (FVS) and
    Backward Variable Elimination (BVE) to select features that maximize variance
    while maintaining a nested sparsity structure.
    
    The nested sparsity structure ensures that the support (non-zero features) of
    the j-th principal component is a subset of the support of the (j-1)-th
    principal component.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to compute. Must be a positive integer.
        
    sparsity_levels : list of int or list of float, optional
        Sparsity levels for each component. Can be specified as:
        - List of integers: Exact number of features for each component
        - List of floats: Proportions (0 < x <= 1) relative to previous component
        - Must be strictly decreasing and match n_components in length.
        If None, must be set before calling fit().
        
    max_sensors : int
        Maximum number of sensors (features) to use to build all the new principal components. Must be a positive integer.

    Attributes
    ----------
    components_ : list of list of int
        List of feature indices for each principal component.
        
    explained_variance_ : list of float
        Variance explained by each principal component.
        
    feature_names_ : np.ndarray
        Names of the features in the input data.

    total_variance_ : float
        Total variance of the input data.
        
    Examples
    --------
    >>> import numpy as np
    >>> from dspca import DSPCA
    >>> X = np.random.randn(100, 50)
    >>> dspca = DSPCA(n_components=3, sparsity_levels=[20, 15, 10], max_sensors=30)
    >>> dspca.fit(X)
    >>> print(f"Explained variance: {dspca.explained_variance_}")
    
    Notes
    -----
    The algorithm iteratively selects features using:
    1. Forward Variable Selection (FVS): Greedily adds features that maximize variance
    2. Backward Variable Elimination (BVE): Removes features whose absence increases variance
    The process continues until the desired sparsity levels are met for each component.
    
    References
    ----------
    .. [1] Wang, Tianhui, et al. "Dynamic sparse PCA: a dimensional reduction method for sensor data in virtual metrology." Expert Systems with Applications 251 (2024): 123995.
    """
    
    def __init__(
        self,
        n_components: int,
        sparsity_levels: Optional[Union[List[int], List[float]]],
        max_sensors: int 
    ) -> None:
        """
        Initialize the DSPCA model.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of principal components to compute.
            
        sparsity_levels : list of int or list of float, optional
            Sparsity levels for each component.
            
        max_sensors : int
            Maximum number of sensors to use.
            
        Raises
        ------
        ValueError
            If n_components is not a positive integer.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError(
                f"n_components must be a positive integer, got {n_components}"
            )

        if not isinstance(sparsity_levels, list):
            raise ValueError(
                "sparsity_levels must be a list of integers or floats"
            )

        if not (all(isinstance(sparsity_level, int) for sparsity_level in sparsity_levels) or all(isinstance(sparsity_level, float) for sparsity_level in sparsity_levels)):
            raise ValueError(
                "sparsity_levels must be a either a list of all integers or a list of all floats"
            )

        if not isinstance(max_sensors, int) or max_sensors <= 0:
            raise ValueError(
                f"max_sensors must be a positive integer, got {max_sensors}"
            )

        if any(sparsity_levels[i] <= 0 for i in range(len(sparsity_levels))):
            raise ValueError(
                "sparsity_levels must be a list of positive integers or floats"
            )

        if (any(sparsity_level > 1 and not isinstance(sparsity_level, int) for sparsity_level in sparsity_levels)):
            raise ValueError(
                "sparsity_levels must be a list of positive integers or floats between 0 and 1"
            )

        if len(sparsity_levels) != n_components:
            raise ValueError(
                "sparsity_levels must have the same length as n_components"
            )

        if isinstance(sparsity_levels, list) and all(isinstance(sparsity_level, int) for sparsity_level in sparsity_levels):
            for n in range(1, len(sparsity_levels)):
                if (sparsity_levels[n] >= sparsity_levels[n-1]):
                    raise ValueError(
                        "Sparsity levels as integers should be strictly decreasing with the components"
                    )

        if (max(sparsity_levels) > max_sensors):
            raise ValueError(
                f"Sparsity levels for any given component cannot include a number of features larger than max_sensors: {max_sensors}"
            )

        self.n_components = n_components
        self.sparsity_levels = sparsity_levels
        self.max_sensors = max_sensors
        
        # Attributes set during fit
        self.components_: Optional[List[List[int]]] = None
        self.weights_: Optional[List[np.ndarray]] = None
        self.explained_variance_: Optional[List[float]] = None
        self.explained_variance_ratio_: Optional[List[float]] = None
        self.feature_names_: Optional[np.ndarray] = None
        self.total_variance_: Optional[float] = None

    def _validate_data(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Centralized data validation.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data.
            
        Returns
        -------
        X_array : np.ndarray
            Validated and converted numpy array.
            
        Raises
        ------
        TypeError
            If X is not numeric or not array-like.
        ValueError
            If X is empty, contains NaNs/Infs, or is not 2D.
        """
        # Type Check & Conversion
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                f"X must be a numpy array or pandas DataFrame, got {type(X)}"
            )
        
        X_array = np.array(X)

        # Empty Check
        if X_array.size == 0:
            raise ValueError("Input data is empty")

        # Numeric Type Check
        if not np.issubdtype(X_array.dtype, np.number):
            raise TypeError("Input data must be numeric")

        # Finite Check (NaN/Inf)
        if not np.isfinite(X_array).all():
            raise ValueError("Input contains NaN or infinite values")
            
        # Dimension Check
        if X_array.ndim != 2:
             raise ValueError(f"X must be a 2D array, got shape {X_array.shape}")

        return X_array

    def _total_variance(self, X: np.ndarray) -> float:
        """
        Compute the total variance of the data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data matrix.
            
        Returns
        -------
        total_variance : float
            Total variance of the data.
        """
        X = self._validate_data(X)
        return np.var(X, axis=0).sum()  

    def _compute_max_variance(
        self,
        X_subset: np.ndarray,
        return_components: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute the maximum variance for a data subset.
        
        Computes the largest eigenvalue of the covariance matrix, which
        corresponds to the maximum variance along any direction.
        
        Parameters
        ----------
        X_subset : np.ndarray of shape (n_samples, n_features_subset)
            Subset of the data matrix.
            
        return_components : bool, default=False
            If True, also return the principal component weights.
            
        Returns
        -------
        variance : float or tuple of (float, np.ndarray)
            If return_components is False: Returns variance as float.
            If return_components is True: Returns (variance, weights).
            
        Raises
        ------
        ValueError
            If X_subset is empty.
        RuntimeError
            If PCA computation fails.
        """
        X_subset = self._validate_data(X_subset)
        
        if X_subset.shape[1] == 0:
            return 0.0 if not return_components else (0.0, np.array([]))
        
        # For single feature
        if X_subset.shape[1] == 1:
            var = float(np.var(X_subset))
            if return_components:
                return var, np.array([1.0])
            return var
        
        # For multiple features, use PCA to find max variance direction
        try:
            pca = PCA(n_components=1)
            pca.fit(X_subset)
            var = float(pca.explained_variance_[0])
            
            if return_components:
                return var, pca.components_[0]
            return var
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute PCA on subset: {str(e)}"
            ) from e

    def _forward_variable_selection(
        self,
        X: np.ndarray,
        V: List[int],
        candidates: List[int],
        k: int
    ) -> Tuple[int, List[int], float]:
        """
        Perform Forward Variable Selection (FVS).
        
        Greedily selects the next feature from candidates that maximizes
        the variance when added to the current feature set V.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data matrix.
            
        V : list of int
            Current list of selected feature indices.
            
        candidates : list of int
            List of candidate feature indices to consider.
            
        k : int
            Current number of selected features.
            
        Returns
        -------
        k : int
            Updated number of selected features (k + 1).
            
        V : list of int
            Updated list of selected feature indices.
            
        max_var : float
            Maximum variance achieved with the selected feature.
            
        Raises
        ------
        ValueError
            If candidates list is empty.
        """
        if not candidates:
            raise ValueError(
                "Cannot perform forward variable selection with empty candidates list"
            )
        
        variances = []
        
        # Evaluate variance for each candidate
        for variable in candidates:
            feature_subset = V + [variable]
            try:
                variance = self._compute_max_variance(X[:, feature_subset], return_components=False)
                variances.append(variance)
            except Exception as e:
                raise RuntimeError(
                    f"Error computing variance for feature {variable}: {str(e)}"
                ) from e
        
        # Select feature with maximum variance
        best_idx = int(np.argmax(variances))
        max_var = variances[best_idx]
        V.append(candidates[best_idx])
        k += 1
        
        return k, V, max_var

    def _backward_variable_elimination(
        self,
        X: np.ndarray,
        V: List[int],
        k: int,
        current_variance: float,
        threshold_variance: float
    ) -> Tuple[int, List[int], float]:
        """
        Perform Backward Variable Elimination (BVE).
        
        Checks if removing any feature from V (resulting in a subset of size k-1)
        yields a variance higher than the threshold_variance.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        V : list of int
            Current list of selected feature indices (size k).
        k : int
            Current number of selected features.
        current_variance : float
            Variance of the current set V.
        threshold_variance : float
            Variance to beat (typically variance of the set before FVS added a feature).
            
        Returns
        -------
        k : int
            Updated number of selected features.
        V : list of int
            Updated list of selected feature indices.
        Var : float
            Updated variance.
        """
        # Need at least 2 features to perform elimination
        if k < 2:
            return k, V, current_variance
        
        variances = []
        
        # Try removing each feature
        for variable in V:
            feature_subset = [v for v in V if v != variable]
            
            try:
                variance = self._compute_max_variance(X[:, feature_subset], return_components=False)
                variances.append(variance)
            except Exception as e:
                import warnings
                warnings.warn(f"Error computing variance: {str(e)}")
                continue
        
        if not variances:
            return k, V, current_variance
        
        max_variance = max(variances)
        best_idx = int(np.argmax(variances))
        
        # If we found a subset of size k-1 that is better than the previous set of size k-1
        if max_variance > threshold_variance:
            V.pop(best_idx)
            return k - 1, V, max_variance
            
        # No improvement, keep current set
        return k, V, current_variance

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'DSPCA':
        """
        Fit the DSPCA model to the data.
        """
        # ... (validation code omitted for brevity, assumes it exists upstream) ...
        # Validate data
        X_array = self._validate_data(X)

        # Check for sparsity
        sparsity = 1.0 - (np.count_nonzero(X_array) / X_array.size)
        if sparsity > 0.5:
            import warnings
            warnings.warn(
                f"Input data is already sparse (sparsity: {sparsity:.2f}). "
                "DSPCA might not be necessary or could behave unexpectedly.",
                UserWarning
            )
        
        # Extract feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = np.array(X.columns)
        else:
            self.feature_names_ = np.array(
                [f'feature_{i}' for i in range(X_array.shape[1])]
            )

        self.total_variance_ = self._total_variance(X_array)
        
        n_samples, n_features = X_array.shape
        
        if n_samples < 2:
            raise ValueError(f"n_samples must be at least 2, got {n_samples}")
            
        if n_features < self.n_components:
            raise ValueError(f"n_features ({n_features}) must be >= n_components ({self.n_components})")
        
        # Process sparsity levels
        K = np.zeros(self.n_components, dtype=int)
        
        if all(isinstance(n, int) for n in self.sparsity_levels):
            K = np.array(self.sparsity_levels, dtype=int)
            for i, k_val in enumerate(K):
                if k_val > n_features:
                    raise ValueError(f"Sparsity level at index {i} ({k_val}) cannot exceed n_features ({n_features})")
                    
        elif all(isinstance(x, (int, float)) for x in self.sparsity_levels):
            K[0] = int(self.sparsity_levels[0] * self.max_sensors)
            for i in range(1, len(self.sparsity_levels)):
                K[i] = int(self.sparsity_levels[i] * K[i-1])
            K = np.maximum(K, 1)
        else:
            # This should theoretically not be reached due to __init__ checks
            raise TypeError("sparsity_levels must contain all integers or all floats")
        
        for i in range(1, len(K)):
            if K[i] > K[i-1]:
                raise ValueError(f"Sparsity levels must be non-increasing. Component {i}: {K[i]}, Component {i-1}: {K[i-1]}")
        
        # Initialize
        X_curr = X_array.copy()
        available_sensors = np.arange(n_features)
        k = np.zeros(self.n_components)
        
        self.components_ = []
        self.weights_ = []
        self.explained_variance_ = []
        self.explained_variance_ratio_ = []
        
        # Fit each component
        for j in range(self.n_components):
            V: List[int] = []
            
            # Initial variance for empty set is 0
            Var = 0.0

            total_used_sensors = list(set().union(*self.components_[:j]))     
            
            # Select features for this component
            while k[j] < K[j]:
                # Determine candidate features
                if len(total_used_sensors) < self.max_sensors:
                    candidate_variables_idx = list(set(available_sensors) - set(V))
                else:
                    candidate_variables_idx = list(set(total_used_sensors) - set(V))
                
                if not candidate_variables_idx:
                    import warnings
                    warnings.warn(
                        f"No more candidate features available for component {j}. "
                        f"Stopping with {len(V)} features instead of {K[j]}."
                    )
                    break
                
                # Store variance before adding feature (threshold for BVE)
                threshold_var = Var
                
                # Forward selection
                try:
                    k[j], V, Var_new = self._forward_variable_selection(
                        X_curr, V, candidate_variables_idx, k[j]
                    )
                except Exception as e:
                    raise RuntimeError(f"Error in forward variable selection: {str(e)}") from e
                
                # Backward elimination
                # Compare subsets of size k (after removal) with threshold_var (variance of previous set of size k)
                try:
                    k[j], V, Var = self._backward_variable_elimination(
                        X_curr, V, k[j], Var_new, threshold_var
                    )
                except Exception as e:
                    raise RuntimeError(f"Error in backward variable elimination: {str(e)}") from e
            
            # Store component
            self.components_.append(V)
            
            # Compute final variance and weights for this component
            try:
                Var, weights = self._compute_max_variance(X_curr[:, V], return_components=True)
                weights = weights / np.linalg.norm(weights)
                    
                self.weights_.append(weights)
                self.explained_variance_.append(float(Var))
                self.explained_variance_ratio_.append(float(Var / self.total_variance_))
                
                projection = np.dot(X_curr[:, V], weights)
                X_curr[:, V] = X_curr[:, V] - np.outer(projection, weights)
                
            except Exception as e:
                raise RuntimeError(f"Error computing variance/deflation: {str(e)}") from e
        
        return self

    def get_feature_names(
        self,
        component_idx: Optional[int] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get feature names for the specified component(s).
        
        Parameters
        ----------
        component_idx : int, optional
            Index of the component. If None, returns feature names for all components.
            
        Returns
        -------
        feature_names : np.ndarray or list of np.ndarray
            Feature names for the specified component(s).
            
        Raises
        ------
        ValueError
            If the model has not been fitted or component_idx is out of range.
        """
        if self.components_ is None or self.feature_names_ is None:
            raise ValueError(
                "Model has not been fitted yet. Call fit() before get_feature_names()."
            )
        
        if component_idx is not None:
            if not isinstance(component_idx, int):
                raise TypeError(
                    f"component_idx must be an integer, got {type(component_idx)}"
                )
                
            if component_idx < 0 or component_idx >= len(self.components_):
                raise ValueError(
                    f"component_idx must be in [0, {len(self.components_)-1}], "
                    f"got {component_idx}"
                )
            
            feature_indices = self.components_[component_idx]
            return self.feature_names_[feature_indices]
        
        # Return feature names for all components
        return [
            self.feature_names_[component]
            for component in self.components_
        ]
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data to the sparse principal component space.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data.
            
        Raises
        ------
        ValueError
            If the model has not been fitted, X has wrong number of features,
            or X contains NaNs/Infs.
        TypeError
            If X is not a numpy array or pandas DataFrame.
        """
        if self.components_ is None:
            raise ValueError(
                "Model has not been fitted yet. Call fit() before transform()."
            )
        
        X_array = self._validate_data(X)
        
        if X_array.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"X has {X_array.shape[1]} features, but model was fitted with "
                f"{len(self.feature_names_)} features"
            )
        
        X_transformed = np.zeros((X_array.shape[0], self.n_components))
        
        for i, component_indices in enumerate(self.components_):
            X_subset = X_array[:, component_indices]
            
            # Project onto principal direction
            if X_subset.shape[1] > 1:
                pca = PCA(n_components=1)
                X_transformed[:, i] = pca.fit_transform(X_subset).ravel()
            else:
                X_transformed[:, i] = X_subset.ravel()
        
        return X_transformed
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data.
            
        Raises
        ------
        ValueError
            If X has invalid shape, contains NaN/Inf, or sparsity_levels
            is not properly configured.
        TypeError
            If X is not a numpy array or pandas DataFrame.
        """
        return self.fit(X).transform(X)
    
    def __repr__(self) -> str:
        """Return string representation of the DSPCA object."""
        return (
            f"DSPCA(n_components={self.n_components}, "
            f"sparsity_levels={self.sparsity_levels})"
        )
    
    def __str__(self) -> str:
        """Return user-friendly string representation."""
        if self.components_ is None:
            return f"DSPCA(n_components={self.n_components}, not fitted)"
        
        return (
            f"DSPCA(n_components={self.n_components}, fitted) \n"
            f"Sparsity per component: {self.sparsity_levels} \n"
            f"Explained variance: {self.explained_variance_} \n"
            f"Explained variance ratio: {self.explained_variance_ratio_}"
        )



