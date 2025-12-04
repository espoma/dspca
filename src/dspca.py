import numpy as np
from sklearn.decomposition import PCA


class DSPCA:
    """
    Dynamic Sparse PCA (DSPCA) using Forward Variable Selection (FVS)
    and Backward Variable Elimination (BVE).

    This implementation enforces a nested sparsity structure where the support
    (non-zero features) of the j-th principal component is a subset of the support
    of the (j-1)-th principal component.

    The FVS is used to select the features with maximum variance for each principal component.
    The BVE is used to ensure a larger parameter space is sweeped in looking for the next principal component.
    """

    def __init__(self, n_components=2, sparsity_levels=None):
        self.n_components = n_components
        self.sparsity_levels = sparsity_levels
        self.components_ = None
        self.explained_variance_ = None

    def _compute_max_variance(self, X_subset):
        """
        Computes the maximum variance (largest eigenvalue of covariance matrix)
        for the given data subset.
        """
        if X_subset.shape[1] == 0:
            return 0.0

        # For a single component, PCA finds the direction of max variance.
        # We use sklearn's PCA for robustness.
        if X_subset.shape[1] == 1:
            return np.var(X_subset)

        pca = PCA(n_components=1)
        pca.fit(X_subset)
        return pca.explained_variance_[0]


    def _forward_variable_selection(self, V, candidates, k):

        _V = []

        for variable in candidates:
            l = V + [variable]
            max_var = self._compute_max_variance(l)
            _V.append(max_var)

        index = np.argmax(_V)
        V.append(candidates[index])
        k += 1

        return k, V

    def _backward_variable_elimination(self, V, candidates, k, Var):
        
        _k = k
        while _k > 2:
            _V = []
            for variable in V:
                l = V - [variable]
                max_var = self._compute_max_variance(l)
                _V.append(max_var)
            
            index = np.argmax(_V)
            if max(_V) > Var:
                V.remove(V[index])
                _k -= 1

        return _k, V

    def fit(self, X, q, sparsity_levels):

        X = np.array(X)
        _, n_features = X.shape

        k = np.zeros(q)

        if isinstance(sparsity_levels, list) and all(isinstance(n, int) for n in sparsity_levels):
            K = sparsity_levels
        elif isinstance(sparsity_levels, list) and all(isinstance(x, float) for x in sparsity_levels)::
            K[0] = sparsity_levels[0] * n_features
            for i in range(1, len(sparsity_levels)):
                K[i] = sparsity_levels[i] * K[i-1]

        for j in range(q):
            V = []
            L = set(x for i in range(j) for x in V[i])
            while k[j] < K[j]:
                








    # def _forward_variable_selection(self, X_curr, candidates, k):
    #     """
    #     Select k features from candidates using Forward Variable Selection.
    #     Starts with empty set and greedily adds feature that maximizes variance.
    #     """
    #     selected = []
    #     candidates_list = list(candidates)

    #     # Cap k if larger than available candidates
    #     k = min(k, len(candidates_list))

    #     for _ in range(k):
    #         best_feature = -1
    #         best_variance = -1.0

    #         # Try adding each candidate not yet selected
    #         for feature in candidates_list:
    #             if feature in selected:
    #                 continue

    #             current_subset = selected + [feature]
    #             X_subset = X_curr[:, current_subset]
    #             variance = self._compute_max_variance(X_subset)

    #             if variance > best_variance:
    #                 best_variance = variance
    #                 best_feature = feature

    #         if best_feature != -1:
    #             selected.append(best_feature)
    #         else:
    #             break

    #     return np.array(selected)

    # def _backward_variable_elimination(self, X_curr, candidates, k):
    #     """
    #     Select k features from candidates using Backward Variable Elimination.
    #     Starts with all candidates and greedily removes feature that minimizes variance loss.
    #     """
    #     current_selection = list(candidates)
    #     num_to_remove = len(candidates) - k

    #     if num_to_remove < 0:  # Should not happen if k <= len(candidates)
    #         return np.array(candidates)

    #     for _ in range(num_to_remove):
    #         best_feature_to_remove = -1
    #         best_variance = -1.0

    #         # Try removing each feature from current selection
    #         for feature in current_selection:
    #             temp_selection = [f for f in current_selection if f != feature]

    #             if not temp_selection:
    #                 variance = 0.0
    #             else:
    #                 X_subset = X_curr[:, temp_selection]
    #                 variance = self._compute_max_variance(X_subset)

    #             # Remove the feature that contributes least
    #             if variance > best_variance:
    #                 best_variance = variance
    #                 best_feature_to_remove = feature

    #         if best_feature_to_remove != -1:
    #             current_selection.remove(best_feature_to_remove)
    #         else:
    #             break

    #     return np.array(current_selection)

    def fit(self, X):
        """
        Fit the model to X.
        """
        X = np.array(X)
        _, n_features = X.shape

        # Default sparsity levels if not provided
        if self.sparsity_levels is None:
            self.sparsity_levels = [
                max(1, n_features - i * (n_features // (self.n_components + 1)))
                for i in range(self.n_components)
            ]

        # Validate sparsity levels
        for i in range(1, len(self.sparsity_levels)):
            if self.sparsity_levels[i] > self.sparsity_levels[i - 1]:
                raise ValueError("Sparsity levels must be non-increasing for DSPCA.")

        self.components_ = np.zeros((self.n_components, n_features))
        self.explained_variance_ = []

        # Center the data
        X_centered = X - np.mean(X, axis=0)
        X_curr = X_centered.copy()

        # Initial active set (all features)
        current_active_indices = np.arange(n_features)

        for i in range(self.n_components):
            k = self.sparsity_levels[i]

            # --- Feature Selection Step (FVS / BVE) ---
            n_candidates = len(current_active_indices)
            if k <= n_candidates / 2:
                selected_indices = self._forward_variable_selection(
                    X_curr, current_active_indices, k
                )
            else:
                selected_indices = self._backward_variable_elimination(
                    X_curr, current_active_indices, k
                )

            # Update active indices for the NEXT iteration (Nested Constraint)
            current_active_indices = selected_indices

            # --- Component Construction ---
            X_selected = X_curr[:, selected_indices]
            pca_final = PCA(n_components=1)
            pca_final.fit(X_selected)

            comp_vector = np.zeros(n_features)
            comp_vector[selected_indices] = pca_final.components_[0]
            self.components_[i, :] = comp_vector

            # --- Deflation ---
            projections = X_curr @ comp_vector
            X_curr = X_curr - np.outer(projections, comp_vector)

            self.explained_variance_.append(np.var(projections))

        return self