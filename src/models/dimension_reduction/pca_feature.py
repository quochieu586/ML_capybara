import numpy as np
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd

class PCA:
    def __init__(self):
        self.W = None
        self.X_center = None

    def fit_by_fix_features(self, X: NDArray[np.float64], top_k: int) -> NDArray[np.float64]:
        """
            Parameters:
            - A: The maxtrix of feature vectors of all samples.
            - top_k: The size of single feature vector after reducing.
        """
        # Step 1: Scaled into mean = 0
        self.X_center = np.mean(X, axis=0)
        X = X - self.X_center

        # Step 2: Compute U and V to get variance
        _, S, VT = randomized_svd(X, n_components=top_k)
        var = S ** 2

        # Step 4: Get top_k component in V with highest variance
        S_full = svd(X, compute_uv=False)
        full_var = S_full ** 2
        print(f"Amount of variance after reducing: ", var.sum() / full_var.sum())

        # Step 5: Save the transformer matrix
        self.W = VT[:top_k, :].T

    def fit_by_amount_of_variance(self, X: NDArray[np.float64], threshold: float = 0.75) -> NDArray[np.float64]:
        """
            Parameters:
            - A: The maxtrix of feature vectors of all samples.
            - thresold: The minimum amount of variance to be kept in the trained features.
        """
        # Step 1: Scaled into mean = 0
        self.X_center = np.mean(X, axis=0)
        X = X - self.X_center

        # Step 2: Compute full variance matrix to get variance
        S_full = svd(X, compute_uv=False)
        full_var = (S_full ** 2)
        full_var = full_var / full_var.sum()

        # Step 3: Get the minimum value of component that give variance greater than threshold
        kept_var = 0
        top_k = 0
        
        while top_k < full_var.shape[0]:
            kept_var += full_var[top_k]
            top_k += 1
            if kept_var > threshold:
                break

            # if top_k % 40 == 0:
            #     print(f"Top {top_k}: remain variance - {kept_var}")
        
        # Step 4: Get the variance matrix of X (eigenvectors of X^T @ X)
        _, S, VT = randomized_svd(X, n_components=top_k)
        print(f"PCA keep {top_k} components correponding to keep {kept_var} of total variance")
        
        # Step 5: Save the result
        self.W = VT[:top_k].T

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = X - self.X_center
        return X @ self.W