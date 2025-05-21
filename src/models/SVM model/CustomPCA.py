import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk


class CustomPCA:
    def __init__(self, n_components=None, variance_ratio=None):
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None

    def fit(self, X):
        """Fit PCA on the training data."""
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store explained variance
        self.explained_variance_ = eigenvalues
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance

        # Determine number of components
        if self.variance_ratio is not None:
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.argmax(cumulative_variance >= self.variance_ratio) + 1
        else:
            self.n_components_ = self.n_components or X.shape[1]

        # Select top components
        self.components_ = eigenvectors[:, :self.n_components_]

        return self

    def transform(self, X):
        """Transform data using the fitted PCA."""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        """Fit PCA and transform the data."""
        return self.fit(X).transform(X)