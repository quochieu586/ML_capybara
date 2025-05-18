import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk

class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.001, n_iters=5000, class_weight=None):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.class_weight = class_weight or {1: 1.0, -1: 1.0}
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                weight = self.class_weight[y[idx]]
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - weight * np.dot(x_i, y[idx])
                    db = -weight * y[idx]
                self.w -= self.lr * dw
                self.b -= self.lr * db
    
    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))