from LinearSVM import LinearSVM
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score

class OvRLinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.001, n_iters=5000, class_weights=None):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.class_weights = class_weights
        self.classifiers = {}
        self.classes = None
        self.score_stats = {}
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            weights = {1: self.class_weights.get(cls, 1.0), -1: 1.0}
            svm = LinearSVM(self.lr, self.lambda_param, self.n_iters, weights)
            svm.fit(X, y_binary)
            self.classifiers[cls] = svm
            scores = svm.decision_function(X)
            self.score_stats[cls] = (np.mean(scores), np.std(scores) + 1e-8)
    
    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            scores[:, idx] = self.classifiers[cls].decision_function(X)
            mean, std = self.score_stats[cls]
            scores[:, idx] = (scores[:, idx] - mean) / std
        return self.classes[np.argmax(scores, axis=1)]
    
    def decision_function(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            scores[:, idx] = self.classifiers[cls].decision_function(X)
            mean, std = self.score_stats[cls]
            scores[:, idx] = (scores[:, idx] - mean) / std
        scores = 1 / (1 + np.exp(-scores))
        return scores