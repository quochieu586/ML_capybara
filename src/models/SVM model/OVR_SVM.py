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