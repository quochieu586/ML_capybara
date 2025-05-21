from scipy import sparse
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

class MulticlassGradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=1.0, max_depth=1, random_state=None):
        self.n_estimators   = n_estimators
        self.learning_rate  = learning_rate
        self.max_depth      = max_depth
        self.random_state   = random_state

    def _one_hot(self, y):
        classes = np.unique(y)
        y_ohe = np.zeros((len(y), len(classes)))
        for i, cls in enumerate(classes):
            y_ohe[:, i] = (y == cls).astype(float)
        return y_ohe, np.sort(classes)

    def _softmax(self, F):
        ex = np.exp(F - F.max(axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)


    def fit(self, X, y, X_val=None, y_val=None):
        if sparse.issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)
        y = np.asarray(y)

        if X_val is not None:
            if sparse.issparse(X_val):
                X_val = X_val.toarray()
            else:
                X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)


        y_ohe, classes = self._one_hot(y)
        self.classes_ = classes          
        n, K = y_ohe.shape


        F = np.zeros((n, K))              # training logits
        if X_val is not None:
            F_val = np.zeros((X_val.shape[0], K))


        self.trees        = [[None]*K for _ in range(self.n_estimators)]
        self.leaf_values  = [[{} for _ in range(K)] for _ in range(self.n_estimators)]
        self.train_score_ = []
        self.val_score_   = []

        # =================== Boosting ====================================
        for m in range(self.n_estimators):

            P = self._softmax(F)          # (n, K)
            R = y_ohe - P                 # residual

            for k in range(K):
                # Residual of class k
                res_k = R[:, k]              # (n,)

                # Use regression because we are working with rediual not labels
                tree = DecisionTreeRegressor(
                    max_depth     = self.max_depth,
                    random_state  = self.random_state
                )
                tree.fit(X, res_k) 
                self.trees[m][k] = tree

                leaf_ids      = tree.apply(X) # (n,)
                if X_val is not None:
                    leaf_ids_val = tree.apply(X_val)

                for leaf in np.unique(leaf_ids):
                    idx = leaf_ids == leaf  # Create a mask to find the residual rows corresponding the the leaf
                    v   = res_k[idx].mean() * self.learning_rate
                    self.leaf_values[m][k][leaf] = v

                    # Nudging the prediction
                    F[idx, k] += v
                    if X_val is not None:
                        idx_val = leaf_ids_val == leaf
                        F_val[idx_val, k] += v

            # track accuracy
            y_train_pred = self.classes_[np.argmax(self._softmax(F), axis=1)]
            self.train_score_.append((y_train_pred == y).mean())

            if X_val is not None:
                y_val_pred = self.classes_[np.argmax(self._softmax(F_val), axis=1)]
                self.val_score_.append((y_val_pred == y_val).mean())


    def predict_proba(self, X):
        # Sometimes saving spaces in one place means a couple extra lines of code in
        if sparse.issparse(X):
            X = X.toarray()
        else:
            X = np.array(X)
        n = X.shape[0]
        F = np.zeros((n, len(self.classes_)))
        for m in range(self.n_estimators):
            for k in range(len(self.classes_)):
                tree = self.trees[m][k]
                vals = self.leaf_values[m][k] #  dict {leaf_ID : v}
                leaf_ids = tree.apply(X) # (n, )
                for leaf, v in vals.items():
                    F[leaf_ids==leaf, k] += v
        return self._softmax(F)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
