import numpy as np


class RidgeRegression:
    
    def __init__(self, bias=True, weight_l2=1e-3):
        self.bias = bias
        self.weight_l2 = weight_l2
        self.weights = None
        
    def fit(self, X, y):
        
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        n_samples, n_features = X.shape
        self.weights = np.linalg.pinv(X.T @ X + self.weight_l2 * np.eye(n_features)) @ X.T @ y
    
    def predict(self, X):
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        return X @ self.weights
    
    
class LogisticRegression:
    
    def __init__(self, lr=1e-2, bias=True, weight_l2=1e-3):
        self.lr = lr
        self.bias = bias
        self.weight_l2 = weight_l2
        self.weights = None
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y, max_iter=100):
        
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(max_iter):
            y_hat = self._sigmoid(X @ self.weights)
            self.weights -= self.lr * (self.weight_l2 * 2 * self.weights + (1 / n_samples) * X.T @ (y_hat - y))
        
    
    def predict(self, X):
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        return self._sigmoid(X @ self.weights)
    
    
class KNNRegressor:
    
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
    
    def _scale(self, X):     
        return (X - self._min) / (self._max - self._min)
    
    def _squared_distances(self, X):
        return np.sum((X[:, : None] - X[:, :, None].T)**2, axis=-1)
    
    def fit(self, X, y):
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        self.X_train = self._scale(X)
        self.y_train = y
    
    def predict(self, X):
        X = self._scale(X)
        distances_squared = -2 * self.X_train @ X.T + np.sum(X**2, axis=1) + np.sum(self.X_train**2, axis=1)[:, None]
        inds = np.argsort(distances_squared, axis=0)
        distances_squared = np.sort(distances_squared, axis=0)
        targets = np.take(self.y_train, inds[:self.n_neighbors], 0)
        
        if self.weights == 'distance':
            w = 1 / np.sqrt((distances_squared[:self.n_neighbors] + 1e-8))
            return np.average(targets, axis=0, weights=w)
        elif self.weights == 'uniform':
            return targets.mean(axis=0)
        
        