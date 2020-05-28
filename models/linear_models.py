import numpy as np


class RidgeRegression:
    
    def __init__(self, bias=True, weight_l2=1e-3, scale=True):
        self.bias = bias
        self.weight_l2 = weight_l2
        self.weights = None
        self.scale = scale
        
    def _scale(self, X):     
        return (X - self._min) / (self._max - self._min)
    
    def fit(self, X, y):
        if self.scale:
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)
            X = self._scale(X)
        
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        n_samples, n_features = X.shape
        self.weights = np.linalg.pinv(X.T @ X + self.weight_l2 * np.eye(n_features)) @ X.T @ y
    
    def predict(self, X):
        if self.scale:
            X = self._scale(X)
        
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