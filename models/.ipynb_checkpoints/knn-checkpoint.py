import numpy as np


class KNNRegressor:
    
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
    
    def _scale(self, X):     
        return (X - self._min) / (self._max - self._min)
    
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