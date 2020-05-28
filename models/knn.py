import numpy as np


class KNNBase:
    
    def __init__(self, n_neighbors, weights, scale):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.scale = scale
    
    def _scale(self, X):     
        return (X - self._min) / (self._max - self._min)
    
    def fit(self, X, y):
        if self.scale:
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)
            self.X_train = self._scale(X)
        else:
            self.X_train = X
        
        self.y_train = y
    
    def predict(self, X):
        if self.scale:
            X = self._scale(X)
        
        distances_squared = -2 * self.X_train @ X.T + np.sum(X**2, axis=1) + np.sum(self.X_train**2, axis=1)[:, None]
        inds = np.argsort(distances_squared, axis=0)
        distances_squared = np.sort(distances_squared, axis=0)
        targets = np.take(self.y_train, inds[:self.n_neighbors], 0)
        
        return distances_squared, targets
    

class KNNRegressor(KNNBase):
    
    def __init__(self, n_neighbors=5, weights='uniform', scale=True):
        super().__init__(n_neighbors=n_neighbors, weights=weights, scale=scale)
    
    def predict(self, X):
        distances_squared, targets = super().predict(X)
        
        if self.weights == 'distance':
            w = 1 / np.sqrt((distances_squared[:self.n_neighbors] + 1e-8))
            return np.average(targets, axis=0, weights=w)
        elif self.weights == 'uniform':
            return targets.mean(axis=0)
        
        
class KNNClassifier(KNNBase):
    
    def __init__(self, n_neighbors=5, weights='uniform', scale=True):
        super().__init__(n_neighbors=n_neighbors, weights=weights, scale=scale)
    
    def predict(self, X):
        distances_squared, targets = super().predict(X)
        
        if self.weights == 'distance':
            w = 1 / np.sqrt((distances_squared[:self.n_neighbors] + 1e-8))
            predictions = np.average(targets, axis=0, weights=w)
        
        return np.array([np.argmax(np.bincount(y)) for y in targets.T])