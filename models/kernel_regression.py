import numpy as np


class RBFKernel:
    
    def __init__(self, gamma=1):
        self.gamma = gamma
        
    def __call__(self, X, Y):
        squarred_eucl_dist = -2 * X @ Y.T + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, None]
        return np.exp( - self.gamma * squarred_eucl_dist)
    

class PolyKernel:
    
    def __init__(self, gamma=1., constant=0., order=1):
        self.gamma = gamma
        self.constant = constant
        self.order = order
        
    def __call__(self, X, Y):
        return (self.gamma * X @ Y.T + self.constant)**self.order

    
class SigmoidKernel:
    
    def __init__(self, gamma=1., constant=0.):
        self.gamma = gamma
        self.constant = constant
        
    def __call__(self, X, Y):
        return np.tanh(self.gamma * X @ Y.T + self.constant)


class KernelRegression:
    
    def __init__(self, kernel):
        self.kernel = kernel
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        similarity = self.kernel(self.X_train, X)
        m_hat = (similarity * self.y_train[:, None]).sum(axis=0) / (similarity.sum(axis=0) + 1e-8)
        
        return m_hat
