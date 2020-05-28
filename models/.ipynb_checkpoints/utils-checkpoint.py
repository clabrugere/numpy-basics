import numpy as np


def confusion_matrix(y_true, y_hat, threshold=.5):
    
    def _to_class(y):
        return np.array([1 if i >= threshold else 0 for i in y])
    
    n_classes = len(np.unique(y_true))
    cm = np.zeros((n_classes, n_classes))
    y_hat = _to_class(y_hat)
    
    for a, p in zip(y_true, y_hat):
        cm[a, p] += 1
    
    return cm

def f1_score(cm):
    precision = cm[0, 0] / cm[0, :].sum()
    recall = cm[0, 0] / cm[:, 0].sum()
    return 2 * (precision * recall) / (precision + recall)