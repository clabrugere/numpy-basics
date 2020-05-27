import numpy as np


class Node:
    
    def __init__(self, child_left, child_right, feature_ind, threshold):
        self.child_left = child_left
        self.child_right = child_right
        self.feature_ind = feature_ind
        self.threshold = threshold
        

class Leaf:
    
    def __init__(self, value):
        self.value = value
        
        
class DecisionTreeClassifier:
    
    def __init__(self, max_depth=None, n_features=None):
        
        if max_depth:
            self.max_depth = max_depth
        else:
            self.max_depth = np.inf
        
        self.n_features = n_features
        self.depth = 0
        self.n_classes = None
        self.root = None
    
    def fit(self, X, y):
        self.n_classes = np.max(y) + 1
        
        if self.n_features:
            self.n_features = min(X.shape[1], self.n_features)
        else:
            self.n_features =  X.shape[1]
        
        self.root = self._grow(X, y)
    
    def predict(self, X, return_probs=False):
        if return_probs:
            return np.array([self._traverse(x, node=self.root) for x in X])
        else:
            return np.array([np.argmax(self._traverse(x, node=self.root)) for x in X])
    
    def _grow(self, X, y, current_depth=0):        
        # if max_depth is reached, stop growing
        if self.depth >= self.max_depth:
            _, counts = np.unique(y, return_counts=True)
            probs = counts / np.sum(counts)
            return Leaf(probs)
        
        # if y are all from the same class, stop growing
        if len(np.unique(y)) == 1:
            probs = np.zeros(self.n_classes)
            probs[y[0]] = 1
            return Leaf(probs)
        
        # grow the tree
        current_depth += 1
        self.depth = max(self.depth, current_depth)
        feature_inds = np.random.choice(X.shape[1], self.n_features, replace=False)
        
        # get the best feature to split on the current depth
        feature_ind, threshold = self._split(X, y, feature_inds)
        left_inds = np.argwhere(X[:, feature_ind] < threshold).ravel()
        right_inds = np.argwhere(X[:, feature_ind] >= threshold).ravel()
        
        # grow the sub trees
        left_tree = self._grow(X[left_inds, :], y[left_inds], current_depth)
        right_tree = self._grow(X[right_inds, :], y[right_inds], current_depth)
        
        return Node(left_tree, right_tree, feature_ind, threshold)
    
    def _split(self, X, y, feature_inds):
        # loop over features and find the one with the best information gain
        best_gain = -np.inf
        feature_ind = None
        threshold = None
        parent_entropy = entropy(y)
        
        for i in feature_inds:
            feature_values = X[:, i]
            support = np.unique(feature_values)
            
            # define all thresholds to test
            if len(support) > 1:
                thresholds = (support[1:] + support[:-1]) / 2
            else:
                thresholds = support
            
            # compute impurity gains for each thresholds
            gains = [self._information_gain(feature_values, y, t, parent_entropy) for t in thresholds]
            
            if np.max(gains) > best_gain:
                best_gain = np.max(gains)
                feature_ind = i
                threshold = thresholds[np.argmax(gains)]
        
        return feature_ind, threshold
    
    def _information_gain(self, X, y, threshold, parent_entropy,):
        left_inds = np.argwhere(X < threshold).ravel()
        right_inds = np.argwhere(X >= threshold).ravel()
        
        n_samples = len(y)
        n_left, n_right = len(left_inds), len(right_inds)
        
        if len(left_inds) == 0 or len(right_inds) == 0:
            return 0.
        
        child_entropy = (n_left * entropy(y[left_inds]) + n_right * entropy(y[right_inds])) / n_samples
        ig = parent_entropy - child_entropy
         
        return ig
    
    def _traverse(self, x, node=None):
        # if the current node is a leaf, return its value
        if isinstance(node, Leaf):
            return node.value
        
        # return left or right child tree depending on the split learned
        if x[node.feature_ind] < node.threshold:
            return self._traverse(x, node=node.child_left)
        else:
            return self._traverse(x, node=node.child_right)
    
    
class RandomForestClassifier:
    
    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        if max_depth:
            self.max_depth = max_depth
        else:
            self.max_depth = np.inf
            
        self.forest = []
    
    def fit(self, X, y):
        # create a forest
        M, N = X.shape
        for _ in range(self.n_trees):
            # bootstrap samples and features bagging
            inds = np.random.choice(M, M, replace=True)
            n_features = int(np.sqrt(N))
            X_i, y_i = X[inds], y[inds]
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth, n_features=n_features)
            tree.fit(X_i, y_i)
            self.forest.append(tree)
    
    def predict(self, X, return_probs=False):
        forest_preds = np.array([tree.predict(X, return_probs) for tree in self.forest])
        
        if return_probs:
            return forest_preds.transpose((1, 2, 0)).mean(axis=-1)
        else:
            return np.array([np.bincount(y_hat).argmax() for y_hat in forest_preds.T])

    
def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    p = counts / np.sum(counts)
    entropy = -np.sum([pi * np.log2(pi) for pi in p if pi > 0])
    
    return entropy