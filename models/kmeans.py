import numpy as np


class KMeans:
    '''Consiste à trouver n_clusters partitions qui minimisent la variance au sein d'une même partition pour un ensemble de points donnés
    '''
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self._cluster_centroids = None
        self._labels = None
        self._score = np.inf
        
    def _init_centroids(self, lower, upper, n_features):
        '''Initialisation naive: les barycentres des clusters sont tirés d'une distribution uniforme définie par lower et upper
        '''
        return (upper - lower) * np.random.random((self.n_clusters, n_features)) + lower
    
    def fit(self, X, max_iter=300, tol=1e-4, n_trials=10):
        '''Assigne un unique cluster à chaque point de X: 
            
        Initialise les barycentres aléatoirement, 
        Jusqu'à convergence ou que le nombre maximal d'itérations soit atteint:
            1 - assigne le cluster avec le barycentre le plus proche
            2 - recalcule les barycentres en prenant la moyenne des points qui sont assigné au cluster correspondant 
        
        Repète l'opération n_trials fois car l'agorithme est heuristique et ne garantit pas de trouver une solution optimale globale.
        '''
        for _ in range(n_trials):
            cluster_centroids = self._init_centroids(np.min(X), np.max(X), X.shape[1])
            labels = np.zeros(X.shape[0])
            score = np.inf
            
            for _ in range(max_iter):
                labels, score = self._cluster_assignement(X, cluster_centroids)
                new_centroids = self._compute_centroids(X, cluster_centroids, labels)

                if  np.sum((new_centroids - cluster_centroids)**2) < tol:
                    break

                cluster_centroids = new_centroids
                
            if score < self._score:
                self._cluster_centroids = cluster_centroids
                self._labels = labels
                self._score = score
        
        return self._labels
    
    def _cluster_assignement(self, X, cluster_centroids):
        '''Calcule les distances euclidiennes au carré entre chaque point et les barycentres: matrice de dimensions (n_points, n_clusters) et retourne
        le cluster avec le barycentre le plus proche: vecteur de dimension (n_points,) ainsi que le score: somme des distances au carré au cluster le plus proche
        
        L'utilisation de la distance euclidienne permet de calculer la matrice de façon efficiente en utilisant les fonctions optimisées de numpy (pas de boucle)
        '''
        squarred_distances = -2 * X @ cluster_centroids.T + np.sum(cluster_centroids**2, axis=1) + np.sum(X**2, axis=1)[:, None]
        labels = squarred_distances.argmin(axis=1)
        score = squarred_distances.min(axis=1).sum()
        
        return labels, score
    
    def _compute_centroids(self, X, old_cluster_centroids, labels):
        '''Calcule les nouveaux barycentres en prenant la moyenne des points appartenant au cluster du barycentre, pour chaque barycentre
        '''
        new_cluster_centroids = old_cluster_centroids.copy()
        
        for i in range(self.n_clusters):
            label_mask = (labels == i)
            
            if X[label_mask].shape[0] > 0:
                new_cluster_centroids[i] = X[label_mask].mean(axis=0)
        
        return new_cluster_centroids