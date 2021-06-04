import numpy as np


class LOWESS:

    def __init__(self, sigma=1., frac=1., eps=1e-8):
        self.sigma = sigma
        self.frac = frac
        self.eps = eps
        self.X_ = None
        self.y_ = None

    def _compute_weights(self, x):
        distances = np.linalg.norm(self.X_ - x[:, None], axis=-1)

        # gaussian kernel where sigma define the brandwidth
        weights = np.exp(-(distances ** 2) / self.sigma)

        # take weights of the closest points only
        weights = weights * (distances <= np.quantile(distances, q=self.frac))

        # clip weights close to zero
        weights = np.where(np.abs(weights) >= self.eps, weights, 0.)

        return weights

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y

    def predict(self, X):
        n_samples = X.shape[0]
        y_hat = np.zeros(n_samples)
        for i in range(n_samples):
            y_hat[i] = np.average(
                self.y_,
                weights=self._compute_weights(X[i, :]),
            )

        return y_hat
