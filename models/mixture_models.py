import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixture:

    def __init__(self, n_components, max_iter=100, n_restart=10, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_restart = n_restart
        self._loss = -np.inf
        self._w = None
        self._mu = None
        self._sigma = None
        self._tol = tol

    def fit(self, X, y=None):
        for i in range(self.n_restart):
            w, mu, sigma, loss = self._fit(X, y)

            if loss > self._loss:
                self._loss = loss
                self._w = w
                self._mu = mu
                self._sigma = sigma

        return self

    def predict(self, X):
        return self._e_step(X, self._w, self._mu, self._sigma)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def _fit(self, X, y=None):
        loss = -np.inf
        w, mu, sigma = self._init_params(X)

        for i in range(self.max_iter):
            loss_prev = loss
            probs = self._e_step(X, w, mu, sigma)
            w, mu, sigma = self._m_step(X, probs)
            loss = self._compute_loss(X, w, mu, sigma)

            if np.abs(loss_prev - loss) < self._tol:
                break

        return w, mu, sigma, loss

    def _init_params(self, X):
        w = np.ones((self.n_components)) / self.n_components
        mu = (X[np.random.randint(X.shape[0], size=self.n_components)]).T
        sigma = np.dstack([np.cov(X.T)] * self.n_components)

        return w, mu, sigma

    def _gaussian_prob(self, x, mu, sigma, log=False):
        if log:
            return multivariate_normal.logpdf(x, mean=mu, cov=sigma)
        else:
            return multivariate_normal.pdf(x, mean=mu, cov=sigma)

    def _e_step(self, X, w, mu, sigma):
        probs = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            probs[:, k] = w[k] * self._gaussian_prob(X, mu[:, k], sigma[:, :, k])

        probs /= probs.sum(axis=1).reshape(-1, 1)

        return probs

    def _m_step(self, X, probs):
        n_k, n = probs.sum(axis=0), X.shape[0]

        w = n_k / n
        mu = (X.T @ probs) / n_k
        sigma = np.zeros((X.shape[1], X.shape[1], self.n_components))
        for k in range(self.n_components):
            sigma[:, :, k] = (probs[:, k].reshape(X.shape[0], 1) * (X - mu[:, k])).T @ (X - mu[:, k]) / (probs[:, k].sum())

        return w, mu, sigma

    def _compute_loss(self, X, w, mu, sigma):
        loss = 0
        for k in range(self.n_components):
            loss += np.log(w[k]) + self._gaussian_prob(X, mu[:, k], sigma[:, :, k], log=True)

        return loss.sum()
