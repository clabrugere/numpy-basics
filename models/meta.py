import numpy as np
from sklearn.utils.extmath import cartesian


class PredictionIntervals:

    def __init__(self, model, n_iter=100, max_samples=1000):
        self.model = model
        self.n_iter = n_iter
        self.max_samples = max_samples
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, alpha=.05):
        n_sample = self.X_train.shape[0]
        self.n_iter = max(self.n_iter, int(np.sqrt(n_sample)))

        y_hat_b = np.zeros((self.n_iter, X.shape[0]))
        residuals_val = []

        # bootstrap
        for b in range(self.n_iter):
            idx_train = np.random.choice(np.arange(n_sample), n_sample, replace=True)
            idx_val = np.setdiff1d(np.arange(n_sample), idx_train)

            self.model.fit(self.X_train[idx_train], self.y_train[idx_train])
            y_hat_train_b = self.model.predict(self.X_train[idx_val])
            residuals_val.append(self.y_train[idx_val] - y_hat_train_b)
            y_hat_b[b] = self.model.predict(X)

        residuals_val = np.concatenate(residuals_val)

        # training residuals
        self.model.fit(self.X_train, self.y_train)
        y_hat_train = self.model.predict(self.X_train)
        residuals_train = self.y_train - y_hat_train

        # take percentiles to allow comparison between train and validation
        # residuals
        residuals_val = np.percentile(residuals_val, q=np.arange(100))
        residuals_train = np.percentile(residuals_train, q=np.arange(100))

        # compute weighted residuals to account for overfitting as we use
        # training residuals set to estimate predictions intervals
        if n_sample > self.max_samples:
            combs_idx = np.random.choice(np.arange(n_sample), self.max_samples)
            combs = cartesian((self.y_train[combs_idx], y_hat_train[combs_idx]))
        else:
            combs = cartesian((self.y_train, y_hat_train))

        no_info_err_rate = ((combs[:, 0] - combs[:, 1]) ** 2).mean()
        relative_overfit_rate = (residuals_val.mean() - residuals_train.mean()) / (no_info_err_rate - residuals_train.mean())
        weight = .632 / (1 - .368 * relative_overfit_rate)
        residuals = (1 - weight) * residuals_train + weight * residuals_val

        # compute the estimate of the noise around the bootstrapped predictions
        # and take percentiles as prediction intervals
        C = np.array([[m + o for m in y_hat_b[:, i] for o in residuals] for i in range(X.shape[0])])
        q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
        percentiles = np.percentile(C, q, axis=1)
        y_hat = self.model.predict(X)

        return y_hat, percentiles
