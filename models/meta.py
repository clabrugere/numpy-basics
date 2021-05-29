import numpy as np
from sklearn.utils.extmath import cartesian


class PredictionIntervals:

    def __init__(self, model, n_iter=100, max_samples_nier=1000):
        self.model = model
        self.n_iter = n_iter
        self.max_sample_nier = max_samples_nier
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, alpha=.05):
        n_sample = self.X_train.shape[0]
        self.n_iter = max(self.n_iter, int(np.sqrt(n_sample)))

        pred_b = np.zeros((self.n_iter, X.shape[0]))
        resid_val = []

        # bootstrap
        for b in range(self.n_iter):
            idx_train = np.random.choice(range(n_sample), n_sample, replace=True)
            idx_val = np.setdiff1d(range(n_sample), idx_train)

            self.model.fit(self.X_train[idx_train], self.y_train[idx_train])
            y_hat_b = self.model.predict(self.X_train[idx_val])
            resid_val.append(self.y_train[idx_val] - y_hat_b)
            pred_b[b] = self.model.predict(X)

        resid_val = np.concatenate(resid_val)

        # training residuals
        self.model.fit(self.X_train, self.y_train)
        y_hat_train = self.model.predict(self.X_train)
        resid_train = self.y_train - y_hat_train

        # take percentiles to allow comparison between train and validation
        # residuals
        resid_val = np.percentile(resid_val, q=np.arange(100))
        resid_train = np.percentile(resid_train, q=np.arange(100))

        # compute weighted residuals to account for overfitting as we use
        # training residuals set to estimate predictions intervals
        if n_sample > self.max_sample_nier:
            combs_idx = np.random.choice(range(n_sample), self.max_sample_nier)
            combs = cartesian((self.y_train[combs_idx], y_hat_train[combs_idx]))
        else:
            combs = cartesian((self.y_train, y_hat_train))

        no_info_err_rate = ((combs[:, 0] - combs[:, 1]) ** 2).mean()
        relative_overfit_rate = (resid_val.mean() - resid_train.mean()) / (no_info_err_rate - resid_train.mean())
        weight = .632 / (1 - .368 * relative_overfit_rate)
        resid = (1 - weight) * resid_train + weight * resid_val

        # compute the estimate of the noise around the bootstrapped predictions
        # and take percentiles as prediction intervals
        C = np.array([[m + o for m in pred_b[:, i] for o in resid] for i in range(X.shape[0])])
        q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
        percentiles = np.percentile(C, q, axis=1)
        y_hat = self.model.predict(X)

        return y_hat, percentiles
