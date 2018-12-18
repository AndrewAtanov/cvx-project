import cvxpy as cvx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise


class SVMPrimal(BaseEstimator, ClassifierMixin):

    def __init__(self, lambd=1.):
        self.lambd = lambd

        self.coef = None
        self.bias = None
        self.labels = None
        self.sparsity = None

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise RuntimeError
        _y = y.copy()
        self.labels = np.array(sorted(np.unique(y)))
        _y[_y == self.labels[0]] = -1
        _y[_y == self.labels[1]] = 1

        n, m = X.shape
        w = cvx.Variable(shape=(m,))
        bias = cvx.Variable()
        reg = cvx.norm(w, 2)
        loss = cvx.sum(cvx.pos(1 - cvx.multiply(_y, X * w - bias))) / float(n)
        prob = cvx.Problem(cvx.Minimize(loss + self.lambd * reg))
        prob.solve()

        self.coef = np.array(w.value).squeeze()
        self.bias = bias.value
        self._status = prob.status

        self.sparsity = np.sum(np.isclose(self.coef, 0)) / m

    def predict(self, X):
        p = np.dot(X, self.coef) - self.bias
        return self.labels[(p > 0).astype(int)]


class SVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1., kernel='linear', kernel_params={}):
        self.C = C
        self.kernel_params = kernel_params
        self.kernel = kernel

        self.coef = None
        self.bias = None
        self.labels = None
        self.y = None
        self.X = None
        self.eps = 1e-9
        self.sparsity = None

    def gramm_matrix(self, X, Y=None):
        if self.kernel == 'linear':
            K = pairwise.linear_kernel(X, Y=Y)
        elif self.kernel == 'poly':
            K = pairwise.polynomial_kernel(X, Y=Y, **self.kernel_params)
        elif self.kernel == 'rbf':
            K = pairwise.rbf_kernel(X, Y=Y, **self.kernel_params)
        else:
            raise NotImplementedError
        return K

    def get_signed_gramm_matrix(self, X, y):
        K = self.gramm_matrix(X)
        K *= y[:, np.newaxis]
        K *= y[np.newaxis]
        return K

    def transform_labels(self, y):
        _y = y.copy()
        self.labels = np.array(sorted(np.unique(y)))
        assert len(self.labels) == 2
        _y[_y == self.labels[0]] = -1
        _y[_y == self.labels[1]] = 1
        return _y

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise RuntimeError

        self.y, self.X = self.transform_labels(y), X.copy()

        n, d = self.X.shape
        lambd = cvx.Variable(shape=(n,))
        K = self.get_signed_gramm_matrix(self.X, self.y)
        objective = 0.5 * cvx.atoms.quad_form(lambd, K) - cvx.sum(lambd)
        constraints = [
            lambd <= self.C,
            lambd >= 0,
            self.y @ lambd == 0
        ]
        problem = cvx.Problem(cvx.Minimize(objective), constraints=constraints)
        problem.solve()
        self._status = problem.status

        self.lambd = lambd.value
        self.support = self.lambd > self.eps
        support_boundary = (self.lambd > self.eps) & (
            self.lambd < self.C - self.eps)
        K = self.gramm_matrix(self.X[self.support], self.X[support_boundary])
        self.coef = self.lambd[self.support] * self.y[self.support]
        wx = np.sum(self.coef[:, np.newaxis] * K, axis=0)
        self.bias = np.median(self.y[support_boundary] - wx)

        self.sparsity = 1 - (np.sum(self.support) / len(self.support))

        return self

    def decision_function(self, X):
        K = self.gramm_matrix(self.X[self.support], X)
        wx = np.sum(
            (self.lambd[self.support] * self.y[self.support])[:, np.newaxis] * K, axis=0)
        return wx + self.bias

    def predict(self, X):
        p = self.decision_function(X)
        return self.labels[(p > 0).astype(int)]


class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, lambd=1., norm=2):
        self.lambd = lambd

        self.coef = None
        self.bias = None
        self.labels = None
        self.norm = norm
        self.sparsity = None

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise RuntimeError
        _y = y.copy()
        self.labels = np.array(sorted(np.unique(y)))
        _y[_y == self.labels[0]] = -1
        _y[_y == self.labels[1]] = 1

        n, m = X.shape
        w = cvx.Variable(shape=(m,))
        bias = cvx.Variable()
        reg = cvx.norm(w, self.norm)
        p = -cvx.multiply(_y, X * w - bias)
        loss = cvx.sum(cvx.logistic(p)) / float(n)
        prob = cvx.Problem(cvx.Minimize(loss + self.lambd * reg))
        prob.solve()

        self.coef = np.array(w.value).squeeze()
        self.bias = bias.value
        self._status = prob.status

        self.sparsity = np.sum(np.isclose(self.coef, 0)) / m

    def predict(self, X):
        p = np.dot(X, self.coef) - self.bias
        return self.labels[(p > 0).astype(int)]
