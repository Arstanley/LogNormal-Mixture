import numpy as np
import math
from scipy import linalg
import warnings
from random import uniform
from scipy.misc import logsumexp

class LogMM_back_up:
    """ Log-Normal Mixture For one-dimensional data

    Parameters
    ----------
    n_components: int, defaults to 1.
        The number of mixture components

    tol: float, defaults to 1e-3
        The convergence threshold for EM

    max_iter: int, defaults to 100
        The maximum iteration threshold for EM

    weights_init: array-like, shape (n_components, ), optional
        PDF for components. [p1, p2, ... , pn]
        If not provided, it will be initalized evenly.

    means_init: array-like, shape (n_components, ), optional
        Initialization for Means. If not provided, calculate using kmeans.

    var_init: array-like, shape (n_components, ), optional
        Initialization for variance.
    """

    def __init__(self, n_components=1, tol=1e-3, max_iter=100, weights_init=None, means_init=None, var_init=None):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.weights_init = weights_init
        self.means_init = means_init
        self.var_init = var_init

    def _check_parameters(self):
        """Check the Log Mixture parameters are well defined."""
        if (self.weights_init is not None):
            if np.sum(self.weights_init) != 1:
                raise ValueError("Invalid value for initial weights: Weights array need to have sum equals to 1.")
            if len(self.weights_init) != self.n_components:
                raise ValueError("Invalid length for initial weights: Weights need to have the same length as number of components.")
        if (self.means_init is not None):
            if len(self.means_init) != self.n_components:
                raise ValueError("Invalid length for initial means: Means need to have the same length as number of components.")

    def _initialize_mv(self, X):
        """Initialize means and vars based on sorted value
        Parameters
        ---------
        X: array-like, input
        """
        _sorted = sorted(X)
        mid = len(_sorted) // 2
        return np.array([np.mean(_sorted[:mid]), np.mean(_sorted[mid:])]), np.array([np.var(X) for _ in range (self.n_components)])

    def _initialize(self, X):
        """Initialization for Mixture Model Parameters
        Parameters
        -----------
            X: array-like, input
        """
        means, var = self._initialize_mv(X)
        self._weights = (np.array([1/self.n_components for _ in range(self.n_components)]) if
                self.weights_init is None else self.weights_init)
        self._means = (means if self.means_init is None
                else self.menas_init)
        self._vars = (var if self.var_init is None
                else self.var_init)

    def density_func(self, y, mean, var):
        """Calculation based on lognormal density function

        Parameters
        ----------
        y: float, data point
        mean, var: float, parameter for current model

        Return
        -------
        Calculated Result
        """
        std = var*(1/2)
        return (1/(y*std*np.sqrt(2*math.pi))) * np.exp(-(np.log(y)-mean)**2/(2*var))

    def _e_step(self, X):
        """The expectation step for EM algorithm
        Parameter
        ---------
        X: array-like, input

        It calculates the responsiveness of model k to data point yj

        Return
        --------
        res: matrix-like, has shape(len(X), n_components)
        """
        n_samples = len(X)
        res = np.zeros((n_samples, self.n_components))

        for idx, data_point in enumerate(X):
            denominator = np.sum([self._weights[j] * self.density_func(data_point, self._means[j], self._vars[j]) for j in range(self.n_components)])
            for j in range (self.n_components):
                numerator = self._weights[j] * self.density_func(data_point, self._means[j], self._vars[j])
                res[idx][j] = numerator / denominator

        epslon = 1e-10
        for i in range(res.shape[0]):
            if (res[i,0] == 0):
                res[i,0] += epslon
                res[i,1] -= epslon
            if (res[i,1] == 0):
                res[i,0] -= epslon
                res[i,1] += epslon

        return res

    def _m_step(self, X, res):
        """Expectation Maximizing For EM
        Parameters
        -----------
        X: Array-like, input
        res: Obtained from expectation step. Has shape (len(X), n_components). Responsiveness matrix.

        Return
        -------
        means: Updated means. Has shape (n_components,)
        vars: Updated vars, Has shape (n_components, )
        weights: Updated weights, Has shape (n_components, )
        """
        means = np.zeros((self.n_components, ))
        var = np.zeros((self.n_components, ))
        weights = np.zeros((self.n_components, ))

        n_samples = len(X)
        for i in range(self.n_components):
            means[i] = (np.sum([res[j][i] * X[j] for j in range(n_samples)])) / (np.sum(res, axis=0)[i])
            var[i] = (np.sum([res[j][i] * (np.log(X[j]) - self._means[i])**2 for j in range(n_samples)])) / (np.sum(res, axis=0)[i])
            weights[i] = (np.sum(res, axis=0)[i]) / (n_samples)

        self._means = means
        self._vars = var
        self._weights = weights

    def calc_log_pdf_sum(self, X, resp_mat):
        n_samples = len(X)
        res = 0
        for k in range(self.n_components):
            std = np.sqrt(self._vars[k])
            for j in range(n_samples):
                res += (resp_mat[j][k]*(np.log(1/(X[j]*(2*math.pi)**(1/2)))-
                   np.log(std)-
                   ((np.log(X[j]) - self._means[k])**2/(2 * self._vars[k]))))
        return res

    def calculate_log_prob(self, X, resp_mat):
        resp_mult_weights = np.sum([np.sum(resp_mat, axis=0)[k] * np.log(self._weights[k]) for k in range(self.n_components)])
        log_pdf_sum = self.calc_log_pdf_sum(X, resp_mat)

        return resp_mult_weights + log_pdf_sum

    def initialize_resp(self, X):
        return np.tile(self._weights, (len(X), 1))

    def fit(self, X):
        """Estimate model parameters using X

        Parameters
        -----------
            X: array-like, shape (n_samples, 1)

        Returns
        --------
            None
        """

        # Initializing parameters
        self._initialize(X)
        resp_mat = self.initialize_resp(X)
        self.converged_ = False

        print("Start fitting the data to logNormal mixture model.")
        print("--------------------Parameters-------------------------")
        print(f"n_components = {self.n_components},tolerance={self.tol}, max_iter={self.max_iter}, init_means = {self._means}, init_variance = {self._vars}, init_weights = {self._weights}")
        print("-------------------------------------------------------")
        ### Function incomplete
        log_prob = self.calculate_log_prob(X, resp_mat)
        self._best_iter = None

        for n_iter in range(1, self.max_iter + 1):
            prev_log_prob = log_prob

            # Response matrix from e-step
            resp_mat = self._e_step(X)
            print(resp_mat)
            self._m_step(X, resp_mat)

            log_prob = self.calculate_log_prob(X, resp_mat)
            change = log_prob - prev_log_prob

            if abs(change) < self.tol:
                self.converged_ = True
                self._best_iter = n_iter
                break

        if not self.converged_:
            print ('--------Initialization did not converge.\n'
                    +'Try different init parameters,\n'
                    +'or increase max_iter, tol\n'
                    +'or check for degenerate data.\n --------------------------')
        else:
            print("Successfully fit the data")
            print("-------Parameters--------")
            print(f"means: {self._means}, variance: {self._vars}, weights: {self._weights}")

class LogNormal:
    def __init__(self, mu, sigma):
        #parameters
        self.mu = mu
        self.sigma = sigma
        self.mean = np.exp(self.mu + self.sigma**2 / 2)
        self.var = (np.exp(self.sigma**2)-1) * np.exp(2 * self.mu + self.sigma ** 2)

    # PDF
    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (np.log(datum)-self.mu) / (abs(self.sigma))
        y = (1/(datum * abs(self.sigma) * np.sqrt(2*math.pi))) * np.exp(-u*u/2)
        return y

    def __repr__(self):
        return f'LogNormal({self.mu}, {self.sigma}), mean: {self.mean}, variance: {self.var}'

class LogNormalMM:

    def __init__(self, mix=0.5, tol=1e-10):

        # weight initialization
        self.mix = mix

        # Tolerance
        self.tol = tol

    def e_step(self):

        for datum in self.data:
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.two.pdf(datum) * (1. - self.mix)

            den = wp1 + wp2

            # Normalization
            wp1 /= den
            wp2 /= den

            self.loglike += np.log(wp1 + wp2)

            yield (wp1, wp2)

    def m_step(self, weights):

        (left, right) = zip(*weights)

        one_den = sum(left)
        two_den = sum(right)

        self.one.mu = sum(w * np.log(d) / one_den for (w, d) in zip(left, self.data))
        self.two.mu = sum(w * np.log(d) / two_den for (w, d) in zip(right, self.data))

        self.one.sigma = np.sqrt(sum(w * ((np.log(d) - self.one.mu) ** 2)
            for (w, d) in zip(left, self.data)) / one_den)
        self.two.sigma = np.sqrt(sum(w * ((np.log(d) - self.two.mu) ** 2)
            for (w, d) in zip(right, self.data)) / two_den)
        
        self.mix = one_den / len(self.data)

        return np.array(weights)

    def fit(self, data, max_iterations=100, verbose=True):
        self.data = data
        self.loglike = 0.
        self.best_loglike = float("-inf")
        self._converged = False

        # Initialize the distributions
        log_data = np.log(data)
        self.one = LogNormal(uniform(np.min(log_data), np.max(log_data)), uniform(0.1, np.sqrt(np.var(log_data))))
        self.two = LogNormal(uniform(np.min(log_data), np.max(log_data)), uniform(0.1, np.sqrt(np.var(log_data))))

        # Iterately Fit The Mixture Model With EM Algorithm
        
        for iter in range(max_iterations):
            try:
                self.loglike = 0.
                # Get the response from Maximization Step
                resp = self.m_step(list(self.e_step()))
                if verbose:
                    print(f'Iteration {iter}: {self}')
                # Stop iteration if converges
                if abs(self.best_loglike-self.loglike) < self.tol:
                    self._converged = True 
                    break
                # Update prameters if there's greater loglike
                if abs(self.loglike) > self.best_loglike and self.mix != np.nan:
                    self.best_loglike = self.loglike
                    self.best_mix = self.mix
                    self.best_iter = iter
                    self.best_resp = resp

            except (ZeroDivisionError, ValueError, RuntimeWarning):
                pass

        # Handle Situation When Not Converged   
        if self._converged == False:
            print ('--------Initialization did not converge.\n'
                    +'Try different init parameters,\n'
                    +'or increase max_iter, tol\n'
                    +'or check for degenerate data.\n --------------------------')
        else: 
            print ('----------Log-Normal Mixture Model Successfully Fit-------------')
            print (f'Distribution 1: {self.one}\n Distribution 2: {self.two}')

        

    def fit_predict(self, data, max_iterations = 100, verbose=True):
        self.fit(data, max_iterations, verbose)
        return np.argmax(self.best_resp, axis=1)

    def pdf(self, x):
        return self.mix * self.one.pdf(x) +  (1-self.mix) * self.two.pdf(x)

    def __repr__(self):
        return f'LogNormal Mixture {self.one}, {self.two}, mix = {self.mix}'

    def __str__(self):
        return f'Mixture: {self.one}, {self.two}, mix = {self.mix}'
