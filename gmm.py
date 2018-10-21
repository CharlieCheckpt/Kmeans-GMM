"""Estimates parameters of gaussian mixtures thanks to Expectation-Maximisation algorithm.

"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import display
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from kmeans import Kmeans
from utils import load_data
from gmm_iso import GaussianIsoMixturesEM


class GaussianMixturesEM(GaussianIsoMixturesEM):
    def __init__(self, data: np.array, K: int, tol: float):
        super(GaussianMixturesEM, self).__init__(data, K, tol)


    def train(self, kmeans_init: bool, notebook_viz=False):
        """Estimates parameters of the gaussian distributions.
            notebook_viz (bool, optional): Defaults to False. Visualize the results (only if inside a notebook).

        Returns:
            np.array, np.array, np.array: clusters, means, covariance matrices of normal distributions.
        """

        # Initialisation
        # means and clusters
        if kmeans_init:
            print("Kmeans initialisation...")
            clusters, mu = Kmeans(
                self.data, self.K, self.tol).train()  #  mu : (K, 2)
        else:
            print("Random initialisation...")
            # random means
            ind_rand_mu = np.random.randint(
                0, self.data.shape[0], size=(self.K,))
            mu = self.data[ind_rand_mu]  #  (K, 2)
            # clusters are also random
            clusters = np.random.randint(
                0, self.K, size=(self.data.shape[0], ))  #  (N, )

        mu_old = np.zeros_like(mu)
        # pi
        pi = [(clusters == k).sum() for k in range(self.K)]
        pi = np.expand_dims(pi, 1)  #  (K, 1) for broadcasting
        pi = pi / self.data.shape[0]
        # covariance matrix
        sigma = [np.cov(self.data[clusters == k].T) for k in range(self.K)]
        sigma = np.dstack(sigma).T  #  (K, 2, 2)
        sigma_old = np.zeros_like(sigma)

        log_lik = []

        iter = 0
        while np.linalg.norm(mu - mu_old) > self.tol or np.linalg.norm(sigma - sigma_old) > self.tol:
            # Computes complete log likelihood
            log_lik.append(self.loglik(clusters, mu, sigma, pi))

            # plot (or not plot)
            if notebook_viz:
                self.plot(clusters, mu, sigma, log_lik[-1], iter)
            else:
                print(f"Iteration : {iter}, log-likelihood : {log_lik[-1]}")

            # E-step
            # computes tau : proba that cluster j contains point i
            tau = [multivariate_normal(mean=mu[k], cov=sigma[k]).pdf(
                self.data) for k in range(self.K)]
            tau = np.array(tau)
            tau = pi * tau
            # broadcasting, tau : (K, N)
            tau = tau / tau.sum(axis=0, keepdims=True)
            # M-step
            # pi
            pi = tau.mean(axis=1, keepdims=True)
            # mu
            mu_old = mu
            mu = tau.dot(self.data) / tau.sum(axis=1, keepdims=True)  #  (K, 2)
            # sigma
            sigma_old = sigma
            sigma = [(self.data - mu[k]).T.dot(np.diag(tau[k])
                                               ).dot(self.data - mu[k]) for k in range(self.K)]
            sigma = np.array(sigma)
            sigma = sigma / \
                np.expand_dims(tau.sum(axis=1, keepdims=True), 3)  #  (K, 2, 2)

            # reallocate cluster
            clusters = np.argmax(tau, axis=0).T  #  (N, )

            iter += 1
        print(f"Algorithm has converged after {iter} iterations")

        return clusters, mu, sigma