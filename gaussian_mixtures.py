"""Estimates parameters of gaussian mixtures thanks to Expectation-Maximisation algorithm.

"""

import time
import numpy as np
import argparse

from utils import load_data
from kmeans import Kmeans

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from IPython import display


class GaussianMixturesEM():
    def __init__(self, data: np.array, K: int, tol: float):
        self.data = data
        self.K = K  # number of clusters.
        self.tol = tol  # tolerance for convergence.

    def train(self, kmeans_init:bool, notebook_viz=False):
        """Estimates parameters of the gaussian distributions.
            notebook_viz (bool, optional): Defaults to False. Visualize the results (only if inside a notebook).
        
        Returns:
            np.array, np.array, np.array: clusters, means, covariance matrices of normal distributions.
        """

        # Initialisation
        # means and clusters
        if kmeans_init:
            print("Kmeans initialisation")
            clusters, mu = Kmeans(
                self.data, self.K, self.tol).train()  #  mu : (K, 2)
        else: 
            # random means 
            ind_rand_mu = np.random.randint(0, self.data.shape[0], size=(self.K,))
            mu = self.data[ind_rand_mu]  #  (K, 2)
            # clusters are also random
            clusters = np.random.randint(0, self.K, size=(self.data.shape[0], ))  #  (N, )

        mu_old = np.zeros_like(mu)
        # pi
        pi = [(clusters == k).sum() for k in range(self.K)]
        pi = np.expand_dims(pi, 1)  # (K, 1) for broadcasting
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
            sigma = sigma / np.expand_dims(tau.sum(axis=1, keepdims=True), 3)  #  (K, 2, 2)

            # reallocate cluster
            clusters = np.argmax(tau, axis=0).T  #  (N, )

            iter += 1

        return clusters, mu, sigma

    def eigsorted(self, cov):
        """computes eigen values and eigen vectors of covariance matrix in decreasing order.
        Useful for plotting confidence intervals.
        
        Args:
            cov (np.array): covariance matrix.
        
        Returns:
            np.array, np.array: eigen values, eigen vectors.
        """ 

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def plot(self, clusters: np.array, mu: np.array, sigma: np.array, log_lik:float, iter:int):
        """Plot data, means and covariance matrix of normal distributions.
        
        Args:
            clusters (np.array): .
            mu (np.array): means of normal distributions.
            sigma (np.array): covariances matrices of normal distributions.
            log_lik (float): log likelihood.
            iter (int): iteration.
        """

        # set colors
        colors = sns.color_palette(None, self.K)
        cluster_colors = [colors[c] for c in clusters]

        plt.figure(figsize=(15, 8))
        ax = plt.gca()  #  get axes of current figure

        # plot data with color corresponding to their cluster
        plt.scatter(self.data[:, 0], self.data[:, 1], color=cluster_colors, alpha=0.5)
        # plot gaussian mean with specific marker
        plt.scatter(mu[:, 0], mu[:, 1], c=colors,
                    edgecolors="black", marker=".", s=300)

        # plot ellipses of each gaussian
        for k in range(self.K):
            eigvals, eigvecs = self.eigsorted(sigma[k])
            theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  # ellipse angle
            w, h = 2 * 2 * np.sqrt(eigvals)
            ellipse = Ellipse(xy=mu[k], width=w, height=h, angle=theta, alpha=0.1, color=colors[k])
            ax.add_patch(ellipse)

        # display title
        plt.title(
            f"Iteration : {iter}, log_likelihood : {log_lik}", fontsize=25)
        display.clear_output(wait=True)
        display.display(plt.gcf())  # plt.gcf : get a reference to the current figure
        time.sleep(0.1)

    def loglik(self, clusters: np.array, mu: np.array, sigma: np.array, pi: np.array):
        """Computes the complete log likelihood of gaussian mixtures model.

        Args:
            clusters (np.array): .
            mu (np.array): means of normal distribution.
            sigma (np.array): covariance matrices of normal distributions.
            pi (np.array) : frequency of each distribution.
        
        Returns:
            float: Complete log likelihood.
        """
        # mask
        cluster_mask = [clusters == k for k in range(self.K)]
        cluster_mask = np.array(cluster_mask)  # (K, N)

        # left part
        ll_left = [np.log(pi[k]) for k in range(self.K)]
        ll_left = np.expand_dims(ll_left, 1)  # (K, 1)
        # right part
        ll_right = [np.log(multivariate_normal(mean=mu[k], cov=sigma[k]).pdf(self.data)) for k in range(self.K)]
        ll_right = np.array(ll_right)  # (K, N)

        # final log likelihood
        ll = ll_left + ll_right
        ll = cluster_mask * ll
        ll = ll.sum() # scalar

        return ll 
