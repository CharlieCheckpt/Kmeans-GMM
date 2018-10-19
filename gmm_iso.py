"""Estimates parameters of gaussian mixtures with isotropic covariances matrices
( $\Sigma = \sigma^{2} I$ ) thanks to Expectation-Maximisation algorithm.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import display
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from kmeans import Kmeans
from utils import load_data


class GaussianIsoMixturesEM():
    def __init__(self, data: np.array, K: int, tol: float):
        self.data = data
        self.K = K
        self.tol = tol

    def train(self, notebook_viz=False):  #  todo:
        raise NotImplementedError

    def plot(self, clusters: np.array, mu: np.array, sigma: np.array, log_lik: float, iter: int):
        """Plots data points,

        Args:
            clusters (np.array): [description]
            mu (np.array): [description]
            sigma (np.array): [description]
            log_lik (float): [description]
            iter (int): [description]
        """

        # set colors
        colors = sns.color_palette(None, self.K)
        cluster_colors = [colors[c] for c in clusters]

        plt.figure(figsize=(15, 8))
        ax = plt.gca()  #  get axes of current figure

        # plot data with color corresponding to their cluster
        plt.scatter(self.data[:, 0], self.data[:, 1],
                    color=cluster_colors, alpha=0.5)
        # plot gaussian mean with specific marker
        plt.scatter(mu[:, 0], mu[:, 1], c=colors,
                    edgecolors="y", marker="*", s=300)

        # plot ellipses of each gaussian
        for j in range(self.K):
            # todo: replace by plotting circles
            ellipse = Ellipse(mu[j], width=sigma[j][0, 0], height=sigma[j]
                              [1, 1], edgecolor=colors[j], fill=False)
            ax.add_patch(ellipse)

        # display title
        plt.title(
            f"Iteration : {iter}, log_likelihood : {log_lik}", fontsize=25)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.1)

    def loglik(self, clusters: np.array, mu: np.array, sigma: np.array, pi: np.array):
        """Computes the complete log likelihood of the gaussian mixtures.

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
        ll1 = [np.log(pi[k]) for k in range(self.K)]
        ll1 = np.expand_dims(ll1, 1)  #  (K, 1)
        # right part
        ll2 = [np.log(multivariate_normal(mean=mu[k], cov=sigma[k]).pdf(
            self.data)) for k in range(self.K)]
        ll2 = np.array(ll2)  #  (K, N)

        # final log likelihood
        ll = ll1 + ll2
        ll = cluster_mask * ll
        ll = ll.sum()

        return ll
