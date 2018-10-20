"""Implementation of Kmeans algorithm.
"""
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils import load_data
from IPython import display
import seaborn as sns


class Kmeans():
    def __init__(self, data:np.array, K:int, tol:float):
        self.data = data  # (N, 2)
        self.K = K  # number of clusters
        self.tol = tol  # tolerance for convergence

    def train(self,notebook_viz=False):
        """Trains kmeans algorithm. 
        Args:
            notebook_viz (bool, optional): Defaults to False. Visualize algorithmm steps (only if inside a notebook).
        
        Returns:
            np.array, np.array: clusters, centroids.
        """

        N = self.data.shape[0]  #  number of points

        # initialisation
        old_centroids = np.zeros((self.K, 2))
        # draw initial centroids at random from dataset
        ind_random_centroids = np.random.randint(0, N, size=(self.K,))
        centroids = self.data[ind_random_centroids]  #  (K, 2)
        # clusters are also random
        clusters = np.random.randint(0, self.K, size=(N, ))  #  (N, )

        iter = 0
        while np.linalg.norm(centroids - old_centroids) > self.tol:
            # computes distance from all centroids
            dist_to_centroids = self.dist2centroids(centroids)  #  (N, 4)
            # update clusters
            clusters = np.argmin(dist_to_centroids, axis=1)  # (N, )

            data_centroids = np.array([centroids[c]
                                    for c in clusters])  #  expand centroids
            # computes distortion
            distortion = self.data - data_centroids
            distortion = np.linalg.norm(distortion, axis=1)  #  (N, )
            distortion = np.sum(distortion)

            # plot data, clusters and centroids
            if notebook_viz:
                self.plot(centroids, clusters, iter, distortion)
            else:
                print(f"Iteration : {iter}, Distortion : {distortion}")

            # update centroids
            old_centroids = centroids
            centroids = [np.mean(self.data[clusters == j], axis=0) for j in range(self.K)]
            centroids = np.array(centroids)  #  converts from list to np.array

            iter += 1

        print(f"Algorithm has converged after {iter} iterations.")
        return clusters, centroids
    

    def dist2centroids(self, centroids:np.array):
        """Computes euclidian distance from centroids for each data point.
    
        Args:
            centroids (np.array): dim (K, 2).
        
        Returns:
            np.array: dist_to_centroids, distance to all centroids for each datapoint. dim (N, K).
        """
        dist_to_centroids = [self.data - centroids[j] for j in range(self.K)]  #  broadcasting
        dist_to_centroids = [np.linalg.norm(dist_to_centroids[c], axis=1) for c in range(self.K)]  #  computes norm
        dist_to_centroids = np.array(dist_to_centroids).T  #  (N, K)

        return dist_to_centroids


    def plot(self, centroids:np.array, clusters:np.array, iter:int, distortion:float):
        """Plot data, clusters and centroids with nice colors for an ipython cell.
    
        Args:
            centroids (np.array): (K, 2)
            clusters (np.array): (N, )
        """

        # set colors
        colors = sns.color_palette(None, self.K)
        cluster_colors = [colors[c] for c in clusters]
        centroids_colors = colors

        plt.figure(figsize=(7, 4))
        # plot data with color corresponding to their cluster
        plt.scatter(self.data[:, 0], self.data[:, 1], color=cluster_colors, alpha=0.5)
        # plot centroids with specific marker
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    c=centroids_colors, edgecolors="black", marker=".", s=300)
        # display title
        plt.title(
            f"Iteration : {iter}, Distortion: {round(distortion, 3)}", fontsize=18)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.1)