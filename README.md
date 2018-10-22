# K-means
Clusters data in K groups via K-means algorithm. 
Find below an example on how to use it : 
```python
from kmeans import Kmeans
km = Kmeans(data=data, K=K, tol=tol)
clusters, centroids = km.train()
```
![alt text](https://github.com/CharlieCheckpt/Kmeans-GMM/blob/master/data/gif/kmeans.gif?raw=true "K-means")

# Gaussian Mixtures Models (GMM) 
Estimates parameters ![](https://latex.codecogs.com/gif.latex?%5Cmu%2C%20%5CSigma) of a mixture of K gaussian distributions using Expectation-Maximisation algorithm. There are two versions of this algorithm in this repository : 
* isotropic : assumes covariance matrices are of the form ![](https://latex.codecogs.com/gif.latex?%5CSigma%20%3D%20%5Csigma%5E2%20I).
* general : no assumptions on form of covariance matrices.

A good practice for GMM is to initialize parameters and clusters with k-means, but you don't have to. You can specify this option with the boolean argument `kmeans_init`. 

## Isotropic GMM
```python
from gmm import GaussianMixturesEM
gm = GaussianIsoMixturesEM(data=data, K=4, tol=0.1)
_ = gm.train(kmeans_init=True, notebook_viz=True)
```
![alt text](https://raw.githubusercontent.com/CharlieCheckpt/Kmeans-GMM/master/data/gif/gmm_iso.gif "Gaussian Mixtures Model")

## General GMM
```python
from gmm import GaussianMixturesEM
gm = GaussianMixturesEM(data=data, K=4, tol=0.1)
clusters, mu, sigma = gm.train(kmeans_init=False)
```
![alt text](https://raw.githubusercontent.com/CharlieCheckpt/Kmeans-GMM/master/data/gif/gmm.gif "Gaussian Mixtures Model")


##### You can visualize the steps of both algorithm in the notebook `viz.ipynb`.

