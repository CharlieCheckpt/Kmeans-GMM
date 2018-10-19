#K-means
Clusters data in K groups via K-means algorithm. 
Find below an example on how to use it : 
```
from kmeans import Kmeans
km = Kmeans(data=data, K=K, tol=tol)
clusters, centroids = km.train()
```

# Gaussian Mixtures Models (GMM) 
Estimates parameters $\mu$, $\Sigma$ of a mixture of K gaussian distributions using Expectation-Maximisation algorithm. There are two versions of this algorithm in this repository : 
* isotropic : assumes covariance matrices are of the form $\Sigma = \sigma^2 I$.
* general : no assumptions on form of covariance matrices.

A good practice for GMM is to initialize parameters and clusters with k-means, but you don't have to. You can specify this option with the boolean argument `kmeans_init`. 
Here is an example with general EM :
```
from gmm import GaussianMixturesEM
gm = GaussianMixturesEM(data=data, K=K, tol=tol)
clusters, mu, sigma = gm.train(kmeans_init=kmeans_init)
```

##### You can visualize the steps of both algorithm in the notebook `viz.py`.

