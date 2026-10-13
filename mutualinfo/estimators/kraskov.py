# mutualinfo/estimators/kraskov.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from mutualinfo.utils import check_input_shapes

def estimate_mi(x, y, k=3):
    """
    Mutual information estimator for two continuous variables
    using the method of Kraskov et al. (2004).
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features_x)
    y : ndarray, shape (n_samples, n_features_y)
    k : int, number of nearest neighbours
    
    Returns
    -------
    float : estimate of the mutual information I(X;Y)
    """
    x, y = check_input_shapes(x, y)

    n = len(x)
    data = np.hstack((x, y))

    # Neighbours in the joint space
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(data)
    distances, _ = nbrs.kneighbors(data)
    eps = distances[:, k]  # Distance to the k-th neighbour (index 0 is the point itself)

    # Number of neighbours in each marginal space
    nx = NearestNeighbors(metric='chebyshev').fit(x)
    ny = NearestNeighbors(metric='chebyshev').fit(y)
    nx_count = np.array([len(nx.radius_neighbors([x[i]], radius=eps[i], return_distance=False)[0]) - 1 for i in range(n)])
    ny_count = np.array([len(ny.radius_neighbors([y[i]], radius=eps[i], return_distance=False)[0]) - 1 for i in range(n)])

    # Kraskov estimator formula
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx_count + 1) + digamma(ny_count + 1))

    return mi
