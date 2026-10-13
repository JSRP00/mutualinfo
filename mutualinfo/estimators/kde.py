# mutualinfo/estimators/kde.py

import numpy as np
from scipy.stats import gaussian_kde

def estimate_mi(x, y, bandwidth=None, n_samples=10000):
    """
    Mutual information estimator based on kernel density estimation (KDE).
    
    Parameters
    ----------
    x : ndarray, shape (n_samples,)
    y : ndarray, shape (n_samples,)
    bandwidth : float or None, optional KDE bandwidth
    n_samples : int, number of samples used to integrate the MI
    
    Returns
    -------
    float : estimate of the mutual information I(X;Y)
    """

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    xy = np.hstack((x, y)).T  # 2 x N for the joint KDE

    # KDEs
    kde_x = gaussian_kde(x.T, bw_method=bandwidth)
    kde_y = gaussian_kde(y.T, bw_method=bandwidth)
    kde_xy = gaussian_kde(xy, bw_method=bandwidth)

    # Random samples from the joint space
    samples = kde_xy.resample(n_samples)

    # Evaluate the densities
    p_xy = kde_xy(samples)
    p_x = kde_x(samples[0:1])
    p_y = kde_y(samples[1:2])

    # Compute the MI estimate
    mi = np.mean(np.log(p_xy / (p_x * p_y)))

    return mi
