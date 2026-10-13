# mutualinfo/estimators/histogram.py

import numpy as np

def estimate_mi(x, y, bins=10):
    """
    Estimates the mutual information I(X;Y) using 2D histograms.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
    y : ndarray, shape (n_samples,)
    bins : int or (bins_x, bins_y), number of bins used for discretisation

    Returns
    -------
    float : estimate of the mutual information I(X;Y)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Joint and marginal histograms
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint_hist / np.sum(joint_hist)

    x_marginal = np.sum(joint_prob, axis=1)
    y_marginal = np.sum(joint_prob, axis=0)

    # Compute the MI
    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            pxy = joint_prob[i, j]
            px = x_marginal[i]
            py = y_marginal[j]
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))

    return mi
