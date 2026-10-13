# mutualinfo/uncertainty/bootstrap.py

import numpy as np

def bootstrap_ci(estimate_func, x, y, n_bootstraps=1000, alpha=0.05, seed=None, **kwargs):
    """
    Computes a bootstrap confidence interval for an MI estimation function.

    Parameters
    ----------
    estimate_func : function that computes I(X;Y), e.g. estimate_mi_kraskov
    x : array-like, variable X
    y : array-like, variable Y
    n_bootstraps : int, number of resamples
    alpha : float, significance level (default 0.05, i.e. a 95% interval)
    seed : int or None, random seed for reproducibility
    **kwargs : extra arguments passed to estimate_func

    Returns
    -------
    (float, float) : lower and upper bounds of the confidence interval
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(x)
    estimates = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n, n, replace=True)
        x_sample = np.array(x)[indices]
        y_sample = np.array(y)[indices]
        est = estimate_func(x_sample, y_sample, **kwargs)
        estimates.append(est)

    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return lower, upper
