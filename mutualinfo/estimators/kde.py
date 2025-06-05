# mutualinfo/estimators/kde.py

import numpy as np
from scipy.stats import gaussian_kde

def estimate_mi_kde(x, y, bandwidth=None, n_samples=10000):
    """
    Estimador de información mutua usando Kernel Density Estimation (KDE).
    
    Parámetros:
    ------------
    x : ndarray, shape (n_samples,)
    y : ndarray, shape (n_samples,)
    bandwidth : float o None, parámetro opcional para KDE
    n_samples : int, número de muestras para integrar la MI
    
    Retorna:
    --------
    float : estimación de la información mutua I(X;Y)
    """

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    xy = np.hstack((x, y)).T  # 2 x N para KDE conjunta

    # KDEs
    kde_x = gaussian_kde(x.T, bw_method=bandwidth)
    kde_y = gaussian_kde(y.T, bw_method=bandwidth)
    kde_xy = gaussian_kde(xy, bw_method=bandwidth)

    # Muestras aleatorias del espacio conjunto
    samples = kde_xy.resample(n_samples)

    # Evaluamos las densidades
    p_xy = kde_xy(samples)
    p_x = kde_x(samples[0:1])
    p_y = kde_y(samples[1:2])

    # Calculamos la estimación de MI
    mi = np.mean(np.log(p_xy / (p_x * p_y)))

    return mi
