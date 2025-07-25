# mutualinfo/estimators/kraskov.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

def estimate_mi(x, y, k=3):
    """
    Estimador de información mutua entre dos variables continuas
    utilizando el método de Kraskov et al. (2004).
    
    Parámetros:
    ------------
    x : ndarray, shape (n_samples, n_features_x)
    y : ndarray, shape (n_samples, n_features_y)
    k : int, número de vecinos más cercanos
    
    Retorna:
    --------
    float : estimación de la información mutua I(X;Y)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    assert len(x) == len(y), "x e y deben tener el mismo número de muestras"

    n = len(x)
    data = np.hstack((x, y))

    # Vecinos en espacio conjunto
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(data)
    distances, _ = nbrs.kneighbors(data)
    eps = distances[:, k]  # Distancia al k-ésimo vecino (ignoramos el 0-ésimo: uno mismo)

    # Número de vecinos en cada espacio marginal
    nx = NearestNeighbors(metric='chebyshev').fit(x)
    ny = NearestNeighbors(metric='chebyshev').fit(y)
    nx_count = np.array([len(nx.radius_neighbors([x[i]], radius=eps[i], return_distance=False)[0]) - 1 for i in range(n)])
    ny_count = np.array([len(ny.radius_neighbors([y[i]], radius=eps[i], return_distance=False)[0]) - 1 for i in range(n)])

    # Fórmula del estimador de Kraskov
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx_count + 1) + digamma(ny_count + 1))

    return mi
