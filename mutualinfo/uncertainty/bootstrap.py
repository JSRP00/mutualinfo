# mutualinfo/uncertainty/bootstrap.py

import numpy as np

def bootstrap_ci(estimate_func, x, y, n_bootstraps=1000, alpha=0.05, seed=None, **kwargs):
    """
    Calcula un intervalo de confianza bootstrap para una función de estimación de MI.

    Parámetros:
    ------------
    estimate_func : función que calcula I(X;Y) → por ejemplo, estimate_mi_kraskov
    x : array-like, variable X
    y : array-like, variable Y
    n_bootstraps : int, número de remuestreos
    alpha : float, nivel de significación (por defecto 0.05 → IC del 95%)
    seed : int o None, semilla para reproducibilidad
    **kwargs : argumentos extra para la función estimate_func

    Retorna:
    --------
    (float, float) → límite inferior y superior del intervalo de confianza
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
