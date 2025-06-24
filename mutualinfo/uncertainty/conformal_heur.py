# mutualinfo/uncertainty/conformal_heur.py

import numpy as np

def conformal_ci(estimate_func, x, y, calibration_ratio=0.3, alpha=0.05, n_samples=100, seed=None, **kwargs):
    """
    Intervalo de predicción conformal para la estimación de información mutua.
    
    Parámetros:
    ------------
    estimate_func : función de estimación de MI
    x : array-like, variable X
    y : array-like, variable Y
    calibration_ratio : float (0-1), proporción de muestras para calibración
    alpha : nivel de significación (por defecto 0.05 → IC 95%)
    n_samples : número de remuestreos para construir distribución
    seed : int o None, semilla
    **kwargs : argumentos adicionales para estimate_func
    
    Retorna:
    --------
    (float, float) → intervalo de predicción (conformal)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.array(x)
    y = np.array(y)
    n = len(x)

    # Partición en calibración y evaluación
    idx = np.random.permutation(n)
    n_cal = int(n * calibration_ratio)
    idx_cal = idx[:n_cal]
    idx_eval = idx[n_cal:]

    x_cal, y_cal = x[idx_cal], y[idx_cal]
    x_eval, y_eval = x[idx_eval], y[idx_eval]

    # Estimación de referencia con evaluación
    mi_eval = estimate_func(x_eval, y_eval, **kwargs)

    # Generamos n_samples estimaciones de MI con datos de calibración
    mi_samples = []
    for _ in range(n_samples):
        idx_boot = np.random.choice(n_cal, n_cal, replace=True)
        x_boot = x_cal[idx_boot]
        y_boot = y_cal[idx_boot]
        mi_sample = estimate_func(x_boot, y_boot, **kwargs)
        mi_samples.append(mi_sample)

    # Conformal interval: percentil sobre desviaciones respecto a referencia
    residuals = np.abs(np.array(mi_samples) - mi_eval)
    q = np.quantile(residuals, 1 - alpha)

    return mi_eval - q, mi_eval + q
