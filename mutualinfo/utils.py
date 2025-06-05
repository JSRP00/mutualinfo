# mutualinfo/utils.py

import numpy as np

def check_input_shapes(x, y):
    """
    Asegura que x e y tengan la misma longitud y los convierte en arrays 2D.

    Retorna:
    --------
    x, y : arrays de shape (n_samples, 1) o (n_samples, n_features)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x e y deben tener el mismo número de muestras")

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return x, y


def set_random_seed(seed):
    """
    Fija una semilla aleatoria para reproducibilidad.

    Parámetros:
    -----------
    seed : int
    """
    np.random.seed(seed)


def generate_dependent_data(n=1000, noise=0.1, seed=None):
    """
    Genera datos dependientes x → y = sin(2πx) + ruido

    Retorna:
    --------
    x, y : arrays de shape (n_samples,)
    """
    if seed is not None:
        set_random_seed(seed)

    x = np.random.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + noise * np.random.randn(n)
    return x, y


def generate_independent_data(n=1000, seed=None):
    """
    Genera datos independientes x ~ U(0,1), y ~ U(0,1)

    Retorna:
    --------
    x, y : arrays de shape (n_samples,)
    """
    if seed is not None:
        set_random_seed(seed)

    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    return x, y
