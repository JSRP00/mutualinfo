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


def generate_dependent_data(n=1000, relation="nonlinear", noise=0.1, seed=None):
    """
    Genera datos dependientes entre x e y.

    Parámetros:
    -----------
    relation : str
        Tipo de relación ('linear' o 'nonlinear')
    """
    if seed is not None:
        set_random_seed(seed)

    x = np.random.uniform(0, 1, n)

    if relation == "linear":
        y = 2 * x + noise * np.random.randn(n)
    elif relation == "nonlinear":
        y = np.sin(2 * np.pi * x) + noise * np.random.randn(n)
    else:
        raise ValueError("relation debe ser 'linear' o 'nonlinear'")

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
