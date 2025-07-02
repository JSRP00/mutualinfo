# mutualinfo/utils.py

import numpy as np

def check_input_shapes(x, y):
    """
    Asegura que x e y tengan la misma longitud y los convierte en arrays 2D.
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
    """
    np.random.seed(seed)


def generate_dependent_data(n=1000, relation="nonlinear", noise=0.1, seed=None):
    """
    Genera datos dependientes entre x e y.
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
    """
    if seed is not None:
        set_random_seed(seed)

    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    return x, y


# ------------------- Funciones de visualización -------------------

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_regression_intervals(X_test, y_test, y_pred, y_interval):
    """
    Visualiza intervalos de predicción para regresión.
    """
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X_test, y_test, alpha=0.3, label="Datos reales")

    X_test = X_test.ravel()
    order = np.argsort(X_test)
    plt.plot(X_test[order], y_pred[order], color="C1", label="Predicción")
    plt.plot(X_test[order], y_interval[order][:, 0, 0], color="C1", ls="--", label="Límite inferior")
    plt.plot(X_test[order], y_interval[order][:, 1, 0], color="C1", ls="--", label="Límite superior")
    plt.fill_between(
        X_test[order],
        y_interval[:, 0, 0][order].ravel(),
        y_interval[:, 1, 0][order].ravel(),
        alpha=0.2
    )
    plt.title("Intervalos de predicción estimados")
    plt.legend()
    plt.show()


def plot_classification_regions(X, y, classifier, resolution=0.1):
    """
    Visualiza regiones de confianza para clasificación 2D.
    """
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    X_mesh = np.stack([xx.ravel(), yy.ravel()], axis=1)

    _, pred_set = classifier.predict_set(X_mesh)

    cmap_back = ListedColormap([
        "#c7e8c0", "#fdd0a2", "#9e9ac8",
        "#c6dbef", "#9e9ac8", "#9e9ac8"
    ])
    cmap_dots = ListedColormap([
        "#3182bd", "#e34a33", "#31a354"
    ])

    plt.scatter(
        X_mesh[:, 0], X_mesh[:, 1],
        c=np.ravel_multi_index(pred_set.T, (2, 2, 2)),
        cmap=cmap_back, marker='.', s=10
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_dots)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Regiones de confianza (clasificación)")
    plt.show()
