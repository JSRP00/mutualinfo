# mutualinfo/utils.py

import numpy as np

def check_input_shapes(x, y):
    """
    Checks that x and y have the same length and converts them to 2D arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return x, y


def set_random_seed(seed):
    """
    Sets a random seed for reproducibility.
    """
    np.random.seed(seed)


def generate_dependent_data(n_samples=1000, relation="nonlinear", noise=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    x = rng.uniform(0, 1, n_samples)

    if relation == "linear":
        y = 2 * x + noise * rng.standard_normal(n_samples)
    elif relation == "nonlinear":
        y = np.sin(2 * np.pi * x) + noise * rng.standard_normal(n_samples)
    else:
        raise ValueError("relation must be 'linear' or 'nonlinear'")

    return x.reshape(-1, 1), y


def generate_independent_data(n=1000, seed=None):
    """
    Generates independent data x ~ U(0,1), y ~ U(0,1).
    """
    if seed is not None:
        set_random_seed(seed)

    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    return x, y


# ------------------- Plotting helpers -------------------

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_regression_intervals(X_test, y_test, y_pred, y_interval):
    """
    Plots prediction intervals for regression.
    """
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X_test, y_test, alpha=0.3, label="Observed data")

    X_test = X_test.ravel()
    order = np.argsort(X_test)
    plt.plot(X_test[order], y_pred[order], color="C1", label="Prediction")
    plt.plot(X_test[order], y_interval[order][:, 0, 0], color="C1", ls="--", label="Lower bound")
    plt.plot(X_test[order], y_interval[order][:, 1, 0], color="C1", ls="--", label="Upper bound")
    plt.fill_between(
        X_test[order],
        y_interval[:, 0, 0][order].ravel(),
        y_interval[:, 1, 0][order].ravel(),
        alpha=0.2
    )
    plt.title("Estimated prediction intervals")
    plt.legend()
    plt.show()


def plot_classification_regions(X, y, classifier, resolution=0.1):
    """
    Plots confidence regions for 2D classification.
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
    plt.title("Confidence regions (classification)")
    plt.show()

# -------------------------

from sklearn.model_selection import train_test_split

def train_conformalize_test_split(X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=None):
    """
    Splits X and y into three disjoint subsets:
    - Train
    - Calibration (conformalize)
    - Test

    The proportions must add up to 1.0.
    """
    assert abs(train_size + conformalize_size + test_size - 1.0) < 1e-6, "Las proporciones deben sumar 1."

    # Split: train vs. remainder
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - train_size), random_state=random_state
    )

    # Split: calibration vs. test
    cal_ratio = conformalize_size / (conformalize_size + test_size)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - cal_ratio), random_state=random_state
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test



from sklearn.datasets import make_classification

def generate_classification_data(n=1000, n_features=2, n_classes=3, class_sep=1.0, seed=None):
    """
    Generates synthetic data for multiclass classification.

    Parameters:
    - n: number of samples
    - n_features: number of features (ideally 2 for plotting)
    - n_classes: number of classes
    - class_sep: class separation
    - seed: random seed

    Returns:
    - X: feature array
    - y: labels
    """
    if seed is not None:
        set_random_seed(seed)

    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=seed
    )
    return X, y

