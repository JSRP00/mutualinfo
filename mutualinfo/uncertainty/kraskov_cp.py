import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import is_classifier, is_regressor
from scipy.stats import entropy
from collections import Counter
from mapie.classification import SplitConformalClassifier
from mapie.regression import SplitConformalRegressor
from mapie.metrics import regression_coverage_score
from mutualinfo.utils import train_conformalize_test_split


def estimate_mi_kraskov_conformal(
    X,
    y,
    model=None,
    alpha=0.1,
    task="auto",
    n_bins=5,
    test_size=0.2,
    cal_size=0.2,
    random_state=42
):
    """
    Estima la información mutua I(Y;X) utilizando Kraskov con predicción conformal
    sobre modelos KNN, tanto en clasificación como regresión.

    Parámetros:
    - X, y: datos de entrada (arrays).
    - model: modelo base (KNN classifier o regressor).
    - alpha: nivel de error (1 - confianza).
    - task: "classification", "regression" o "auto".
    - n_bins: número de bins si se discretiza y (para entropía).
    - test_size, cal_size: proporciones del split.
    - random_state: semilla de aleatoriedad.

    Devuelve:
    - mi: información mutua estimada.
    - h_y: entropía marginal de Y.
    - h_y_given_x: entropía condicional estimada.
    - coverage: cobertura empírica alcanzada.
    """

    # Inferencia automática del tipo de tarea si no se especifica
    if task == "auto":
        task = "regression" if np.issubdtype(y.dtype, np.floating) else "classification"

    # Discretización si es regresión (para calcular H(Y) y H(Y|X))
    if task == "regression":
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        y_disc = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    else:
        y_disc = y

    # División en train/cal/test
    X_train, X_cal, X_test, y_train, y_cal, y_test = train_conformalize_test_split(
        X, y_disc,
        train_size=1 - cal_size - test_size,
        conformalize_size=cal_size,
        test_size=test_size,
        random_state=random_state
    )

    # Modelo base por defecto si no se especifica
    if model is None:
        model = KNeighborsClassifier(n_neighbors=5) if task == "classification" else KNeighborsRegressor(n_neighbors=5)

    model.fit(X_train, y_train)
    confidence = 1 - alpha

    if task == "classification":
        # Conformal Prediction para clasificación
        cp = SplitConformalClassifier(estimator=model, confidence_level=confidence, prefit=True)
        cp.conformalize(X_cal, y_cal)
        _, y_pred_set = cp.predict_set(X_test)

        n_classes = len(np.unique(y_disc))
        entropies = []

        for pred in y_pred_set:
            if isinstance(pred, np.ndarray) and pred.dtype == bool:
                pred = np.where(pred)[0].tolist()
            elif isinstance(pred, np.ndarray):
                pred = pred.tolist()
            if len(pred) == 0:
                continue
            probs = np.zeros(n_classes)
            for c in pred:
                probs[c] = 1 / len(pred)
            entropies.append(entropy(probs, base=2))

        h_y_given_x = np.mean(entropies)

        counts = np.array(list(Counter(y_test).values()))
        probs_y = counts / counts.sum()
        h_y = entropy(probs_y, base=2)

        mi = h_y - h_y_given_x
        coverage = cp.coverage_score(X_test, y_test)

        return mi, h_y, h_y_given_x, coverage

    else:
        # Conformal Prediction para regresión
        cp = SplitConformalRegressor(estimator=model, confidence_level=confidence, prefit=True)
        cp.conformalize(X_cal, y_cal)
        _, intervals = cp.predict_interval(X_test)

        lengths = intervals[:, 1] - intervals[:, 0]
        pseudo_entropy = np.log(lengths + 1e-8)  # evitar log(0)
        h_y_given_x = np.mean(pseudo_entropy)

        counts = np.array(list(Counter(y_test).values()))
        probs_y = counts / counts.sum()
        h_y = entropy(probs_y, base=2)

        mi = h_y - h_y_given_x
        coverage = regression_coverage_score(y_test, intervals)[0]

        return mi, h_y, h_y_given_x, coverage
