# mutualinfo/uncertainty/conformal.py

from sklearn.base import is_regressor, is_classifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy
from collections import Counter
from mutualinfo.utils import train_conformalize_test_split
from mapie.regression import SplitConformalRegressor
from mapie.classification import SplitConformalClassifier
from mapie.metrics.regression import regression_coverage_score
from mapie.metrics.classification import classification_coverage_score
import numpy as np

def split_conformal_prediction(
    X,
    y,
    model,
    alpha=0.1,
    test_size=0.2,
    cal_size=0.2,
    random_state=42
):
    """
    Aplica Split Conformal Prediction para regresión o clasificación.

    Parámetros:
    - X, y: Conjunto de datos.
    - model: Estimador sklearn (regresor o clasificador).
    - alpha: 1 - nivel de confianza.
    - test_size: Tamaño del conjunto de test.
    - cal_size: Tamaño del conjunto de calibración.
    - random_state: Semilla aleatoria.

    Devuelve:
    - y_pred: Predicción puntual.
    - y_interval/set: Intervalos de predicción (regresión) o conjunto de clases (clasificación).
    - coverage: Cobertura empírica del conjunto de test.
    """
    if not (is_regressor(model) or is_classifier(model)):
        raise ValueError("El modelo debe ser un regresor o clasificador válido de scikit-learn.")

    X_train, X_cal, X_test, y_train, y_cal, y_test = train_conformalize_test_split(
        X, y,
        train_size=1 - cal_size - test_size,
        conformalize_size=cal_size,
        test_size=test_size,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    confidence_level = 1 - alpha

    if is_regressor(model):
        scr = SplitConformalRegressor(
            estimator=model,
            confidence_level=confidence_level,
            prefit=True
        )
        scr.conformalize(X_cal, y_cal)
        y_pred, y_interval = scr.predict_interval(X_test)
        coverage = regression_coverage_score(y_test, y_interval)[0]
        return y_pred, y_interval, coverage

    elif is_classifier(model):
        scc = SplitConformalClassifier(
            estimator=model,
            confidence_level=confidence_level,
            prefit=True
        )
        scc.conformalize(X_cal, y_cal)
        y_pred, y_set = scc.predict_set(X_test)
        coverage = classification_coverage_score(y_test, y_set)[0]
        return y_pred, y_set, coverage

def encode_prediction_sets(y_pred_set, n_classes):
    """
    Codifica conjuntos de predicción multiclase como un entero único por fila
    mediante codificación binaria.

    Parámetros:
    - y_pred_set: array (n_samples, k) con etiquetas predichas por fila
    - n_classes: número total de clases

    Devuelve:
    - encoded: array (n_samples,) de valores enteros únicos por fila
    """
    binary_matrix = np.zeros((y_pred_set.shape[0], n_classes), dtype=int)
    for i, pred_set in enumerate(y_pred_set):
        for label in pred_set:
            if label < n_classes:
                binary_matrix[i, label] = 1
    powers_of_two = 1 << np.arange(n_classes)
    return binary_matrix.dot(powers_of_two)

def predict_confidence_regions(model, X, y, X_grid, alpha=0.1, random_state=42):
    """
    Aplica Split Conformal Prediction sobre una malla para clasificación.

    Parámetros:
    - model: Clasificador sklearn
    - X, y: Datos originales
    - X_grid: Datos sobre los que predecir las regiones (malla)
    - alpha: Nivel de error (1 - nivel de confianza)
    - random_state: Semilla para reproducibilidad

    Devuelve:
    - y_pred_set_mesh: Conjuntos predichos para la malla
    """
    mapie = SplitConformalClassifier(estimator=model, confidence_level=1 - alpha, prefit=False)
    X_train, X_cal, _, y_train, y_cal, _ = train_conformalize_test_split(
        X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=random_state
    )
    mapie.fit(X_train, y_train)
    mapie.conformalize(X_cal, y_cal)
    _, y_pred_set_mesh = mapie.predict_set(X_grid)
    return y_pred_set_mesh[:, :, 0] 

def estimate_mi_from_conformal_prediction_sets(
    X,
    y,
    model,
    alpha=0.1,
    test_size=0.2,
    cal_size=0.2,
    random_state=42
):
    """
    Estima la información mutua I(Y;X) usando Split Conformal Prediction sobre clasificación.

    Parámetros:
    - X, y: Datos completos.
    - model: Clasificador sklearn.
    - alpha: Nivel de error (1 - nivel de confianza).
    - test_size: Tamaño del conjunto de test.
    - cal_size: Tamaño del conjunto de calibración.
    - random_state: Semilla.

    Devuelve:
    - mi: Información mutua estimada.
    - h_y: Entropía marginal H(Y).
    - h_y_given_x: Entropía condicional H(Y|X) basada en prediction sets.
    """

    if not is_classifier(model):
        raise ValueError("Este método solo es compatible con clasificadores.")

    # Split del dataset
    X_train, X_cal, X_test, y_train, y_cal, y_test = train_conformalize_test_split(
        X, y,
        train_size=1 - cal_size - test_size,
        conformalize_size=cal_size,
        test_size=test_size,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # Obtener prediction sets con Mapie
    scc = SplitConformalClassifier(
        estimator=model,
        confidence_level=1 - alpha,
        prefit=True
    )
    scc.conformalize(X_cal, y_cal)
    _, y_pred_set = scc.predict_set(X_test)

    n_classes = len(np.unique(y))

    # Calcular H(Y)
    counts = np.array(list(Counter(y_test).values()))
    probs_y = counts / counts.sum()
    h_y = entropy(probs_y, base=2)

    # Calcular H(Y|X) a partir de prediction sets (con soporte para arrays y máscaras booleanas)
    entropies = []
    for pred in y_pred_set:
        # Asegurar que el prediction set sea una lista de índices de clase
        if isinstance(pred, np.ndarray):
            if pred.dtype == bool:
                pred = np.where(pred)[0].tolist()
            elif pred.ndim > 1:
                pred = np.array(pred).flatten().tolist()
            else:
                pred = pred.tolist()
        elif not isinstance(pred, list):
            pred = [int(pred)]

        if len(pred) == 0:
            continue  # Evita división por cero si el set está vacío

        probs = np.zeros(n_classes)
        for c in pred:
            probs[c] = 1 / len(pred)
        entropies.append(entropy(probs, base=2))

    h_y_given_x = np.mean(entropies)

    # MI estimada
    mi = h_y - h_y_given_x
    return mi, h_y, h_y_given_x

def estimate_mi_with_uncertainty(
    X,
    y,
    model,
    alphas=(0.05, 0.1, 0.2),
    test_size=0.2,
    cal_size=0.2,
    random_state=42
):
    """
    Estima la información mutua con intervalos de incertidumbre usando Conformal Prediction.

    Devuelve:
    - mi_central: estimación central con alpha medio
    - mi_low: límite inferior con alpha mayor
    - mi_high: límite superior con alpha menor
    """
    results = {}
    for alpha in alphas:
        mi, _, _ = estimate_mi_from_conformal_prediction_sets(
            X, y, model,
            alpha=alpha,
            test_size=test_size,
            cal_size=cal_size,
            random_state=random_state
        )
        results[alpha] = mi

    alpha_mid = sorted(alphas)[1]
    alpha_low = max(alphas)
    alpha_high = min(alphas)

    mi_low = results[alpha_low]
    mi_central = results[alpha_mid]
    mi_high = results[alpha_high]

    return mi_central, mi_low, mi_high

def estimate_mi_kraskov_conformal(
    X,
    y,
    model,
    alpha=0.1,
    test_size=0.2,
    cal_size=0.2,
    n_bins=10,
    random_state=42
):
    """
    Estima la información mutua I(X;Y) aplicando Conformal Prediction y propagando la incertidumbre.
    Funciona para tareas de regresión y clasificación.

    Devuelve:
    - MI estimada
    - H(Y)
    - H(Y|X)
    - coverage empírico
    """
    from sklearn.utils.multiclass import type_of_target

    is_reg = is_regressor(model)
    is_clf = is_classifier(model)
    if not (is_reg or is_clf):
        raise ValueError("El modelo debe ser un regresor o clasificador válido de sklearn.")

    X_train, X_cal, X_test, y_train, y_cal, y_test = train_conformalize_test_split(
        X, y, train_size=1 - cal_size - test_size,
        conformalize_size=cal_size,
        test_size=test_size,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    confidence_level = 1 - alpha

    if is_reg:
        scr = SplitConformalRegressor(estimator=model, confidence_level=confidence_level, prefit=True)
        scr.conformalize(X_cal, y_cal)
        y_pred, y_interval = scr.predict_interval(X_test)

        interval_lengths = np.abs(y_interval[:, 1] - y_interval[:, 0])
        eps = 1e-8
        interval_lengths = np.clip(interval_lengths, eps, None)
        h_y_given_x = np.mean(np.log2(interval_lengths))

        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        y_binned = discretizer.fit_transform(y_test.reshape(-1, 1)).astype(int).ravel()
        counts = np.array(list(Counter(y_binned).values()))
        probs_y = counts / counts.sum()
        h_y = entropy(probs_y, base=2)

        mi = h_y - h_y_given_x
        coverage = regression_coverage_score(y_test, y_interval)[0]

    elif is_clf:
        scc = SplitConformalClassifier(estimator=model, confidence_level=confidence_level, prefit=True)
        scc.conformalize(X_cal, y_cal)
        _, y_pred_set = scc.predict_set(X_test)

        n_classes = len(np.unique(y))

        entropies = []
        for pred in y_pred_set:
            if isinstance(pred, np.ndarray):
                if pred.dtype == bool:
                    pred = np.where(pred)[0].tolist()
                elif pred.ndim > 1:
                    pred = np.array(pred).flatten().tolist()
                else:
                    pred = pred.tolist()
            elif not isinstance(pred, list):
                pred = [int(pred)]
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
        coverage = classification_coverage_score(y_test, y_pred_set)[0]

    return mi, h_y, h_y_given_x, coverage

def estimate_mi_cp_radius(
    X,
    y,
    alpha=0.1,
    test_size=0.2,
    cal_size=0.2,
    k=5,
    random_state=42
):
    """
    Estima la información mutua usando KNeighbors + Conformal Prediction en lugar del radio de Kraskov.
    """
    X_train, X_cal, X_test, y_train, y_cal, y_test = train_conformalize_test_split(
        X, y, train_size=1 - cal_size - test_size,
        conformalize_size=cal_size,
        test_size=test_size,
        random_state=random_state
    )

    base_knn = KNeighborsClassifier(n_neighbors=k)
    scc = SplitConformalClassifier(
        estimator=base_knn,
        confidence_level=1 - alpha,
        prefit=False
    )
    scc.fit(X_train, y_train)
    scc.conformalize(X_cal, y_cal)
    _, y_pred_set = scc.predict_set(X_test)

    n_classes = len(np.unique(y))
    entropies = []
    for pred in y_pred_set:
        pred = list(np.where(pred)[0]) if pred.dtype == bool else pred.tolist()
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

    coverage = classification_coverage_score(y_test, y_pred_set)[0]
    return mi, h_y, h_y_given_x, coverage


