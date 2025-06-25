# mutualinfo/uncertainty/conformal.py

from sklearn.base import is_regressor, is_classifier
from mapie.utils import train_conformalize_test_split
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
    # Verificación temprana del tipo de modelo
    if not (is_regressor(model) or is_classifier(model)):
        raise ValueError("El modelo debe ser un regresor o clasificador válido de scikit-learn.")

    # División de datos
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
