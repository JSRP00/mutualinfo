from mapie.regression import MapieRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def split_conformal_regression(X, y, model=None, alpha=0.1, test_size=0.2, cal_size=0.2, random_state=42):
    """
    Aplica Split Conformal Prediction usando MAPIE para estimar intervalos de predicción.

    Parámetros
    ----------
    X : array-like
        Variables predictoras.
    y : array-like
        Variable objetivo.
    model : sklearn regressor or None
        Modelo base de regresión. Si es None, se usará RandomForestRegressor.
    alpha : float
        Nivel de error (1 - nivel de confianza).
    test_size : float
        Proporción del conjunto de test.
    cal_size : float
        Proporción del conjunto de calibración.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    y_pred : array
        Predicciones puntuales.
    y_interval : array
        Intervalos de predicción (inferior y superior).
    coverage : float
        Proporción de valores reales de test dentro del intervalo.
    """
    # Modelo por defecto si no se pasa uno
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    # División del conjunto en test/cal/train
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    cal_fraction = cal_size / (1 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=cal_fraction, random_state=random_state)

    # Concatenar train + cal para entrenar modelo único
    X_combined = np.concatenate([X_train, X_cal])
    y_combined = np.concatenate([y_train, y_cal])

    # Inicializar MAPIE con método "naive" (split conformal)
    mapie = MapieRegressor(estimator=model, method="naive", cv="split", random_state=random_state)
    mapie.fit(X_combined, y_combined)

    # Predecir con intervalos
    y_pred, y_interval = mapie.predict(X_test, alpha=alpha)

    # Calcular cobertura
    lower, upper = y_interval[:, 0], y_interval[:, 1]
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    return y_pred, y_interval, coverage
