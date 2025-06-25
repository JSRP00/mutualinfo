from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mapie.regression import SplitConformalRegressor
import numpy as np

def split_conformal_regression(X, y, alpha=0.1, test_size=0.2, cal_size=0.2, random_state=42):
    """
    Aplica Split Conformal Prediction para regresión con intervalos de predicción.

    Parámetros
    ----------
    X : array-like
        Variables predictoras.
    y : array-like
        Variable objetivo.
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
    # División en test/cal/train
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    cal_fraction = cal_size / (1 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=cal_fraction, random_state=random_state)

    # Modelo base
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    # SCR sin prefit
    scr = SplitConformalRegressor(estimator=model)
    scr.fit(X_train, y_train)
    scr.conformalize(X_cal, y_cal)

    # Predicciones con intervalos
    y_pred_full = scr.predict(X_test, alpha=alpha)  # (n, 3) -> [pred, lower, upper]
    y_pred = y_pred_full[:, 0]
    y_lower = y_pred_full[:, 1]
    y_upper = y_pred_full[:, 2]

    # Coverage
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    y_interval = np.column_stack((y_lower, y_upper))

    return y_pred, y_interval, coverage
