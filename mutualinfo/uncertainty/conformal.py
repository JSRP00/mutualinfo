# mutualinfo/uncertainty/conformal.py

from mapie.regression import SplitConformalRegressor
from sklearn.model_selection import train_test_split
import numpy as np

def split_conformal_regression(X, y, model, alpha=0.1, test_size=0.2, cal_size=0.2, random_state=42):
    """
    Aplica Split Conformal Prediction para regresión con intervalos de predicción.

    Parámetros
    ----------
    X : array-like
        Variables predictoras.
    y : array-like
        Variable objetivo.
    model : sklearn regressor
        Modelo base de regresión.
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

    # División train/cal/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    cal_fraction = cal_size / (1 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=cal_fraction, random_state=random_state)

    # Entrenar modelo base
    model.fit(X_train, y_train)

    # Aplicar Split Conformal Regressor (prefit=True porque el modelo ya está entrenado)
    scr = SplitConformalRegressor(estimator=model, alpha=alpha, cv="prefit")
    scr.fit(X_cal, y_cal)

    # Predicción
    y_pred, y_interval = scr.predict(X_test, return_pred_int=True)

    # Calcular coverage
    lower, upper = y_interval[:, 0], y_interval[:, 1]
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    return y_pred, y_interval, coverage
