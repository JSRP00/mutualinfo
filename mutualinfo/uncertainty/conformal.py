from mapie.regression import SplitConformalRegressor
from sklearn.model_selection import train_test_split
import numpy as np

def split_conformal_regression(X, y, model, alpha=0.1, test_size=0.2, cal_size=0.2, random_state=42):
    """
    Aplica Split Conformal Prediction para regresión con intervalos de predicción.

    Parámetros
    ----------
    X : array-like
        Matriz de características (features).
    y : array-like
        Vector de etiquetas (target).
    model : objeto sklearn
        Regressor compatible con .fit() y .predict().
    alpha : float, optional
        Nivel de significación (por defecto 0.1 → 90% intervalo).
    test_size : float, optional
        Proporción del conjunto de test.
    cal_size : float, optional
        Proporción del conjunto de calibración.
    random_state : int, optional
        Semilla para reproducibilidad.

    Retorna
    -------
    y_pred : np.ndarray
        Predicciones puntuales.
    y_interval : np.ndarray
        Intervalos de predicción (inferior, superior).
    coverage : float
        Porcentaje de observaciones reales dentro del intervalo.
    """
    # Dividir en test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Dividir en train y cal (del 80%)
    cal_fraction = cal_size / (1 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=cal_fraction, random_state=random_state
    )

    # Entrenar modelo base
    model.fit(X_train, y_train)

    # Aplicar conformal prediction (entrenar SCR con conjunto de calibración)
    scr = SplitConformalRegressor(estimator=model)
    scr.fit(X_cal, y_cal)

    # Predecir con intervalos
    y_pred, y_interval = scr.predict(X_test, alpha=alpha, return_prediction_interval=True)

    # Calcular cobertura empírica
    lower, upper = y_interval[:, 0], y_interval[:, 1]
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    return y_pred, y_interval, coverage
