from mapie.regression import SplitConformalRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

     # 1. División en train/cal/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    cal_fraction = cal_size / (1 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=cal_fraction, random_state=random_state
    )

    # 2. Concatenar train + cal
    X_combined = np.concatenate([X_train, X_cal])
    y_combined = np.concatenate([y_train, y_cal])

    # 3. Modelo NUEVO sin entrenar (clave)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    # 4. Inicializar y entrenar SplitConformalRegressor (¡NO usar prefit!)
    scr = SplitConformalRegressor(estimator=model)
    scr.fit(X_combined, y_combined)

    # 5. Predecir intervalos sobre el test
    preds = scr.predict(X_test, alpha=alpha)

    # 6. Separar predicciones e intervalos
    y_pred = preds[:, 0]
    y_lower = preds[:, 1]
    y_upper = preds[:, 2]

    # 7. Calcular cobertura
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

    return y_pred, np.stack([y_lower, y_upper], axis=1), coverage
