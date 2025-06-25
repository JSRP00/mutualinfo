# /mutualinfo/mutualinfo/uncertainty/conformal.py

from mapie.regression import SplitConformalRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def split_conformal_regression(X, y, alpha=0.1, test_size=0.2, cal_size=0.2, random_state=42):
    """
    Aplica Split Conformal Prediction para regresión con intervalos de predicción.
    """
    # División train/cal/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    cal_fraction = cal_size / (1 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=cal_fraction, random_state=random_state)

    # Concatenar train y cal
    X_combined = np.concatenate([X_train, X_cal])
    y_combined = np.concatenate([y_train, y_cal])

    # Crear modelo nuevo sin entrenar
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    # Inicializar y entrenar SCR
    scr = SplitConformalRegressor(estimator=model, prefit=False)
    scr.fit(X_combined, y_combined)

    # Predicción con intervalos
    y_pred, y_interval = scr.predict(X_test, return_pred_int=True, alpha=alpha)

    # Coverage
    lower, upper = y_interval[:, 0], y_interval[:, 1]
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    return y_pred, y_interval, coverage
