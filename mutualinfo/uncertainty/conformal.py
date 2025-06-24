# mutualinfo/uncertainty/conformal.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# MAPIE para regresión
from mapie.regression import MapieRegressor

# CREPES para clasificación
from crepes import ConformalClassifier


def split_data(x, y, train_size=0.6, cal_size=0.2, test_size=0.2, seed=42):
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size=train_size, random_state=seed)
    cal_ratio = cal_size / (cal_size + test_size)
    x_cal, x_test, y_cal, y_test = train_test_split(x_temp, y_temp, train_size=cal_ratio, random_state=seed)
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def apply_conformal_regression(x_train, y_train, x_cal, y_cal, x_test, y_test, alpha=0.1):
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    mapie = MapieRegressor(estimator=base_model, method="naive", cv="prefit", agg_function="mean")

    base_model.fit(x_train, y_train)
    mapie.fit(x_cal, y_cal)
    y_pred, y_pis = mapie.predict(x_test, alpha=alpha)
    return y_pred, y_pis


def apply_conformal_classification(x_train, y_train, x_cal, y_cal, x_test, y_test, alpha=0.1):
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    conformal = ConformalClassifier(model=base_model)

    conformal.fit(x_train, y_train, x_cal, y_cal)
    prediction_sets = conformal.predict(x_test, alpha=alpha)

    coverage = np.mean([y_test[i] in prediction_sets[i] for i in range(len(y_test))])
    return prediction_sets, coverage
