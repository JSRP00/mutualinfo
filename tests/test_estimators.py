# mutualinfo/tests/test_estimators.py

import numpy as np
from mutualinfo.estimators import (
    estimate_mi_kraskov,
    estimate_mi_kde,
    estimate_mi_histogram
)
from mutualinfo.utils import generate_dependent_data, generate_independent_data

# Semilla para reproducibilidad
SEED = 123
N = 300

def test_kraskov_mi_greater_for_dependent():
    x_dep, y_dep = generate_dependent_data(n=N, seed=SEED)
    x_ind, y_ind = generate_independent_data(n=N, seed=SEED)

    mi_dep = estimate_mi_kraskov(x_dep, y_dep)
    mi_ind = estimate_mi_kraskov(x_ind, y_ind)

    assert mi_dep > mi_ind, "Kraskov MI should be greater for dependent variables"


def test_kde_mi_greater_for_dependent():
    x_dep, y_dep = generate_dependent_data(n=N, seed=SEED)
    x_ind, y_ind = generate_independent_data(n=N, seed=SEED)

    mi_dep = estimate_mi_kde(x_dep, y_dep)
    mi_ind = estimate_mi_kde(x_ind, y_ind)

    assert mi_dep > mi_ind, "KDE MI should be greater for dependent variables"


def test_histogram_mi_greater_for_dependent():
    x_dep, y_dep = generate_dependent_data(n=N, seed=SEED)
    x_ind, y_ind = generate_independent_data(n=N, seed=SEED)

    mi_dep = estimate_mi_histogram(x_dep, y_dep, bins=15)
    mi_ind = estimate_mi_histogram(x_ind, y_ind, bins=15)

    assert mi_dep > mi_ind, "Histogram MI should be greater for dependent variables"
