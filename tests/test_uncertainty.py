# tests/test_uncertainty.py

import numpy as np
from mutualinfo.estimators import estimate_mi_kraskov
from mutualinfo.uncertainty import bootstrap_ci, conformal_ci
from mutualinfo.utils import generate_dependent_data

SEED = 123
N = 300

def test_bootstrap_interval_contains_mi():
    x, y = generate_dependent_data(n=N, seed=SEED)
    mi = estimate_mi_kraskov(x, y)
    lower, upper = bootstrap_ci(estimate_mi_kraskov, x, y, n_bootstraps=100, seed=SEED)

    assert lower < upper, "Bootstrap: lower bound should be less than upper bound"
    assert lower <= mi <= upper, "Bootstrap: MI estimate should lie within the confidence interval"

def test_conformal_interval_contains_mi():
    x, y = generate_dependent_data(n=N, seed=SEED)
    mi = estimate_mi_kraskov(x, y)
    lower, upper = conformal_ci(estimate_mi_kraskov, x, y, n_samples=100, seed=SEED)

    assert lower < upper, "Conformal: lower bound should be less than upper bound"
    assert lower <= mi <= upper, "Conformal: MI estimate should lie within the prediction interval"
