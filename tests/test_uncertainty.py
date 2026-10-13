# tests/test_uncertainty.py

import numpy as np
from mutualinfo.estimators import estimate_mi_histogram
from mutualinfo.uncertainty import bootstrap_ci, estimate_mi_kraskov_conformal
from mutualinfo.utils import generate_dependent_data

SEED = 123
N = 300

def test_bootstrap_interval_contains_mi():
    x, y = generate_dependent_data(n_samples=N, random_state=SEED)
    # Histogram estimator: kNN estimators such as Kraskov are biased upwards under
    # bootstrap resampling, because resampled duplicates have zero neighbour distance.
    mi = estimate_mi_histogram(x, y)
    lower, upper = bootstrap_ci(estimate_mi_histogram, x, y, n_bootstraps=200, seed=SEED)

    assert lower < upper, "Bootstrap: lower bound should be less than upper bound"
    assert lower <= mi <= upper, "Bootstrap: MI estimate should lie within the confidence interval"

def test_kraskov_conformal_returns_valid_quantities():
    x, y = generate_dependent_data(n_samples=N, random_state=SEED)
    mi, h_y, h_y_given_x, coverage = estimate_mi_kraskov_conformal(x, y, alpha=0.1, random_state=SEED)

    assert 0 <= h_y_given_x <= h_y + 1e-6, "Conditional entropy should not exceed marginal entropy"
    assert mi >= -1e-6, "MI estimate should be non-negative"
    assert 0 <= coverage <= 1, "Coverage should be a proportion"
