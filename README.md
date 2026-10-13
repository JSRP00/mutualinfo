# mutualinfo

[![tests](https://github.com/JSRP00/mutualinfo/actions/workflows/tests.yml/badge.svg)](https://github.com/JSRP00/mutualinfo/actions/workflows/tests.yml)

Mutual information estimation and uncertainty quantification in Python.

Developed as my MSc thesis in Big Data Science at the University of Navarra, proposed and supervised by the Institute of Data Science and Artificial Intelligence (DATAI).

## Features

**Estimators of I(X;Y)**
- Kraskov–Stögbauer–Grassberger (k-nearest neighbours)
- Kernel density estimation (KDE)
- Histogram (discretisation)

**Uncertainty quantification**
- Bootstrap confidence intervals for any estimator
- Conformal prediction: mutual information estimated from conformal prediction sets and intervals, built on [MAPIE](https://github.com/scikit-learn-contrib/MAPIE)

**Utilities**
- Synthetic data generators with linear and non-linear dependence
- Plotting helpers for prediction intervals and confidence regions

## Installation

```bash
git clone https://github.com/JSRP00/mutualinfo.git
cd mutualinfo
pip install -e ".[test]"
```

Requires Python 3.9 or later.

## Quick start

```python
from mutualinfo import estimate_mi_kraskov, estimate_mi_kde, estimate_mi_histogram, bootstrap_ci
from mutualinfo.utils import generate_dependent_data

x, y = generate_dependent_data(n_samples=500, relation="nonlinear", random_state=0)

estimate_mi_kraskov(x, y, k=3)
estimate_mi_kde(x, y)
estimate_mi_histogram(x, y, bins=15)

# 95% bootstrap confidence interval
lower, upper = bootstrap_ci(estimate_mi_histogram, x, y, n_bootstraps=200, seed=0, bins=15)
```

More examples are in [`examples/`](examples), including a notebook that runs the full set of experiments.

## Note on bootstrap with k-NN estimators

Resampling with replacement creates duplicate points, which have zero distance to each other. This biases k-NN estimators such as Kraskov upwards, so their bootstrap intervals can sit above the original estimate. The histogram and KDE estimators do not have this problem.

## Tests

```bash
pytest
```

## License

MIT
