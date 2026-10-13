import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
from collections import Counter
from mapie.classification import SplitConformalClassifier
from mapie.regression import SplitConformalRegressor
from mutualinfo.utils import train_conformalize_test_split

def regression_coverage_score_manual(y_true, intervals):
    """
    Computes the empirical coverage: the proportion of times y_true falls inside the interval.
    """
    lower_bounds = intervals[:, 0]
    upper_bounds = intervals[:, 1]
    return np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))

def estimate_mi_kraskov_conformal(
    X,
    y,
    model=None,
    alpha=0.1,
    task="auto",
    n_bins=5,
    test_size=0.2,
    cal_size=0.2,
    random_state=42
):
    """
    Estimates the mutual information I(Y;X) using conformal prediction with KNN models, for both classification and regression.

    Parameters:
    - X, y: input data (arrays).
    - model: base model (KNN classifier or regressor).
    - alpha: error level (1 - confidence).
    - task: "classification", "regression" or "auto".
    - n_bins: number of bins when y is discretised (for the entropy).
    - test_size, cal_size: split proportions.
    - random_state: random seed.

    Returns:
    - mi: estimated mutual information.
    - h_y: marginal entropy of Y.
    - h_y_given_x: estimated conditional entropy.
    - coverage: empirical coverage achieved.
    """

    # Infer the task type when not given
    if task == "auto":
        task = "regression" if np.issubdtype(y.dtype, np.floating) else "classification"

    # Discretise y for regression (to compute H(Y) and H(Y|X))
    if task == "regression":
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        y_disc = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    else:
        y_disc = y

    # Train / calibration / test split
    X_train, X_cal, X_test, y_train, y_cal, y_test = train_conformalize_test_split(
        X, y_disc,
        train_size=1 - cal_size - test_size,
        conformalize_size=cal_size,
        test_size=test_size,
        random_state=random_state
    )

    # Modelo base por defecto si no se especifica
    if model is None:
        model = KNeighborsClassifier(n_neighbors=5) if task == "classification" else KNeighborsRegressor(n_neighbors=5)

    model.fit(X_train, y_train)
    confidence = 1 - alpha

    if task == "classification":
        # Conformal prediction for classification
        cp = SplitConformalClassifier(estimator=model, confidence_level=confidence, prefit=True)
        cp.conformalize(X_cal, y_cal)
        _, y_pred_set = cp.predict_set(X_test)

        n_classes = len(np.unique(y_disc))
        entropies = []

        for pred in y_pred_set:
            if isinstance(pred, np.ndarray) and pred.dtype == bool:
                pred = np.where(pred)[0].tolist()
            elif isinstance(pred, np.ndarray):
                pred = pred.tolist()
            if len(pred) == 0:
                continue
            probs = np.zeros(n_classes)
            for c in pred:
                probs[c] = 1 / len(pred)
            entropies.append(entropy(probs, base=2))

        h_y_given_x = np.mean(entropies)

        counts = np.array(list(Counter(y_test).values()))
        probs_y = counts / counts.sum()
        h_y = entropy(probs_y, base=2)

        mi = h_y - h_y_given_x
        coverage = cp.coverage_score(X_test, y_test)

    else:
        # Conformal prediction for regression
        cp = SplitConformalRegressor(estimator=model, confidence_level=confidence, prefit=True)
        cp.conformalize(X_cal, y_cal)
        _, intervals = cp.predict_interval(X_test)

        lengths = intervals[:, 1] - intervals[:, 0]
        pseudo_entropy = np.log(lengths + 1e-8)  # evitar log(0)
        h_y_given_x = np.mean(pseudo_entropy)

        # Marginal entropy from the empirical distribution
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        y_test_binned = discretizer.fit_transform(y_test.reshape(-1, 1)).ravel()
        counts = np.array(list(Counter(y_test_binned).values()))
        probs_y = counts / counts.sum()
        h_y = entropy(probs_y, base=2)

        mi = h_y - h_y_given_x
        coverage = regression_coverage_score_manual(y_test, intervals)

        return mi, h_y, h_y_given_x, coverage
