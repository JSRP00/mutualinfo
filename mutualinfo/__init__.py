# mutualinfo/__init__.py

from .estimators import (
    estimate_mi_kraskov,
    estimate_mi_kde,
    estimate_mi_histogram
)

from .uncertainty import (
    bootstrap_ci,
    split_conformal_prediction,
    encode_prediction_sets,
    predict_confidence_regions,
    estimate_mi_from_conformal_prediction_sets,
    estimate_mi_with_uncertainty
)

__all__ = [
    "estimate_mi_kraskov",
    "estimate_mi_kde",
    "estimate_mi_histogram",
    "bootstrap_ci",
    "split_conformal_prediction",
    "encode_prediction_sets",
    "predict_confidence_regions",
    "estimate_mi_from_conformal_prediction_sets",
    "estimate_mi_with_uncertainty"
]
