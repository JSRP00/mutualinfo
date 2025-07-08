import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from mutualinfo.uncertainty import estimate_mi_from_conformal_prediction_sets

def test_mi_with_conformal_prediction_sets():
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_classes=3,
        n_informative=3,
        random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=5)
    mi, h_y, h_y_given_x = estimate_mi_from_conformal_prediction_sets(
        X, y, model, alpha=0.1, test_size=0.2, cal_size=0.2
    )

    # Comprobaciones básicas
    assert 0 <= h_y_given_x <= h_y <= np.log2(3) + 0.1
    assert 0 <= mi <= h_y + 1e-6  # Tolerancia numérica
    print("Test passed: Información mutua estimada correctamente con CP.")
