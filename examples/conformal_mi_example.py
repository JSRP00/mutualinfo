from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from mutualinfo.uncertainty import estimate_mi_from_conformal_prediction_sets

# Datos sintéticos
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_classes=3,
    n_informative=3,
    random_state=42
)

# Modelo base
model = KNeighborsClassifier(n_neighbors=5)

# Estimar MI con conformal prediction
mi, h_y, h_y_given_x = estimate_mi_from_conformal_prediction_sets(
    X, y, model, alpha=0.1, test_size=0.2, cal_size=0.2
)

# Resultados
print("----- Estimación de Información Mutua con CP -----")
print(f"H(Y):          {h_y:.4f}")
print(f"H(Y|X):        {h_y_given_x:.4f}")
print(f"I(Y; X):       {mi:.4f}")
print("--------------------------------------------------")
