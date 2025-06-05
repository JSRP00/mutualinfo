# examples/demo_estimators.py

import numpy as np
from mutualinfo.utils import (
    generate_dependent_data,
    generate_independent_data
)

from mutualinfo import (
    estimate_mi_kraskov,
    estimate_mi_kde,
    estimate_mi_histogram,
    bootstrap_ci,
    conformal_ci
)

def print_results(name, mi, ci_bootstrap, ci_conformal):
    print(f"\n{name}")
    print(f"  MI estimada         : {mi:.4f}")
    print(f"  Bootstrap IC 95%    : [{ci_bootstrap[0]:.4f}, {ci_bootstrap[1]:.4f}]")
    print(f"  Conformal IC 95%    : [{ci_conformal[0]:.4f}, {ci_conformal[1]:.4f}]")


def main():
    n = 500
    seed = 42

    # Datos dependientes
    x_dep, y_dep = generate_dependent_data(n=n, noise=0.2, seed=seed)

    # Datos independientes
    x_indep, y_indep = generate_independent_data(n=n, seed=seed)

    print("===== DATOS DEPENDIENTES =====")

    # Estimador Kraskov
    mi_k = estimate_mi_kraskov(x_dep, y_dep)
    ci_k_boot = bootstrap_ci(estimate_mi_kraskov, x_dep, y_dep, n_bootstraps=200, seed=seed)
    ci_k_conf = conformal_ci(estimate_mi_kraskov, x_dep, y_dep, n_samples=100, seed=seed)
    print_results("Kraskov", mi_k, ci_k_boot, ci_k_conf)

    # Estimador KDE
    mi_kde = estimate_mi_kde(x_dep, y_dep)
    ci_kde_boot = bootstrap_ci(estimate_mi_kde, x_dep, y_dep, n_bootstraps=200, seed=seed)
    ci_kde_conf = conformal_ci(estimate_mi_kde, x_dep, y_dep, n_samples=100, seed=seed)
    print_results("KDE", mi_kde, ci_kde_boot, ci_kde_conf)

    # Estimador Histogramas
    mi_hist = estimate_mi_histogram(x_dep, y_dep, bins=20)
    ci_hist_boot = bootstrap_ci(estimate_mi_histogram, x_dep, y_dep, n_bootstraps=200, seed=seed, bins=20)
    ci_hist_conf = conformal_ci(estimate_mi_histogram, x_dep, y_dep, n_samples=100, seed=seed, bins=20)
    print_results("Histogramas", mi_hist, ci_hist_boot, ci_hist_conf)

    print("\n===== DATOS INDEPENDIENTES =====")

    mi_k_ind = estimate_mi_kraskov(x_indep, y_indep)
    ci_k_boot_ind = bootstrap_ci(estimate_mi_kraskov, x_indep, y_indep, n_bootstraps=200, seed=seed)
    ci_k_conf_ind = conformal_ci(estimate_mi_kraskov, x_indep, y_indep, n_samples=100, seed=seed)
    print_results("Kraskov (independientes)", mi_k_ind, ci_k_boot_ind, ci_k_conf_ind)

if __name__ == "__main__":
    main()
