"""
formative3_utils
================
Utility package for Formative 3.

Module layout
-------------
data_loading.py   — load_education_data(), load_imdb_data()
distribution.py   — bivariate normal PDF math  (James)
bayesian.py       — Bayes' Theorem math        (Favor)
visualization.py  — ALL plot functions         (Parts 1, 2, 4)
gradient_descent.py — GD implementation        (Ryan)
"""

# ── Data loading ──────────────────────────────────────────────────────────────
from formative3_utils.data_loading import (
    load_education_data,
    load_imdb_data,
)

# ── Part 1 math (James) ───────────────────────────────────────────────────────
from formative3_utils.distribution import (
    compute_bivariate_pdf,
    compute_distribution_parameters,
)

# ── Part 2 math (Favor) ───────────────────────────────────────────────────────
from formative3_utils.bayesian import (
    prepare_dataframe,
    calculate_prior,
    calculate_likelihood,
    calculate_marginal,
    calculate_posterior,
    analyze_keywords,
)

# ── All visualisations (Parts 1, 2, 4) ───────────────────────────────────────
from formative3_utils.visualization import (
    # Part 1
    plot_contour,
    plot_3d_surface,
    # Part 2
    plot_sentiment_distribution,
    plot_prior_vs_posterior,
    plot_probability_heatmap,
    # Part 4
    plot_gd_convergence,
    plot_gd_final_line,
    plot_gd_compare_methods,
)

# ── Part 4 GD (Ryan) ──────────────────────────────────────────────────────────
from formative3_utils.gradient_descent import (
    mse_cost,
    gradient_m,
    gradient_b,
    run_manual_gradient_descent,
    run_scipy_gradient_descent,
    compare_results,
    X_POINTS, Y_POINTS,
    M_INIT, B_INIT, ALPHA,
)

__all__ = [
    # data
    "load_education_data", "load_imdb_data",
    # Part 1 math
    "compute_bivariate_pdf", "compute_distribution_parameters",
    # Part 2 math
    "prepare_dataframe", "calculate_prior", "calculate_likelihood",
    "calculate_marginal", "calculate_posterior", "analyze_keywords",
    # visualisations
    "plot_contour", "plot_3d_surface",
    "plot_sentiment_distribution", "plot_prior_vs_posterior", "plot_probability_heatmap",
    "plot_gd_convergence", "plot_gd_final_line", "plot_gd_compare_methods",
    # Part 4 GD
    "mse_cost", "gradient_m", "gradient_b",
    "run_manual_gradient_descent", "run_scipy_gradient_descent", "compare_results",
    "X_POINTS", "Y_POINTS", "M_INIT", "B_INIT", "ALPHA",
]