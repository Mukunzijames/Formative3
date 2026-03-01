"""
Part 1: Bivariate Normal Distribution
Author: James Mukunzi

Implements the bivariate normal PDF from scratch — no scipy.stats used.
Provides contour plot and 3D surface visualisations.

Formula
-------
f(x,y) = 1/(2π σx σy √(1−ρ²))
         × exp(−1/(2(1−ρ²)) × [(x−μx)²/σx² − 2ρ(x−μx)(y−μy)/(σxσy) + (y−μy)²/σy²])
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection


def compute_bivariate_pdf(X, Y, mu, sigma):
    """
    Evaluate the Bivariate Normal PDF across a meshgrid.
    Implemented from scratch — no scipy.stats.

    Parameters
    ----------
    X, Y  : np.ndarray        Meshgrid arrays of equal shape.
    mu    : array-like (2,)   Mean vector [mu_x, mu_y].
    sigma : np.ndarray (2,2)  Covariance matrix [[sx², cov], [cov, sy²]].

    Returns
    -------
    Z : np.ndarray  PDF values, same shape as X and Y.

    Raises
    ------
    ValueError  If sigma_x or sigma_y <= 0, or if |rho| >= 1.
    """
    # Unpack the mean vector into scalar values
    mu_x, mu_y = float(mu[0]), float(mu[1])

    # Extract standard deviations from the diagonal of the covariance matrix
    sigma_x = np.sqrt(float(sigma[0, 0]))   # std of x
    sigma_y = np.sqrt(float(sigma[1, 1]))   # std of y
    cov     = float(sigma[0, 1])            # covariance between x and y

    # Both standard deviations must be strictly positive
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValueError(
            f"Standard deviations must be > 0. "
            f"Got sigma_x={sigma_x:.4f}, sigma_y={sigma_y:.4f}."
        )

    # Compute Pearson correlation from the covariance matrix
    rho = cov / (sigma_x * sigma_y)

    # |rho| must be strictly less than 1; otherwise sqrt(1-rho²) = 0 → division by zero
    if abs(rho) >= 1.0:
        raise ValueError(
            f"|rho| = {abs(rho):.6f} >= 1. This pair has near-perfect correlation "
            "which causes division by zero in sqrt(1-rho²). Choose a different pair."
        )

    # Standardise: shift by mean and scale by standard deviation
    z_x = (X - mu_x) / sigma_x
    z_y = (Y - mu_y) / sigma_y

    # The (1−ρ²) term appears in both the exponent and the normalisation constant
    one_minus_rho2 = 1.0 - rho ** 2

    # Mahalanobis-like exponent — measures distance from the mean in 2D
    exponent = (z_x**2 - 2.0 * rho * z_x * z_y + z_y**2) / one_minus_rho2

    # Normalisation constant — ensures the PDF integrates to 1 over all (x, y)
    norm = 1.0 / (2.0 * np.pi * sigma_x * sigma_y * np.sqrt(one_minus_rho2))

    return norm * np.exp(-0.5 * exponent)


def compute_distribution_parameters(data, col_x, col_y):
    """
    Compute mean vector, covariance matrix, and Pearson correlation for two columns.

    Uses population statistics (ddof=0) — consistent with James's original implementation.

    Parameters
    ----------
    data  : pd.DataFrame  Cleaned data with NaNs in col_x/col_y already dropped.
    col_x : str           Name of the x-variable column.
    col_y : str           Name of the y-variable column.

    Returns
    -------
    mu         : np.ndarray (2,)    [mu_x, mu_y]
    cov_matrix : np.ndarray (2, 2)
    rho        : float              Pearson correlation coefficient
    """
    # Extract raw arrays from the dataframe columns
    x = data[col_x].values.astype(float)
    y = data[col_y].values.astype(float)

    # Compute population means
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    # Population standard deviations (ddof=0 = divide by N, not N-1)
    sigma_x    = np.std(x, ddof=0)
    sigma_y    = np.std(y, ddof=0)

    # Population covariance: E[(x − μx)(y − μy)]
    covariance = np.mean((x - mu_x) * (y - mu_y))

    # Assemble the mean vector [μx, μy]
    mu = np.array([mu_x, mu_y])

    # Assemble the 2×2 symmetric covariance matrix
    cov_matrix = np.array([
        [sigma_x ** 2, covariance],
        [covariance,   sigma_y ** 2],
    ])

    # Pearson ρ normalises covariance to the range [−1, 1]
    rho = covariance / (sigma_x * sigma_y)

    return mu, cov_matrix, rho


# Note: plot functions (contour plot, 3D surface) have moved to visualization.py
# Import from there: from formative3_utils.visualization import plot_contour, plot_3d_surface