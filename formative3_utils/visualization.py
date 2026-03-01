"""
Shared Visualisation Utilities
================================
All plot functions for Formative 3 live in this single file.
The notebook imports from here; the math functions stay in their own modules.

Part 1  (James)  — plot_contour(), plot_3d_surface()
Part 2  (Favor)  — plot_sentiment_distribution(), plot_prior_vs_posterior(),
                   plot_probability_heatmap()
Part 4  (Ryan)   — plot_gd_convergence(), plot_gd_final_line(), plot_gd_compare_methods()

Design principle
----------------
Every function receives exactly the data it needs as arguments.
The caller (notebook) always does plt.show() after calling these functions.
This keeps each function testable in isolation and the notebook cells clean.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Bivariate Normal Distribution  (James Mukunzi)
# ══════════════════════════════════════════════════════════════════════════════

def plot_contour(X_grid, Y_grid, Z_grid, x_data, y_data,
                 pdf_values, mu1, mu2, col_x, col_y, rho):
    """
    Filled contour plot of the bivariate normal PDF.
    Reproduces James's original contour visualisation from his notebook.

    Layers (drawn in order so nothing is hidden)
    --------------------------------------------
    1. Filled colour regions  — 20 PDF levels, viridis colourmap
    2. Black contour outlines — 10 levels, 30% opacity
    3. Scatter of actual data points coloured by their own PDF value
    4. Red star at the distribution mean

    Parameters
    ----------
    X_grid, Y_grid : np.ndarray  100×100 meshgrid (from np.meshgrid).
    Z_grid         : np.ndarray  PDF values on the grid, same shape as X_grid.
    x_data, y_data : np.ndarray  Raw column arrays (after dropna).
    pdf_values     : np.ndarray  PDF evaluated at each data point — used for dot colour.
    mu1, mu2       : float       Mean of x and y (placed as the red star).
    col_x, col_y   : str         Column names — used as axis labels.
    rho            : float       Pearson correlation — shown in the title.
    """
    plt.figure(figsize=(10, 8))

    # Layer 1: filled contour regions — brighter = higher probability density
    plt.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='viridis')

    # Layer 2: black outlines so each density ring is clearly visible
    plt.contour(X_grid, Y_grid, Z_grid, levels=10, colors='black', alpha=0.3)

    # Layer 3: scatter the actual country data points, coloured by PDF value
    # This shows which countries are in the high-density core vs the tails
    plt.scatter(x_data, y_data, c=pdf_values, s=30, cmap='viridis',
                edgecolors='black', alpha=0.6)

    # Layer 4: red star marks the mean — the most probable (x, y) combination
    plt.plot(mu1, mu2, 'r*', markersize=20, label='Mean')

    plt.colorbar(label='PDF')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Bivariate Normal Distribution - Contour Plot\nρ = {rho:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_3d_surface(X_grid, Y_grid, Z_grid, col_x, col_y, rho):
    """
    3D surface plot of the bivariate normal PDF.
    Reproduces James's original 3D visualisation from his notebook.

    The height at every (x, y) point equals the probability density.
    Contours are projected onto z = 0 so the viewer can read the shape
    from both the top-down and side-on perspectives simultaneously.

    Parameters
    ----------
    X_grid, Y_grid : np.ndarray  100×100 meshgrid.
    Z_grid         : np.ndarray  PDF values on the grid.
    col_x, col_y   : str         Column names for axis labels.
    rho            : float       Correlation shown in the title.
    """
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Main surface — height directly encodes probability density
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8)

    # Project the contours onto the floor (z=0) — matches the 2D contour plot above
    ax.contour(X_grid, Y_grid, Z_grid, zdir='z', offset=0, cmap='viridis', alpha=0.5)

    ax.set_xlabel(col_x, labelpad=10)
    ax.set_ylabel(col_y, labelpad=10)
    ax.set_zlabel('PDF')
    ax.set_title(f'3D Bivariate Normal Surface\nρ = {rho:.4f}')

    # 25° elevation, 45° rotation gives a clear diagonal view of the narrow ridge
    ax.view_init(elev=25, azim=45)

    plt.colorbar(surf, shrink=0.5, label='PDF')
    plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Bayesian Probability  (Favor)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_distribution(sentiment_counts, total_count):
    """
    Bar chart showing the positive / negative class distribution.
    Reproduces Favor's sentiment distribution chart.

    Parameters
    ----------
    sentiment_counts : pd.Series  value_counts() output from sentiment_clean column.
    total_count      : int        Total number of reviews (for percentage labels).
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Green for positive, red for negative — matches Favor's original colour scheme
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values,
                  color=['#2ECC71', '#E74C3C'], alpha=0.85, edgecolor='black')

    # Annotate each bar with its count and percentage of total reviews
    for bar, val in zip(bars, sentiment_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f'{val:,}  ({val / total_count:.0%})',
                ha='center', fontsize=10)

    ax.set_title('IMDb Dataset — Sentiment Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Reviews')
    ax.set_ylim(0, sentiment_counts.max() * 1.15)
    plt.tight_layout()


def plot_prior_vs_posterior(results_df, target_sentiment="negative"):
    """
    Grouped bar chart comparing prior vs posterior for every keyword.
    Reproduces Favor's prior vs posterior bar chart.

    Reading guide
    -------------
    Blue bars  = prior (same for all keywords — baseline before seeing any word)
    Green bars = positive-indicator keywords  (posterior should DROP below prior)
    Red bars   = negative-indicator keywords  (posterior should RISE above prior)
    Dashed 50% line = decision boundary: above this → classify as negative

    Parameters
    ----------
    results_df       : pd.DataFrame  Output of analyze_keywords() from bayesian.py.
                                     Must have columns: 'Keyword', 'Type',
                                     'P(Negative)', 'P(Negative|keyword)'.
    target_sentiment : str           Sentiment class name (default 'negative').
    """
    t         = target_sentiment.capitalize()
    prior_col = f"P({t})"
    post_col  = f"P({t}|keyword)"

    keywords   = results_df["Keyword"].tolist()
    prior_val  = float(results_df[prior_col].iloc[0])  # same for every keyword
    posteriors = results_df[post_col].tolist()
    types      = results_df["Type"].tolist()

    x     = np.arange(len(keywords))
    width = 0.35

    # Red for negative indicators, green for positive indicators
    bar_colours = ["#E74C3C" if tp == "Negative Indicator" else "#2ECC71"
                   for tp in types]

    fig, ax = plt.subplots(figsize=(13, 6))

    # Prior bars — same height for every keyword (our baseline belief)
    ax.bar(x - width / 2, [prior_val] * len(keywords), width,
           label=f"Prior  {prior_col}", color="steelblue", alpha=0.75)

    # Posterior bars — updated belief after observing this keyword
    bars = ax.bar(x + width / 2, posteriors, width,
                  label=f"Posterior  {post_col}", color=bar_colours, alpha=0.85)

    # Annotate each posterior bar with its exact value
    for bar, val in zip(bars, posteriors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # 50% classification boundary
    ax.axhline(0.50, color="black", linestyle="--", linewidth=1.2,
               label="50% decision boundary")

    ax.set_xticks(x)
    ax.set_xticklabels(keywords, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(
        f"Bayesian Sentiment Analysis — Prior vs Posterior\n"
        f"Target: P({t} | keyword)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()


def plot_probability_heatmap(results_df, target_sentiment="negative"):
    """
    Heatmap of all four Bayesian probabilities for every keyword.
    Reproduces Favor's probability heatmap.

    Warmer colours = higher probability values.

    Column guide
    ------------
    P(Negative)         — prior: identical for all rows (baseline)
    P(keyword)          — marginal: how common the word is overall
    P(keyword|Negative) — likelihood: how often it appears in negative reviews
    P(Negative|keyword) — posterior: the final Bayes result (most important)

    Parameters
    ----------
    results_df       : pd.DataFrame  Output of analyze_keywords() from bayesian.py.
    target_sentiment : str
    """
    t = target_sentiment.capitalize()

    # Columns in the logical order: prior → marginal → likelihood → posterior
    prob_cols = [
        f"P({t})",           # prior
        "P(keyword)",        # marginal
        f"P(keyword|{t})",   # likelihood
        f"P({t}|keyword)",   # posterior
    ]

    heat_data = results_df.set_index("Keyword")[prob_cols]

    fig, ax = plt.subplots(figsize=(11, max(4, len(results_df) * 0.9)))
    sns.heatmap(
        heat_data,
        annot=True, fmt=".4f",
        cmap="YlOrRd", linewidths=0.5,
        ax=ax, cbar_kws={"label": "Probability"}
    )
    ax.set_title(
        f"Bayesian Probability Heatmap — target: {target_sentiment}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=10)
    plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Gradient Descent  (Ryan)
# ══════════════════════════════════════════════════════════════════════════════

def plot_gd_convergence(manual_result, scipy_result=None, max_display=50):
    """
    Two-panel convergence chart for gradient descent.

    Panel 1 — m (slope) and b (intercept) values over iterations.
              Vertical dashed lines mark the 4 Part 3 manual iterations.
    Panel 2 — MSE over iterations on a log scale.
              SciPy path is overlaid if provided.

    Parameters
    ----------
    manual_result : dict  Output of run_manual_gradient_descent() from gradient_descent.py.
                          Must have 'm_history', 'b_history', 'mse_history'.
    scipy_result  : dict  Output of run_scipy_gradient_descent() — optional overlay.
    max_display   : int   Cap the x-axis at this many iterations for readability.
    """
    # Slice histories to max_display for cleaner charts
    m_hist   = manual_result["m_history"][:max_display + 1]
    b_hist   = manual_result["b_history"][:max_display + 1]
    mse_hist = manual_result["mse_history"][:max_display + 1]
    iters    = range(len(m_hist))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Part 4 — Gradient Descent Convergence", fontsize=14, fontweight="bold")

    # ── Panel 1: m and b over iterations ──────────────────────────────────────
    ax1.plot(iters, m_hist, color="#2E86AB", linewidth=2.5,
             marker="o", markersize=3, label="m (slope)")
    ax1.plot(iters, b_hist, color="#E84855", linewidth=2.5,
             marker="s", markersize=3, label="b (intercept)")

    # Dashed vertical lines marking the 4 iterations done by hand in Part 3
    for step in range(1, 5):
        if step <= max_display:
            ax1.axvline(step, color="grey", linestyle="--", alpha=0.5, linewidth=0.9)
            ax1.text(step + 0.3, min(m_hist + b_hist),
                     f"Iter {step}", fontsize=7, color="grey")

    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Parameter Value", fontsize=11)
    ax1.set_title("m and b Converging Over Iterations", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: MSE over iterations ──────────────────────────────────────────
    ax2.plot(iters, mse_hist, color="#A23B72", linewidth=2.5,
             marker="o", markersize=3, label="MSE — manual GD")

    # Overlay SciPy MSE path if provided
    if scipy_result is not None:
        s_mse  = scipy_result["mse_history"][:max_display + 1]
        ax2.plot(range(len(s_mse)), s_mse, color="#F18F01", linewidth=2,
                 linestyle="--", marker="^", markersize=4, label="MSE — SciPy L-BFGS-B")

    # Mark the 4 manual iterations
    for step in range(1, 5):
        if step <= max_display:
            ax2.axvline(step, color="grey", linestyle="--", alpha=0.5, linewidth=0.9)

    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax2.set_title("MSE Decreasing Over Iterations", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    # Log scale makes both the early rapid drop and late fine-tuning visible
    ax2.set_yscale("log")

    plt.tight_layout()


def plot_gd_final_line(X_points, Y_points, m_init, b_init, m_final, b_final):
    """
    Plot the initial line and the final converged line of best fit.

    Parameters
    ----------
    X_points, Y_points : np.ndarray  The two data points (1,3) and (3,6).
    m_init, b_init     : float       Starting slope and intercept (m₀=-1, b₀=1).
    m_final, b_final   : float       Converged slope and intercept.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the two data points
    ax.scatter(X_points, Y_points, color='black', s=100, zorder=5,
               label='Data points  (1,3) and (3,6)')
    ax.annotate('(1, 3)', (1, 3), textcoords='offset points',
                xytext=(8, -14), fontsize=10)
    ax.annotate('(3, 6)', (3, 6), textcoords='offset points',
                xytext=(8, -14), fontsize=10)

    x_line = np.linspace(0, 4, 100)

    # Initial line — dashed red, shows where we started
    ax.plot(x_line, m_init * x_line + b_init,
            linestyle='--', color='#E84855', alpha=0.65,
            label=f'Initial line   m={m_init}, b={b_init}')

    # Final converged line — solid blue
    ax.plot(x_line, m_final * x_line + b_final,
            color='#2E86AB', linewidth=2.5,
            label=f'Final line   m={m_final:.4f}, b={b_final:.4f}')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Gradient Descent — Initial vs Final Line of Best Fit',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_gd_compare_methods(manual_result, scipy_result):
    """
    Side-by-side plot showing the optimisation path of manual GD vs SciPy.

    Left  — parameter-space path: trajectory of (m, b) over iterations.
    Right — convergence curves of both methods on the same axes.

    Parameters
    ----------
    manual_result : dict  Output of run_manual_gradient_descent().
    scipy_result  : dict  Output of run_scipy_gradient_descent().
    """
    # Build (m, b) path arrays from the history lists
    manual_m = manual_result["m_history"]
    manual_b = manual_result["b_history"]
    scipy_m  = scipy_result["m_history"]
    scipy_b  = scipy_result["b_history"]

    manual_mse = manual_result["mse_history"]
    scipy_mse  = scipy_result["mse_history"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Manual GD vs SciPy L-BFGS-B — Optimisation Paths",
                 fontsize=13, fontweight="bold")

    # ── Left: trajectory in (m, b) parameter space ────────────────────────────
    ax1.plot(manual_m, manual_b, "o-", color="#E74C3C", linewidth=1.5,
             markersize=3, label=f"Manual GD  ({len(manual_m)-1} iters)", alpha=0.8)
    ax1.plot(scipy_m, scipy_b, "s-", color="#3498DB", linewidth=1.5,
             markersize=4, label=f"SciPy BFGS ({len(scipy_m)-1} iters)", alpha=0.8)

    # Mark start (black triangle) and final points (stars)
    ax1.plot(manual_m[0], manual_b[0], "k^", markersize=10, label="Start (m=-1, b=1)")
    ax1.plot(manual_m[-1], manual_b[-1], "r*", markersize=12,
             label=f"GD end  ({manual_m[-1]:.3f}, {manual_b[-1]:.3f})")
    ax1.plot(scipy_m[-1], scipy_b[-1], "b*", markersize=12,
             label=f"BFGS end ({scipy_m[-1]:.3f}, {scipy_b[-1]:.3f})")

    ax1.set_xlabel("m (slope)", fontsize=11)
    ax1.set_ylabel("b (intercept)", fontsize=11)
    ax1.set_title("Trajectory in Parameter Space", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Right: MSE convergence curves side by side ────────────────────────────
    ax2.plot(range(len(manual_mse)), manual_mse, color="#E74C3C", linewidth=1.8,
             label=f"Manual GD  (final={manual_mse[-1]:.2e})")
    ax2.plot(range(len(scipy_mse)), scipy_mse, color="#3498DB", linewidth=1.8,
             label=f"SciPy BFGS (final={scipy_mse[-1]:.2e})")

    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("MSE", fontsize=11)
    ax2.set_title("MSE Convergence — Both Methods", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    if min(manual_mse + scipy_mse) > 0:
        ax2.set_yscale("log")

    plt.tight_layout()