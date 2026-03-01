"""
Part 4: Gradient Descent in Code
Author: Ryan

Converts the manual calculations from Part 3 into Python code using SciPy.
Uses IDENTICAL initial conditions and data as the manual work:
    Data points : (1, 3) and (3, 6)
    Model       : ŷ = mx + b
    Initial m   : m₀ = -1
    Initial b   : b₀ = 1
    Learning rate α = 0.1

Functions
---------
mse_cost(m, b, X, Y)          — MSE cost function  J(m, b)
gradient_m(m, b, X, Y)        — ∂J/∂m  (derived by chain rule in Part 3)
gradient_b(m, b, X, Y)        — ∂J/∂b  (derived by chain rule in Part 3)
run_manual_gradient_descent()  — step-by-step GD loop, returns full history
run_scipy_gradient_descent()   — SciPy wrapper using the same cost/gradient
plot_convergence()             — two-panel convergence chart (m & b | MSE)
compare_results()              — side-by-side comparison table
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


# ── Problem constants ─────────────────────────────────────────────────────────

X_POINTS    = np.array([1.0, 3.0])
Y_POINTS    = np.array([3.0, 6.0])
M_INIT      = -1.0
B_INIT      =  1.0
ALPHA       =  0.1
N           =  2


# ── Cost function ─────────────────────────────────────────────────────────────

def mse_cost(m, b, X=X_POINTS, Y=Y_POINTS):
    """
    Mean Squared Error cost function.

        J(m, b) = (1/n) Σ (yᵢ - ŷᵢ)²
    """
    n      = len(X)
    y_pred = m * X + b
    errors = Y - y_pred
    return (1.0 / n) * np.sum(errors ** 2)


# ── Partial derivatives ───────────────────────────────────────────────────────

def gradient_m(m, b, X=X_POINTS, Y=Y_POINTS):
    """
    ∂J/∂m = (2/n) Σ (m·xᵢ + b - yᵢ) · xᵢ
    For assignment data simplifies to: 10m + 4b - 21
    """
    n         = len(X)
    residuals = (m * X + b) - Y
    return (2.0 / n) * np.sum(residuals * X)


def gradient_b(m, b, X=X_POINTS, Y=Y_POINTS):
    """
    ∂J/∂b = (2/n) Σ (m·xᵢ + b - yᵢ)
    For assignment data simplifies to: 4m + 2b - 9
    """
    n         = len(X)
    residuals = (m * X + b) - Y
    return (2.0 / n) * np.sum(residuals)


# ── Manual Gradient Descent ───────────────────────────────────────────────────

def run_manual_gradient_descent(m_init   = M_INIT,
                                 b_init   = B_INIT,
                                 alpha    = ALPHA,
                                 n_iters  = 100,
                                 X        = X_POINTS,
                                 Y        = Y_POINTS,
                                 verbose  = True):
    """
    Gradient descent for linear regression — every step explicitly visible.

    Returns
    -------
    dict with keys:
        m_history, b_history, mse_history, dm_history, db_history,
        final_m, final_b, final_mse, iterations
    """
    m = float(m_init)
    b = float(b_init)

    m_history   = [m]
    b_history   = [b]
    mse_history = [mse_cost(m, b, X, Y)]
    dm_history  = []
    db_history  = []

    if verbose:
        print(f"{'Iter':>5}  {'m':>10}  {'b':>10}  {'∂J/∂m':>10}  {'∂J/∂b':>10}  {'MSE':>12}")
        print("-" * 65)
        print(f"{'0 (init)':>8}  {m:>10.6f}  {b:>10.6f}  "
              f"{'—':>10}  {'—':>10}  {mse_history[0]:>12.6f}")

    for i in range(n_iters):

        # ── STEP 1: Predict ŷᵢ = m·xᵢ + b ───────────────────────────────────
        y_pred = m * X + b

        # ── STEP 2: Compute errors eᵢ = yᵢ - ŷᵢ ─────────────────────────────
        errors = Y - y_pred

        # ── STEP 3: MSE = (1/n) Σ eᵢ² ────────────────────────────────────────
        cost = (1.0 / len(X)) * np.sum(errors ** 2)

        # ── STEP 4a: Gradient ∂J/∂m ───────────────────────────────────────────
        dm = (2.0 / len(X)) * np.sum(-errors * X)

        # ── STEP 4b: Gradient ∂J/∂b ───────────────────────────────────────────
        db = (2.0 / len(X)) * np.sum(-errors)

        # ── STEP 5: Update m and b ─────────────────────────────────────────────
        m_new = m - alpha * dm
        b_new = b - alpha * db

        dm_history.append(dm)
        db_history.append(db)
        m_history.append(m_new)
        b_history.append(b_new)
        mse_history.append(mse_cost(m_new, b_new, X, Y))

        if verbose and i < 10:
            print(f"{i+1:>8}  {m_new:>10.6f}  {b_new:>10.6f}  "
                  f"{dm:>10.6f}  {db:>10.6f}  {mse_history[-1]:>12.6f}")

        m = m_new
        b = b_new

    if verbose and n_iters > 10:
        print(f"     ...  ({n_iters - 10} more iterations)")
        print(f"\nFinal  m = {m:.8f}")
        print(f"Final  b = {b:.8f}")
        print(f"Final MSE = {mse_history[-1]:.10f}")

    return {
        "m_history":   m_history,
        "b_history":   b_history,
        "mse_history": mse_history,
        "dm_history":  dm_history,
        "db_history":  db_history,
        "final_m":     m,
        "final_b":     b,
        "final_mse":   mse_history[-1],
        "iterations":  n_iters,
    }


# ── SciPy Wrapper ─────────────────────────────────────────────────────────────

def run_scipy_gradient_descent(m_init = M_INIT,
                                b_init = B_INIT,
                                X      = X_POINTS,
                                Y      = Y_POINTS):
    """
    Use scipy.optimize.minimize (L-BFGS-B) to minimise the same MSE cost function.

    Returns
    -------
    dict with keys:
        final_m, final_b, final_mse, iterations, converged, scipy_result,
        m_history, b_history, mse_history
    """
    m_history   = [float(m_init)]
    b_history   = [float(b_init)]
    mse_history = [mse_cost(m_init, b_init, X, Y)]

    def callback(params):
        m_history.append(float(params[0]))
        b_history.append(float(params[1]))
        mse_history.append(mse_cost(params[0], params[1], X, Y))

    def cost_flat(params):
        return mse_cost(params[0], params[1], X, Y)

    def grad_flat(params):
        return np.array([
            gradient_m(params[0], params[1], X, Y),
            gradient_b(params[0], params[1], X, Y),
        ])

    result = optimize.minimize(
        cost_flat,
        x0       = np.array([float(m_init), float(b_init)]),
        jac      = grad_flat,
        method   = "L-BFGS-B",
        callback = callback,
        options  = {"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10},
    )

    return {
        "final_m":     float(result.x[0]),
        "final_b":     float(result.x[1]),
        "final_mse":   float(result.fun),
        "iterations":  result.nit,
        "converged":   result.success,
        "scipy_result": result,
        "m_history":   m_history,
        "b_history":   b_history,
        "mse_history": mse_history,
    }


# ── Convergence Plots ─────────────────────────────────────────────────────────

def plot_convergence(manual_result, scipy_result=None, max_display=50):
    """
    Two-panel convergence chart.

    Panel 1 — m and b values over iterations
    Panel 2 — MSE over iterations

    Parameters
    ----------
    manual_result : dict  Output of run_manual_gradient_descent().
    scipy_result  : dict  Output of run_scipy_gradient_descent() — optional.
    max_display   : int   Cap x-axis at this iteration for readability.
    """
    m_hist   = manual_result["m_history"][:max_display + 1]
    b_hist   = manual_result["b_history"][:max_display + 1]
    mse_hist = manual_result["mse_history"][:max_display + 1]
    iters    = range(len(m_hist))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Part 4 — Gradient Descent Convergence", fontsize=14, fontweight="bold")

    ax1.plot(iters, m_hist, color="#2E86AB", linewidth=2.5,
             marker="o", markersize=3, label="m (slope)  — manual GD")
    ax1.plot(iters, b_hist, color="#E84855", linewidth=2.5,
             marker="s", markersize=3, label="b (intercept) — manual GD")

    for step, label in [(1, "Iter 1"), (2, "Iter 2"), (3, "Iter 3"), (4, "Iter 4")]:
        if step <= max_display:
            ax1.axvline(step, color="grey", linestyle="--", alpha=0.5, linewidth=0.9)

    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Parameter Value", fontsize=11)
    ax1.set_title("m and b Converging Over Iterations", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, mse_hist, color="#A23B72", linewidth=2.5,
             marker="o", markersize=3, label="MSE — manual GD")

    if scipy_result is not None:
        s_mse  = scipy_result["mse_history"][:max_display + 1]
        s_iter = range(len(s_mse))
        ax2.plot(s_iter, s_mse, color="#F18F01", linewidth=2,
                 linestyle="--", marker="^", markersize=4, label="MSE — SciPy L-BFGS-B")

    for step in range(1, 5):
        if step <= max_display:
            ax2.axvline(step, color="grey", linestyle="--", alpha=0.5, linewidth=0.9)

    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax2.set_title("MSE Decreasing Over Iterations", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()


# ── Comparison Table ──────────────────────────────────────────────────────────

def compare_results(manual_result, scipy_result):
    """
    Print a side-by-side comparison of manual GD vs SciPy results.
    """
    print("\n" + "=" * 62)
    print("  COMPARISON: Manual Gradient Descent  vs  SciPy L-BFGS-B")
    print("=" * 62)
    print(f"  {'':22s}  {'Manual GD':>15}  {'SciPy BFGS':>15}")
    print("  " + "-" * 58)
    print(f"  {'Final m':22s}  {manual_result['final_m']:>15.8f}  "
          f"{scipy_result['final_m']:>15.8f}")
    print(f"  {'Final b':22s}  {manual_result['final_b']:>15.8f}  "
          f"{scipy_result['final_b']:>15.8f}")
    print(f"  {'Final MSE':22s}  {manual_result['final_mse']:>15.10f}  "
          f"{scipy_result['final_mse']:>15.10f}")
    print(f"  {'Iterations':22s}  {manual_result['iterations']:>15d}  "
          f"{scipy_result['iterations']:>15d}")
    print(f"  {'Converged':22s}  {'Yes (tolerance)':>15}  "
          f"{'Yes' if scipy_result['converged'] else 'No':>15}")

    dm = abs(manual_result["final_m"] - scipy_result["final_m"])
    db = abs(manual_result["final_b"] - scipy_result["final_b"])
    print("\n  Difference in final m:", f"{dm:.2e}")
    print("  Difference in final b:", f"{db:.2e}")

    if dm < 1e-4 and db < 1e-4:
        print("\n  ✓ Both methods converged to the same solution.")
    else:
        print("\n  ⚠ Small difference — manual GD may need more iterations.")
    print("=" * 62)