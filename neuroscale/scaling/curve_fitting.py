"""Power-law curve fitting for per-sample scaling laws.

Given profiling data (accuracy at different timesteps for each sample),
fits the power-law model: Performance(T) = α × T^β + γ

This produces ground-truth labels for training the Complexity Predictor.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Optional


def power_law(t: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Power-law scaling function.

    Performance(T) = α × T^β + γ

    Args:
        t: Timestep values.
        alpha: Amplitude parameter (> 0).
        beta: Exponent parameter (0 < β ≤ 1 for diminishing returns).
        gamma: Baseline/asymptote parameter.

    Returns:
        Predicted performance at each timestep.
    """
    return alpha * np.power(t, beta) + gamma


def fit_power_law(
    timesteps: np.ndarray,
    performances: np.ndarray,
    max_retries: int = 3,
) -> tuple[float, float, float]:
    """Fit power-law parameters to observed (timestep, performance) data.

    Args:
        timesteps: Array of timestep values (e.g., [1, 2, 4, 8, 16, 32, 64]).
        performances: Corresponding performance values (e.g., accuracy at each T).
        max_retries: Number of fitting attempts with different initializations.

    Returns:
        Tuple of (alpha, beta, gamma) — fitted parameters.
        If fitting fails, returns default values (0.5, 0.5, 0.5).
    """
    timesteps = np.asarray(timesteps, dtype=np.float64)
    performances = np.asarray(performances, dtype=np.float64)

    # Filter out any NaN or invalid values
    valid = np.isfinite(performances) & np.isfinite(timesteps) & (timesteps > 0)
    if valid.sum() < 3:
        return (0.5, 0.5, performances.mean() if len(performances) > 0 else 0.5)

    timesteps = timesteps[valid]
    performances = performances[valid]

    # Different initial guesses to improve convergence
    init_guesses = [
        (0.1, 0.5, performances.min()),
        (0.5, 0.3, 0.0),
        (1.0, 0.7, performances[0]),
    ]

    best_params = None
    best_residual = float("inf")

    for p0 in init_guesses[:max_retries]:
        try:
            params, _ = curve_fit(
                power_law,
                timesteps,
                performances,
                p0=p0,
                bounds=(
                    [0.0, 0.01, -1.0],  # Lower bounds
                    [10.0, 2.0, 1.0],   # Upper bounds
                ),
                maxfev=5000,
            )

            # Compute residual
            predicted = power_law(timesteps, *params)
            residual = np.mean((performances - predicted) ** 2)

            if residual < best_residual:
                best_residual = residual
                best_params = params

        except (RuntimeError, ValueError):
            continue

    if best_params is not None:
        alpha, beta, gamma = best_params
        # Ensure physically meaningful values
        alpha = max(alpha, 1e-6)
        beta = np.clip(beta, 0.01, 1.5)
        gamma = np.clip(gamma, -0.5, 1.0)
        return (float(alpha), float(beta), float(gamma))

    # Fallback: simple linear interpolation-based estimate
    return _fallback_estimate(timesteps, performances)


def _fallback_estimate(
    timesteps: np.ndarray, performances: np.ndarray
) -> tuple[float, float, float]:
    """Fallback parameter estimation when curve_fit fails."""
    gamma = performances[0] if len(performances) > 0 else 0.0
    perf_range = performances[-1] - performances[0] if len(performances) > 1 else 0.5
    alpha = perf_range / (timesteps[-1] ** 0.5) if len(timesteps) > 0 else 0.5
    beta = 0.5  # Default to square-root scaling
    return (max(float(alpha), 1e-6), float(beta), float(gamma))


def compute_optimal_timestep(
    alpha: float,
    beta: float,
    gamma: float,
    target_performance: float,
    min_timestep: int = 1,
    max_timestep: int = 64,
) -> int:
    """Compute the minimum timestep to reach target performance.

    Solves: target = α × T^β + γ  →  T = ((target - γ) / α) ^ (1/β)

    Args:
        alpha: Amplitude parameter.
        beta: Exponent parameter.
        gamma: Baseline parameter.
        target_performance: Desired performance level.
        min_timestep: Minimum allowed timestep.
        max_timestep: Maximum allowed timestep.

    Returns:
        Optimal timestep (integer, clamped to [min_timestep, max_timestep]).
    """
    # If target is already met at baseline, use minimum timestep
    if gamma >= target_performance:
        return min_timestep

    # If alpha is essentially zero, the sample can't improve
    if alpha < 1e-6:
        return max_timestep

    # Solve for T
    numerator = target_performance - gamma
    if numerator <= 0:
        return min_timestep

    ratio = numerator / alpha

    if beta <= 0.01:
        return max_timestep

    t_optimal = ratio ** (1.0 / beta)

    # Clamp and round up
    t_optimal = int(np.ceil(t_optimal))
    t_optimal = max(min_timestep, min(t_optimal, max_timestep))

    return t_optimal


def batch_fit_power_law(
    timesteps: np.ndarray,
    all_performances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit scaling law parameters for a batch of samples.

    Args:
        timesteps: Array of timestep values (num_timesteps,).
        all_performances: Performance matrix (num_samples, num_timesteps).

    Returns:
        Tuple of (alphas, betas, gammas) — each shape (num_samples,).
    """
    num_samples = all_performances.shape[0]
    alphas = np.zeros(num_samples)
    betas = np.zeros(num_samples)
    gammas = np.zeros(num_samples)

    for i in range(num_samples):
        alpha, beta, gamma = fit_power_law(timesteps, all_performances[i])
        alphas[i] = alpha
        betas[i] = beta
        gammas[i] = gamma

    return alphas, betas, gammas
