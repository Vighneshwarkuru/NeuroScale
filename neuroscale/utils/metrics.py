"""Evaluation metrics for NeuroScale++."""

import torch
import numpy as np
from typing import Optional


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list[float]:
    """Compute top-k accuracy.

    Args:
        output: Model predictions (batch_size, num_classes).
        target: Ground truth labels (batch_size,).
        topk: Tuple of k values to compute accuracy for.

    Returns:
        List of accuracy values for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def energy_savings(timesteps_used: np.ndarray, max_timestep: int) -> float:
    """Compute energy savings relative to using max timesteps for all samples.

    Args:
        timesteps_used: Array of actual timesteps used per sample.
        max_timestep: Maximum possible timesteps.

    Returns:
        Fraction of energy saved (0.0 to 1.0).
    """
    total_possible = len(timesteps_used) * max_timestep
    total_used = timesteps_used.sum()
    return 1.0 - (total_used / total_possible)


def compute_flops(timesteps_used: np.ndarray, flops_per_timestep: float) -> float:
    """Compute total FLOPs used.

    Args:
        timesteps_used: Array of actual timesteps used per sample.
        flops_per_timestep: FLOPs for a single SNN timestep.

    Returns:
        Total FLOPs used.
    """
    return timesteps_used.sum() * flops_per_timestep


def average_timestep(timesteps_used: np.ndarray) -> float:
    """Compute average timesteps used across samples."""
    return timesteps_used.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Synaptic Operations (SOPs)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sops(
    timesteps_used: np.ndarray,
    sops_per_timestep: float,
    max_timestep: int,
) -> dict:
    """Compute Synaptic Operations (SOPs) — the standard SNN energy metric.

    SOPs count the number of synaptic addition events, which is proportional
    to the firing rate and number of synapses. Each SNN timestep contributes
    a fixed number of SOPs determined by network connectivity and firing rate.

    Args:
        timesteps_used: Array of actual timesteps used per sample (N,).
        sops_per_timestep: Expected SOPs per timestep per sample.
            Approximated as: num_synapses * mean_firing_rate
            For ResNet-20 on CIFAR-10 this is typically ~1e6–1e7.
        max_timestep: Maximum timestep (baseline uses this for all samples).

    Returns:
        Dict with adaptive_sops, baseline_sops, sops_reduction (fraction),
        sops_per_sample (mean), and sops_ratio.
    """
    adaptive_total = float(timesteps_used.sum()) * sops_per_timestep
    baseline_total = float(len(timesteps_used) * max_timestep) * sops_per_timestep
    adaptive_per_sample = float(timesteps_used.mean()) * sops_per_timestep
    baseline_per_sample = float(max_timestep) * sops_per_timestep

    return {
        "adaptive_sops_total": adaptive_total,
        "baseline_sops_total": baseline_total,
        "adaptive_sops_per_sample": adaptive_per_sample,
        "baseline_sops_per_sample": baseline_per_sample,
        "sops_reduction": 1.0 - (adaptive_total / max(baseline_total, 1)),
        "sops_ratio": adaptive_total / max(baseline_total, 1),
    }


def estimate_sops_per_timestep(model: torch.nn.Module, input_shape: tuple) -> float:
    """Estimate SOPs per timestep from model architecture.

    Uses the number of multiply-accumulate (MAC) operations as a proxy.
    For SNNs, MACs become additions (no multiplications) due to binary spikes,
    so this is an upper bound; scale by mean_firing_rate for tighter estimate.

    Args:
        model: The SNN or ANN model.
        input_shape: Shape of one input sample, e.g. (3, 32, 32).

    Returns:
        Estimated SOPs per timestep (float).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        dummy = torch.zeros(1, *input_shape)
        with FlopCounterMode(model, display=False) as fcm:
            model(dummy)
        total_flops = sum(fcm.get_flop_counts().values())
        # For SNNs, FLOPs ≈ 2 × MACs; SOPs ≈ MACs (additions only)
        return float(total_flops) / 2.0
    except Exception:
        # Fallback: rough estimate based on param count
        params = sum(p.numel() for p in model.parameters())
        return float(params)


# ─────────────────────────────────────────────────────────────────────────────
# Scaling Law Fit Quality
# ─────────────────────────────────────────────────────────────────────────────

def compute_r2_scores(
    timesteps: np.ndarray,
    performances: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> np.ndarray:
    """Compute R² (coefficient of determination) for each sample's curve fit.

    R² = 1 - SS_res / SS_tot
    where SS_res = sum((y - y_hat)^2), SS_tot = sum((y - y_mean)^2)

    Args:
        timesteps: (T,) array of timestep values.
        performances: (N, T) array of actual accuracy values.
        alphas, betas, gammas: (N,) fitted curve parameters.

    Returns:
        (N,) array of R² values per sample. Values near 1.0 = good fit.
    """
    N = len(alphas)
    r2_scores = np.zeros(N)

    for i in range(N):
        y_true = performances[i]  # (T,)
        y_pred = alphas[i] * (timesteps ** betas[i]) + gammas[i]
        y_mean = y_true.mean()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        if ss_tot < 1e-10:
            # Constant signal — perfect fit if residuals also ~0
            r2_scores[i] = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r2_scores[i] = 1.0 - ss_res / ss_tot

    return r2_scores


def compute_fit_mse(
    timesteps: np.ndarray,
    performances: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> np.ndarray:
    """Compute per-sample MSE of power-law fit vs actual performance.

    Args:
        timesteps: (T,) array of timestep values.
        performances: (N, T) actual accuracy values.
        alphas, betas, gammas: (N,) fitted parameters.

    Returns:
        (N,) array of MSE values per sample.
    """
    N = len(alphas)
    mse = np.zeros(N)
    for i in range(N):
        y_true = performances[i]
        y_pred = alphas[i] * (timesteps ** betas[i]) + gammas[i]
        mse[i] = np.mean((y_true - y_pred) ** 2)
    return mse


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error (ECE)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well a model's confidence aligns with its accuracy.
    A perfectly calibrated model has ECE = 0: when it says 80% confident,
    it's right 80% of the time.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: (N,) predicted max-softmax probabilities [0, 1].
        correct: (N,) binary array, 1 if prediction was correct.
        n_bins: Number of confidence bins (default 15).

    Returns:
        Dict with 'ece', 'mce' (max calibration error), 'bin_accs',
        'bin_confs', 'bin_counts' for reliability diagram plotting.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (confidences > bins[b]) & (confidences <= bins[b + 1])
        if mask.sum() > 0:
            bin_accs[b] = correct[mask].mean()
            bin_confs[b] = confidences[mask].mean()
            bin_counts[b] = mask.sum()

    N = len(confidences)
    gaps = np.abs(bin_accs - bin_confs)
    ece = float(np.sum(gaps * bin_counts) / max(N, 1))
    mce = float(gaps[bin_counts > 0].max()) if (bin_counts > 0).any() else 0.0

    return {
        "ece": ece,
        "mce": mce,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bins,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predictor Timestep Error
# ─────────────────────────────────────────────────────────────────────────────

def compute_predictor_t_error(
    predicted_t: np.ndarray,
    optimal_t: np.ndarray,
) -> dict:
    """Measure how accurately the predictor assigns the right timestep.

    Args:
        predicted_t: (N,) timesteps assigned by the predictor at eval time.
        optimal_t: (N,) ground-truth T_optimal from Phase 2 profiling.

    Returns:
        Dict with mae, rmse, exact_match_rate, over_allocation_rate,
        under_allocation_rate, mean_error (signed).
    """
    diff = predicted_t.astype(float) - optimal_t.astype(float)
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    exact = float((diff == 0).mean())
    over = float((diff > 0).mean())   # predicted too high (wasteful)
    under = float((diff < 0).mean())  # predicted too low (may hurt accuracy)
    mean_err = float(diff.mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "exact_match_rate": exact,
        "over_allocation_rate": over,
        "under_allocation_rate": under,
        "mean_error": mean_err,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-5 accuracy helper (numpy-based, for eval arrays)
# ─────────────────────────────────────────────────────────────────────────────

def topk_accuracy_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
) -> float:
    """Compute top-k accuracy from numpy logit/score arrays.

    Args:
        logits: (N, C) score array.
        targets: (N,) ground-truth class indices.
        k: k value for top-k accuracy.

    Returns:
        Top-k accuracy as a percentage.
    """
    topk_preds = np.argsort(logits, axis=1)[:, -k:]  # (N, k)
    correct = np.any(topk_preds == targets[:, None], axis=1)
    return float(correct.mean() * 100)
