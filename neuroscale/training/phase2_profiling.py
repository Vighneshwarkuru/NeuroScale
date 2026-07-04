"""Phase 2: SNN Profiling and Curve Fitting.

After converting the trained ANN to an SNN (Phase 1 -> conversion), this phase:
1. Runs every training sample through the SNN at multiple timesteps
2. Records the accuracy/confidence at each timestep checkpoint
3. Fits power-law scaling curves per sample
4. Saves the fitted parameters (α, β, γ) as ground-truth labels

These labels are used to train the Complexity Predictor in Phase 3.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..spiking.snn_model import SNNModel
from ..conversion.converter import ANNtoSNNConverter
from ..scaling.curve_fitting import fit_power_law, batch_fit_power_law, compute_optimal_timestep
from ..utils.logging import setup_logger
from ..utils.metrics import compute_r2_scores, compute_fit_mse
from ..utils.results_manager import ResultsManager


def profile_snn(
    ann_model: nn.Module,
    train_loader: DataLoader,
    config: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    results_manager: Optional[ResultsManager] = None,
) -> dict[str, np.ndarray]:
    """Profile the SNN and fit per-sample scaling laws (Phase 2).

    Steps:
    1. Convert ANN to SNN using threshold balancing + calibration
    2. For each training sample, run SNN at all checkpoint timesteps
    3. Record whether prediction is correct at each timestep
    4. Fit power-law curve per sample to get (α, β, γ)
    5. Save profiling results

    Args:
        ann_model: Trained ANN model from Phase 1.
        train_loader: Training data loader (no augmentation preferred).
        config: Configuration dictionary.
        device: Computation device.
        save_dir: Directory to save profiling results.
        log_dir: Directory for logs.
        results_manager: Optional ResultsManager for CSV/metric logging.

    Returns:
        Dictionary with:
            'alphas': (num_samples,) fitted α values
            'betas': (num_samples,) fitted β values
            'gammas': (num_samples,) fitted γ values
            'performances': (num_samples, num_timesteps) accuracy at each T
            'timesteps': (num_timesteps,) timestep values used
    """
    logger = setup_logger("phase2_profiling", log_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    snn_cfg = config["snn"]
    conversion_cfg = config["conversion"]
    timesteps = snn_cfg["timesteps"]  # e.g., [1, 2, 4, 8, 16, 32, 64]

    logger.info(f"Phase 2: Profiling SNN at timesteps {timesteps}")

    start_time = time.time()

    # Step 1: Convert ANN to SNN
    logger.info("Converting ANN to SNN...")
    conversion_start = time.time()
    converter = ANNtoSNNConverter(
        percentile=conversion_cfg.get("percentile", 99.9),
        calibrate=True,
    )

    # Create a small calibration loader from training data
    calib_loader = _make_calibration_loader(train_loader, config)

    snn_base, thresholds = converter.convert(ann_model, calib_loader, device)
    threshold_list = list(thresholds.values())
    conversion_time = time.time() - conversion_start

    # Build SNN model
    snn_model = SNNModel(
        ann_model=snn_base,
        thresholds=threshold_list,
        max_timestep=max(timesteps),
        neuron_type="if",
    ).to(device)
    snn_model.eval()

    # Save converted SNN
    torch.save({
        "snn_state_dict": snn_model.state_dict(),
        "thresholds": threshold_list,
    }, save_dir / "snn_converted.pth")
    logger.info(f"SNN model saved. {len(threshold_list)} layer thresholds computed.")

    # Step 2: Profile each sample at all timesteps
    logger.info("Profiling samples at multiple timesteps...")
    all_performances = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="Profiling")):
            images, targets = images.to(device), targets.to(device)

            # Get SNN outputs at all checkpoint timesteps
            outputs_at_t = snn_model.forward_at_timesteps(images, timesteps)

            # Compute correctness at each timestep
            batch_perf = []
            for t in timesteps:
                logits = outputs_at_t[t]
                preds = logits.argmax(dim=1)
                correct = (preds == targets).float()  # 1 if correct, 0 if wrong
                batch_perf.append(correct.cpu())

            # Stack: (batch_size, num_timesteps)
            batch_perf = torch.stack(batch_perf, dim=1).numpy()
            all_performances.append(batch_perf)
            all_labels.append(targets.cpu().numpy())
            all_images.append(images.cpu())

            # Progress logging every 100 batches
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Profiled {(batch_idx + 1) * images.size(0)} samples...")

    # Concatenate all results
    performances = np.concatenate(all_performances, axis=0)  # (N, num_timesteps)
    labels = np.concatenate(all_labels, axis=0)  # (N,)
    # Note: we don't store all_images to save memory; re-load during Phase 3

    logger.info(f"Profiling complete. Total samples: {performances.shape[0]}")
    logger.info(f"Avg accuracy at each T: {performances.mean(axis=0)}")

    # Step 3: Smooth performances using cumulative max (monotonicity)
    # If a sample is correct at T=8, it should also be "reachable" at T>8
    # Use cumulative correctness rate instead of binary per-run
    performances_smooth = _compute_cumulative_accuracy(performances)

    # Step 4: Fit power-law curves
    logger.info("Fitting power-law curves per sample...")
    timesteps_arr = np.array(timesteps, dtype=np.float64)
    alphas, betas, gammas = batch_fit_power_law(timesteps_arr, performances_smooth)

    logger.info(f"Curve fitting complete.")
    logger.info(f"  α: mean={alphas.mean():.4f}, std={alphas.std():.4f}")
    logger.info(f"  β: mean={betas.mean():.4f}, std={betas.std():.4f}")
    logger.info(f"  γ: mean={gammas.mean():.4f}, std={gammas.std():.4f}")

    # Step 5: Compute curve-fit quality per sample (R² and MSE)
    logger.info("Computing scaling law fit quality (R², MSE)...")
    r2_scores = compute_r2_scores(timesteps_arr, performances_smooth, alphas, betas, gammas)
    fit_mse_arr = compute_fit_mse(timesteps_arr, performances_smooth, alphas, betas, gammas)
    logger.info(f"  R²: mean={r2_scores.mean():.4f}, median={np.median(r2_scores):.4f}, "
                f"pct≥0.9={100*(r2_scores>=0.9).mean():.1f}%")

    # Step 6: Measure ANN→SNN accuracy gap at T_max on training data
    # Use the last column of performances (highest timestep = best SNN accuracy)
    snn_max_t_acc = float(performances_smooth[:, -1].mean() * 100)
    logger.info(f"  SNN accuracy at T_max={max(timesteps)}: {snn_max_t_acc:.2f}%")

    # Step 7: Save results
    results = {
        "alphas": alphas,
        "betas": betas,
        "gammas": gammas,
        "performances": performances_smooth,
        "timesteps": timesteps_arr,
        "labels": labels,
        "r2_scores": r2_scores,
        "fit_mse": fit_mse_arr,
        "snn_max_t_acc": np.array(snn_max_t_acc),
    }

    np.savez(
        save_dir / "profiling_results.npz",
        **results
    )
    logger.info(f"Profiling results saved to {save_dir / 'profiling_results.npz'}")

    # Log distribution of sample difficulties
    _log_difficulty_distribution(alphas, betas, gammas, timesteps, config, logger)

    # Log to ResultsManager
    if results_manager is not None:
        profiling_time = time.time() - start_time
        mean_acc = performances_smooth.mean(axis=0)
        std_acc = performances_smooth.std(axis=0)

        results_manager.save_phase2_profiling_stats(
            timesteps=timesteps_arr,
            mean_acc_per_t=mean_acc,
            std_acc_per_t=std_acc,
        )

        # Compute T_optimal for each sample
        target_acc = config["scaling_law"]["target_accuracy"]
        max_t = config["snn"]["max_timestep"]
        t_optimals = np.array([
            compute_optimal_timestep(a, b, g, target_acc, max_timestep=max_t)
            for a, b, g in zip(alphas, betas, gammas)
        ])

        results_manager.save_phase2_sample_difficulty(
            alphas=alphas, betas=betas, gammas=gammas,
            t_optimals=t_optimals, labels=labels,
            r2_scores=r2_scores, fit_mse=fit_mse_arr,
        )

        # ANN baseline acc: read from ResultsManager if Phase 1 ran first
        ann_baseline_acc = getattr(results_manager, "_ann_best_acc", 0.0)
        results_manager.save_phase2_fit_quality(
            r2_scores=r2_scores,
            fit_mse=fit_mse_arr,
            ann_baseline_acc=ann_baseline_acc,
            snn_max_t_acc=snn_max_t_acc,
        )

        results_manager.save_phase2_summary(
            num_samples=len(alphas),
            num_thresholds=len(threshold_list),
            conversion_time_seconds=conversion_time,
            profiling_time_seconds=profiling_time,
        )

    return results


def _compute_cumulative_accuracy(performances: np.ndarray) -> np.ndarray:
    """Convert binary correctness to cumulative accuracy rate.

    For each sample, compute the running fraction of correct predictions
    up to each timestep. This gives a smoother signal for curve fitting.

    If a sample has performances [0, 0, 1, 1, 1, 1, 1] at timesteps
    [1, 2, 4, 8, 16, 32, 64], the cumulative accuracy is the running
    mean: [0, 0, 0.33, 0.5, 0.6, 0.67, 0.71]

    Actually, for better scaling law behavior, we use a sliding window
    approach: at timestep T, accuracy = fraction of recent trials that
    were correct. But simplest: just use the raw binary correctness
    smoothed with cumulative max (once correct, stays correct).
    """
    # Use cumulative max: if correct at T, also correct at all T' > T
    # This is a reasonable assumption for rate-coded SNNs
    cummax = np.maximum.accumulate(performances, axis=1)

    # Convert to a continuous scale by using exponential smoothing
    # Actually, let's use the simplest approach: cumulative mean
    # At position i, value = mean(performances[:i+1])
    cumsum = np.cumsum(cummax, axis=1)
    counts = np.arange(1, cummax.shape[1] + 1).reshape(1, -1)
    smooth = cumsum / counts

    return smooth


def _make_calibration_loader(
    train_loader: DataLoader, config: dict
) -> DataLoader:
    """Extract a small calibration subset from the training loader."""
    num_samples = config["conversion"].get("calibration_samples", 1024)
    batch_size = 64

    all_images = []
    all_labels = []
    count = 0

    for images, labels in train_loader:
        all_images.append(images)
        all_labels.append(labels)
        count += images.size(0)
        if count >= num_samples:
            break

    images = torch.cat(all_images, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _log_difficulty_distribution(
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    timesteps: list[int],
    config: dict,
    logger,
):
    """Log the distribution of sample difficulties."""
    from ..scaling.curve_fitting import compute_optimal_timestep

    target_acc = config["scaling_law"]["target_accuracy"]
    max_t = config["snn"]["max_timestep"]

    optimal_timesteps = np.array([
        compute_optimal_timestep(a, b, g, target_acc, max_timestep=max_t)
        for a, b, g in zip(alphas, betas, gammas)
    ])

    logger.info("Sample difficulty distribution (T_optimal):")
    for t in timesteps:
        frac = (optimal_timesteps <= t).mean()
        logger.info(f"  Samples needing T<={t:3d}: {frac*100:.1f}%")

    logger.info(f"  Mean T_optimal: {optimal_timesteps.mean():.1f}")
    logger.info(f"  Median T_optimal: {np.median(optimal_timesteps):.1f}")


def load_profiling_results(save_dir: str) -> dict[str, np.ndarray]:
    """Load previously saved profiling results.

    Args:
        save_dir: Directory containing profiling_results.npz.

    Returns:
        Dictionary with profiling data.
    """
    path = Path(save_dir) / "profiling_results.npz"
    if not path.exists():
        raise FileNotFoundError(f"Profiling results not found at {path}")

    data = np.load(path)
    result = {
        "alphas": data["alphas"],
        "betas": data["betas"],
        "gammas": data["gammas"],
        "performances": data["performances"],
        "timesteps": data["timesteps"],
        "labels": data["labels"],
    }
    # Include new fields if present (backwards-compatible)
    if "r2_scores" in data:
        result["r2_scores"] = data["r2_scores"]
    if "fit_mse" in data:
        result["fit_mse"] = data["fit_mse"]
    if "snn_max_t_acc" in data:
        result["snn_max_t_acc"] = float(data["snn_max_t_acc"])
    return result
