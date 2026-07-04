"""Phase 3: Joint Training of Complexity Predictor + Multi-Exit SNN.

Using the profiling data from Phase 2, this phase trains:
1. The Complexity Predictor to predict (α, β, γ) from raw images
2. The Multi-Exit SNN's exit branches for classification at each timestep

The training alternates between:
- Predictor loss: MSE on scaling law parameters + energy regularization
- Exit branch loss: CrossEntropy at each exit point, weighted by importance

The SNN backbone (converted weights) is frozen — only exit branches and
the predictor are trained.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..predictor.complexity_predictor import ComplexityPredictor
from ..spiking.snn_model import SNNModel
from ..spiking.multi_exit_snn import MultiExitSNN
from ..scaling.scaling_law import ScalingLawModule
from ..utils.logging import setup_logger
from ..utils.results_manager import ResultsManager


def train_joint(
    snn_model: SNNModel,
    train_loader: DataLoader,
    profiling_results: dict[str, np.ndarray],
    config: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    results_manager: Optional[ResultsManager] = None,
) -> tuple[MultiExitSNN, ComplexityPredictor]:
    """Joint training of Multi-Exit SNN + Complexity Predictor (Phase 3).

    Args:
        snn_model: Converted SNN model (backbone frozen).
        train_loader: Training data loader.
        profiling_results: Dict from Phase 2 with 'alphas', 'betas', 'gammas'.
        config: Configuration dictionary.
        device: Training device.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory for logs.
        results_manager: Optional ResultsManager for CSV/metric logging.

    Returns:
        Tuple of (trained MultiExitSNN, trained ComplexityPredictor).
    """
    logger = setup_logger("phase3_joint", log_dir)
    writer = SummaryWriter(log_dir=str(Path(log_dir) / "phase3"))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Config
    pred_cfg = config["predictor"]
    snn_cfg = config["snn"]
    scaling_cfg = config["scaling_law"]
    dataset_cfg = config["dataset"]

    exit_timesteps = snn_cfg["exit_points"]
    num_classes = dataset_cfg["num_classes"]
    image_size = dataset_cfg["image_size"]
    epochs = pred_cfg["epochs"]
    lr = pred_cfg["lr"]
    batch_size = pred_cfg["batch_size"]

    logger.info(f"Phase 3: Joint training for {epochs} epochs")
    logger.info(f"Exit points: {exit_timesteps}, Num classes: {num_classes}")

    # Build Multi-Exit SNN (backbone frozen)
    multi_exit_snn = MultiExitSNN(
        snn_model=snn_model,
        exit_timesteps=exit_timesteps,
        num_classes=num_classes,
        hidden_dim=128,
    ).to(device)

    # Freeze SNN backbone — only train exit branches
    for param in multi_exit_snn.snn.parameters():
        param.requires_grad = False

    # Build Complexity Predictor
    predictor = ComplexityPredictor(
        in_channels=3,
        image_size=image_size,
        hidden_dims=pred_cfg["hidden_dims"],
        num_params=3,
    ).to(device)

    # Build Scaling Law Module
    scaling_module = ScalingLawModule(
        exit_timesteps=exit_timesteps,
        target_accuracy=scaling_cfg["target_accuracy"],
        energy_weight=config["inference"].get("energy_weight", 0.5),
    ).to(device)

    # Prepare profiling labels as tensors
    target_alphas = torch.tensor(profiling_results["alphas"], dtype=torch.float32)
    target_betas = torch.tensor(profiling_results["betas"], dtype=torch.float32)
    target_gammas = torch.tensor(profiling_results["gammas"], dtype=torch.float32)

    # Optimizers
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=lr, weight_decay=1e-4)
    exit_optimizer = optim.Adam(
        multi_exit_snn.exit_branches.parameters(), lr=lr, weight_decay=1e-4
    )

    # Schedulers
    predictor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        predictor_optimizer, T_max=epochs
    )
    exit_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        exit_optimizer, T_max=epochs
    )

    # Loss functions
    ce_criterion = nn.CrossEntropyLoss()

    best_combined_metric = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        # Train one epoch
        metrics = _train_epoch_joint(
            multi_exit_snn=multi_exit_snn,
            predictor=predictor,
            scaling_module=scaling_module,
            train_loader=train_loader,
            target_alphas=target_alphas,
            target_betas=target_betas,
            target_gammas=target_gammas,
            ce_criterion=ce_criterion,
            predictor_optimizer=predictor_optimizer,
            exit_optimizer=exit_optimizer,
            exit_timesteps=exit_timesteps,
            device=device,
        )

        # Step schedulers
        predictor_scheduler.step()
        exit_scheduler.step()

        # Log metrics
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Pred Loss: {metrics['pred_loss']:.4f} | "
            f"Exit Loss: {metrics['exit_loss']:.4f} | "
            f"Avg T_pred: {metrics['avg_t_predicted']:.1f} | "
            f"Exit Accs: {_format_exit_accs(metrics['exit_accuracies'])}"
        )

        writer.add_scalar("predictor/loss", metrics["pred_loss"], epoch)
        writer.add_scalar("predictor/param_loss", metrics["param_loss"], epoch)
        writer.add_scalar("predictor/energy_loss", metrics["energy_loss"], epoch)
        writer.add_scalar("predictor/avg_t_predicted", metrics["avg_t_predicted"], epoch)
        writer.add_scalar("exits/loss", metrics["exit_loss"], epoch)
        for t, acc in metrics["exit_accuracies"].items():
            writer.add_scalar(f"exits/accuracy_T{t}", acc, epoch)

        # Log to ResultsManager
        if results_manager is not None:
            results_manager.log_phase3_epoch(
                epoch=epoch + 1,
                pred_loss=metrics["pred_loss"],
                param_loss=metrics["param_loss"],
                energy_loss=metrics["energy_loss"],
                exit_loss=metrics["exit_loss"],
                avg_t_predicted=metrics["avg_t_predicted"],
                exit_accuracies=metrics["exit_accuracies"],
            )

        # Save best (use combination of final exit accuracy + energy savings)
        final_acc = metrics["exit_accuracies"].get(exit_timesteps[-1], 0)
        energy_savings = 1.0 - (metrics["avg_t_predicted"] / max(exit_timesteps))
        combined = final_acc * 0.7 + energy_savings * 100 * 0.3

        if combined > best_combined_metric:
            best_combined_metric = combined
            _save_phase3_checkpoint(
                checkpoint_dir / "phase3_best.pth",
                multi_exit_snn, predictor, epoch, metrics
            )

    # Save final
    _save_phase3_checkpoint(
        checkpoint_dir / "phase3_final.pth",
        multi_exit_snn, predictor, epochs - 1, metrics
    )

    logger.info(f"Phase 3 complete. Best combined metric: {best_combined_metric:.2f}")
    writer.close()

    # Save summary to ResultsManager
    if results_manager is not None:
        training_time = time.time() - start_time
        predictor_params = sum(p.numel() for p in predictor.parameters())
        exit_params = sum(p.numel() for p in multi_exit_snn.exit_branches.parameters())
        best_exit_accs = metrics["exit_accuracies"]  # from last epoch
        results_manager.save_phase3_summary(
            best_exit_accuracies=best_exit_accs,
            best_avg_t=metrics["avg_t_predicted"],
            total_epochs=epochs,
            predictor_params=predictor_params,
            exit_branch_params=exit_params,
            training_time_seconds=training_time,
        )

    # Load best checkpoint
    best_ckpt = torch.load(checkpoint_dir / "phase3_best.pth", map_location=device)
    multi_exit_snn.exit_branches.load_state_dict(best_ckpt["exit_branches_state_dict"])
    predictor.load_state_dict(best_ckpt["predictor_state_dict"])

    return multi_exit_snn, predictor


def _train_epoch_joint(
    multi_exit_snn: MultiExitSNN,
    predictor: ComplexityPredictor,
    scaling_module: ScalingLawModule,
    train_loader: DataLoader,
    target_alphas: torch.Tensor,
    target_betas: torch.Tensor,
    target_gammas: torch.Tensor,
    ce_criterion: nn.Module,
    predictor_optimizer: optim.Optimizer,
    exit_optimizer: optim.Optimizer,
    exit_timesteps: list[int],
    device: torch.device,
) -> dict:
    """Train one epoch jointly."""
    multi_exit_snn.train()
    predictor.train()

    # Tracking
    total_pred_loss = 0.0
    total_param_loss = 0.0
    total_energy_loss = 0.0
    total_exit_loss = 0.0
    total_samples = 0
    all_t_predicted = []
    exit_correct = {t: 0 for t in exit_timesteps}
    exit_total = {t: 0 for t in exit_timesteps}

    sample_idx = 0

    for images, targets in tqdm(train_loader, desc="Phase 3 Training", leave=False):
        batch_size = images.size(0)
        images, targets = images.to(device), targets.to(device)

        # Get ground-truth scaling parameters for this batch
        batch_end = min(sample_idx + batch_size, len(target_alphas))
        if sample_idx >= len(target_alphas):
            # Wrap around if loader has more batches than profiled samples
            sample_idx = 0
            batch_end = batch_size

        gt_alpha = target_alphas[sample_idx:batch_end].to(device)
        gt_beta = target_betas[sample_idx:batch_end].to(device)
        gt_gamma = target_gammas[sample_idx:batch_end].to(device)

        # Adjust batch if sizes don't match (edge case)
        actual_batch = min(batch_size, len(gt_alpha))
        if actual_batch < batch_size:
            images = images[:actual_batch]
            targets = targets[:actual_batch]
            gt_alpha = gt_alpha[:actual_batch]
            gt_beta = gt_beta[:actual_batch]
            gt_gamma = gt_gamma[:actual_batch]

        # --- Train Complexity Predictor ---
        predictor_optimizer.zero_grad()

        pred_params = predictor(images)
        target_params = {
            "alpha": gt_alpha,
            "beta": gt_beta,
            "gamma": gt_gamma,
        }
        scaling_losses = scaling_module.compute_loss(pred_params, target_params)
        pred_loss = scaling_losses["total"]
        pred_loss.backward()
        predictor_optimizer.step()

        # Track predicted timesteps
        with torch.no_grad():
            t_pred = scaling_module.compute_t_optimal(
                pred_params["alpha"].detach(),
                pred_params["beta"].detach(),
                pred_params["gamma"].detach(),
            )
            all_t_predicted.extend(t_pred.cpu().tolist())

        # --- Train Exit Branches ---
        exit_optimizer.zero_grad()

        # Forward through multi-exit SNN (backbone frozen)
        exit_outputs = multi_exit_snn(images)

        # Multi-exit loss: weighted sum of CE losses at each exit
        exit_loss = torch.tensor(0.0, device=device)
        num_exits = len(exit_timesteps)

        for i, t in enumerate(exit_timesteps):
            if t in exit_outputs:
                logits = exit_outputs[t]
                # Later exits are more important (higher weight)
                weight = (i + 1) / num_exits
                exit_loss = exit_loss + weight * ce_criterion(logits, targets)

                # Track accuracy
                preds = logits.argmax(dim=1)
                exit_correct[t] += (preds == targets).sum().item()
                exit_total[t] += targets.size(0)

        exit_loss.backward()
        exit_optimizer.step()

        # Accumulate metrics
        total_pred_loss += pred_loss.item() * actual_batch
        total_param_loss += scaling_losses["param_loss"].item() * actual_batch
        total_energy_loss += scaling_losses["energy_loss"].item() * actual_batch
        total_exit_loss += exit_loss.item() * actual_batch
        total_samples += actual_batch
        sample_idx += batch_size

    # Compute epoch metrics
    exit_accuracies = {
        t: 100.0 * exit_correct[t] / max(exit_total[t], 1)
        for t in exit_timesteps
    }

    return {
        "pred_loss": total_pred_loss / max(total_samples, 1),
        "param_loss": total_param_loss / max(total_samples, 1),
        "energy_loss": total_energy_loss / max(total_samples, 1),
        "exit_loss": total_exit_loss / max(total_samples, 1),
        "avg_t_predicted": np.mean(all_t_predicted) if all_t_predicted else 0,
        "exit_accuracies": exit_accuracies,
    }


def _format_exit_accs(exit_accs: dict[int, float]) -> str:
    """Format exit accuracies for logging."""
    parts = [f"T{t}={acc:.1f}%" for t, acc in sorted(exit_accs.items())]
    return " | ".join(parts)


def _save_phase3_checkpoint(
    path: Path,
    multi_exit_snn: MultiExitSNN,
    predictor: ComplexityPredictor,
    epoch: int,
    metrics: dict,
):
    """Save Phase 3 checkpoint."""
    torch.save({
        "exit_branches_state_dict": multi_exit_snn.exit_branches.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def load_phase3_checkpoint(
    path: str,
    multi_exit_snn: MultiExitSNN,
    predictor: ComplexityPredictor,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load Phase 3 checkpoint.

    Args:
        path: Path to checkpoint file.
        multi_exit_snn: MultiExitSNN to load exit branch weights into.
        predictor: ComplexityPredictor to load weights into.
        device: Device to load on.

    Returns:
        Checkpoint metadata dict.
    """
    ckpt = torch.load(path, map_location=device)
    multi_exit_snn.exit_branches.load_state_dict(ckpt["exit_branches_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    return ckpt
