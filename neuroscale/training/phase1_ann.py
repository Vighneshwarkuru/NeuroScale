"""Phase 1: ANN Pretraining.

Standard training of the ANN (ResNet/VGG) on the target dataset.
This produces the pretrained weights that will be converted to SNN.

Supports:
- Cosine annealing and step LR schedules
- Warmup epochs
- Checkpoint saving (best + periodic)
- TensorBoard logging
- ResultsManager integration for CSV/metrics/plots
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..models.factory import get_ann_model
from ..datasets.factory import get_dataloaders
from ..utils.metrics import accuracy
from ..utils.logging import setup_logger
from ..utils.results_manager import ResultsManager


def train_ann(
    config: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    resume_from: Optional[str] = None,
    results_manager: Optional[ResultsManager] = None,
) -> nn.Module:
    """Train the ANN model (Phase 1).

    Args:
        config: Full configuration dictionary.
        device: Training device.
        checkpoint_dir: Directory to save model checkpoints.
        log_dir: Directory for TensorBoard logs.
        resume_from: Path to checkpoint to resume from.
        results_manager: Optional ResultsManager for CSV/metric logging.

    Returns:
        Trained ANN model.
    """
    logger = setup_logger("phase1_ann", log_dir)
    writer = SummaryWriter(log_dir=str(Path(log_dir) / "phase1"))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create model and data
    model = get_ann_model(config).to(device)
    train_loader, test_loader = get_dataloaders(config)

    # Training config
    model_cfg = config["model"]
    epochs = model_cfg["epochs"]
    lr = model_cfg["lr"]
    weight_decay = model_cfg.get("weight_decay", 1e-4)
    momentum = model_cfg.get("momentum", 0.9)
    num_classes = config["dataset"]["num_classes"]

    # Use top-5 only when there are enough classes
    compute_top5 = num_classes >= 5

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # LR scheduler
    scheduler = _get_scheduler(optimizer, model_cfg, epochs)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0
    best_top5_acc = 0.0
    if resume_from is not None:
        start_epoch, best_acc = _load_checkpoint(
            resume_from, model, optimizer, scheduler, device
        )
        logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    logger.info(f"Starting Phase 1 training: {model_cfg['ann']} on {config['dataset']['name']}")
    logger.info(f"Epochs: {epochs}, LR: {lr}, Batch size: {model_cfg['batch_size']}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # Train one epoch
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate — returns top-1 and top-5
        test_loss, test_acc, test_top5_acc = _evaluate(
            model, test_loader, criterion, device, compute_top5=compute_top5
        )

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        top5_str = f", Top-5: {test_top5_acc:.2f}%" if compute_top5 else ""
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%{top5_str} | "
            f"LR: {current_lr:.6f}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", test_acc, epoch)
        if compute_top5:
            writer.add_scalar("test/top5_accuracy", test_top5_acc, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # Log to ResultsManager
        if results_manager is not None:
            results_manager.log_phase1_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                lr=current_lr,
                test_top5_acc=test_top5_acc,
            )

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_top5_acc = test_top5_acc
            best_epoch = epoch + 1
            _save_checkpoint(
                checkpoint_dir / "ann_best.pth",
                model, optimizer, scheduler, epoch, best_acc
            )

        # Periodic save
        if (epoch + 1) % 50 == 0:
            _save_checkpoint(
                checkpoint_dir / f"ann_epoch{epoch+1}.pth",
                model, optimizer, scheduler, epoch, best_acc
            )

    # Save final model
    _save_checkpoint(
        checkpoint_dir / "ann_final.pth",
        model, optimizer, scheduler, epochs - 1, best_acc
    )

    training_time = time.time() - start_time
    logger.info(f"Phase 1 complete. Best test accuracy: {best_acc:.2f}% (epoch {best_epoch})")
    if compute_top5:
        logger.info(f"  Best top-5 accuracy: {best_top5_acc:.2f}%")
    logger.info(f"Training time: {training_time:.1f}s")

    # Save summary to ResultsManager
    if results_manager is not None:
        results_manager.save_phase1_summary(
            best_acc=best_acc,
            best_epoch=best_epoch,
            total_epochs=epochs,
            model_name=model_cfg["ann"],
            total_params=total_params,
            training_time_seconds=training_time,
            best_top5_acc=best_top5_acc,
        )

    writer.close()

    # Load best weights
    best_ckpt = torch.load(checkpoint_dir / "ann_best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    return model


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_top5: bool = True,
) -> tuple[float, float, float]:
    """Evaluate model on test/val set.

    Returns:
        (avg_loss, top1_acc, top5_acc) — top5_acc is 0.0 if compute_top5=False.
    """
    model.eval()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            total_correct_top1 += (outputs.argmax(dim=1) == targets).sum().item()

            if compute_top5:
                topk_res = accuracy(outputs, targets, topk=(1, 5))
                # accuracy() returns per-batch %, scale back to count
                total_correct_top5 += topk_res[1] / 100.0 * images.size(0)

            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    top1_acc = 100.0 * total_correct_top1 / total_samples
    top5_acc = 100.0 * total_correct_top5 / total_samples if compute_top5 else 0.0
    return avg_loss, top1_acc, top5_acc


def _get_scheduler(
    optimizer: optim.Optimizer, model_cfg: dict, epochs: int
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create LR scheduler from config."""
    schedule = model_cfg.get("lr_schedule", "cosine")

    if schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif schedule == "step":
        step_size = model_cfg.get("lr_step_size", 30)
        gamma = model_cfg.get("lr_gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif schedule == "multistep":
        milestones = model_cfg.get("lr_milestones", [100, 150])
        gamma = model_cfg.get("lr_gamma", 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        return None


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
):
    """Save training checkpoint."""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Load training checkpoint. Returns (start_epoch, best_acc)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"] + 1, ckpt.get("best_acc", 0.0)
