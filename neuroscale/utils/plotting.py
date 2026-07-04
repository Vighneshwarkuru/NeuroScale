"""Comprehensive plotting module for NeuroScale++.

Generates all possible visualizations:
- Phase 1: Training/test loss & accuracy curves, LR schedule
- Phase 2: Accuracy vs timestep, difficulty histograms, scaling law fits
- Phase 3: Predictor loss curves, exit accuracy progression, T_optimal dist
- Evaluation: Exit distribution pie/bar, energy savings, confusion matrix,
              per-class accuracy, confidence histograms, comparison charts

All plots saved as both PNG (300dpi) and PDF for paper quality.
"""

import numpy as np
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# Style setup
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save_fig(fig, plots_dir: Path, name: str):
    """Save figure as PNG and PDF."""
    fig.savefig(plots_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 PLOTS: ANN Training
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase1_training_curves(metrics: list[dict], plots_dir: Path):
    """Plot training and test loss/accuracy curves."""
    if not metrics:
        return

    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    test_loss = [m["test_loss"] for m in metrics]
    train_acc = [m["train_acc"] for m in metrics]
    test_acc = [m["test_acc"] for m in metrics]
    lr_vals = [m["lr"] for m in metrics]

    # Combined figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss curves
    axes[0].plot(epochs, train_loss, "b-", linewidth=1.5, label="Train Loss")
    axes[0].plot(epochs, test_loss, "r-", linewidth=1.5, label="Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Test Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, train_acc, "b-", linewidth=1.5, label="Train Acc")
    axes[1].plot(epochs, test_acc, "r-", linewidth=1.5, label="Test Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR schedule
    axes[2].plot(epochs, lr_vals, "g-", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Phase 1: ANN Pretraining", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "phase1_training_curves")


def plot_phase1_loss_separate(metrics: list[dict], plots_dir: Path):
    """Individual high-quality loss plot."""
    if not metrics:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [m["train_loss"] for m in metrics], "b-", lw=2, label="Train")
    ax.plot(epochs, [m["test_loss"] for m in metrics], "r-", lw=2, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Phase 1: Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, plots_dir, "phase1_loss")


def plot_phase1_accuracy_separate(metrics: list[dict], plots_dir: Path):
    """Individual high-quality accuracy plot."""
    if not metrics:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [m["train_acc"] for m in metrics], "b-", lw=2, label="Train")
    ax.plot(epochs, [m["test_acc"] for m in metrics], "r-", lw=2, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phase 1: Accuracy Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    best_epoch = max(range(len(metrics)), key=lambda i: metrics[i]["test_acc"])
    best_acc = metrics[best_epoch]["test_acc"]
    ax.axhline(y=best_acc, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(f"Best: {best_acc:.2f}%", xy=(best_epoch, best_acc),
                xytext=(best_epoch + 5, best_acc - 3), fontsize=9)
    _save_fig(fig, plots_dir, "phase1_accuracy")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 PLOTS: SNN Profiling
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase2_accuracy_vs_timestep(
    timesteps: np.ndarray,
    mean_acc: np.ndarray,
    std_acc: np.ndarray,
    plots_dir: Path,
):
    """Plot SNN accuracy vs timestep with error bands."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, mean_acc * 100, "b-o", linewidth=2, markersize=8, label="Mean Accuracy")
    ax.fill_between(
        timesteps,
        (mean_acc - std_acc) * 100,
        (mean_acc + std_acc) * 100,
        alpha=0.2, color="blue", label="±1 Std Dev"
    )
    ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phase 2: SNN Accuracy vs. Timesteps")
    ax.set_xscale("log", base=2)
    ax.set_xticks(timesteps)
    ax.set_xticklabels([str(int(t)) for t in timesteps])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, plots_dir, "phase2_accuracy_vs_timestep")


def plot_phase2_difficulty_histogram(
    t_optimals: np.ndarray,
    exit_timesteps: list[int],
    plots_dir: Path,
):
    """Plot histogram of sample difficulty (T_optimal distribution)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.array(exit_timesteps + [exit_timesteps[-1] + exit_timesteps[-1]]) - 0.5
    ax.hist(t_optimals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    for t in exit_timesteps:
        ax.axvline(x=t, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Optimal Timestep (T_optimal)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Phase 2: Sample Difficulty Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics text
    stats_text = (
        f"Mean: {t_optimals.mean():.1f}\n"
        f"Median: {np.median(t_optimals):.1f}\n"
        f"Std: {t_optimals.std():.1f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    _save_fig(fig, plots_dir, "phase2_difficulty_histogram")


def plot_phase2_scaling_law_samples(
    timesteps: np.ndarray,
    performances: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    plots_dir: Path,
    num_samples: int = 20,
):
    """Plot scaling law curves for individual samples (actual vs fitted)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Select samples: easy, medium, hard, and one random
    t_optimals = []
    for a, b, g in zip(alphas, betas, gammas):
        t_opt = ((0.9 - g) / max(a, 1e-6)) ** (1.0 / max(b, 0.01))
        t_optimals.append(np.clip(t_opt, 1, 64))
    t_optimals = np.array(t_optimals)

    categories = [
        ("Easy Samples (T_opt < 8)", t_optimals < 8),
        ("Medium Samples (8 ≤ T_opt < 32)", (t_optimals >= 8) & (t_optimals < 32)),
        ("Hard Samples (T_opt ≥ 32)", t_optimals >= 32),
        ("Random Samples", np.ones(len(t_optimals), dtype=bool)),
    ]

    t_fine = np.linspace(1, 64, 200)

    for ax_idx, (title, mask) in enumerate(categories):
        ax = axes[ax_idx]
        indices = np.where(mask)[0]
        if len(indices) == 0:
            ax.set_title(f"{title} (no samples)")
            continue
        selected = np.random.choice(indices, min(5, len(indices)), replace=False)

        for i, idx in enumerate(selected):
            # Actual performance
            ax.scatter(timesteps, performances[idx] * 100, s=30, alpha=0.7)
            # Fitted curve
            fitted = alphas[idx] * t_fine ** betas[idx] + gammas[idx]
            ax.plot(t_fine, fitted * 100, "--", alpha=0.7, linewidth=1.2)

        ax.set_xlabel("Timesteps (T)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.set_xlim(0, 68)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 2: Per-Sample Scaling Law Fits", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "phase2_scaling_law_samples")


def plot_phase2_parameter_distributions(
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    plots_dir: Path,
):
    """Plot distributions of fitted scaling law parameters."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    params = [
        (alphas, "α (Amplitude)", "steelblue"),
        (betas, "β (Exponent)", "forestgreen"),
        (gammas, "γ (Baseline)", "coral"),
    ]
    for ax, (data, label, color) in zip(axes, params):
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(data.mean(), color="black", linestyle="--", lw=1.5, label=f"Mean={data.mean():.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phase 2: Scaling Law Parameter Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "phase2_parameter_distributions")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 PLOTS: Joint Training
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase3_training_curves(metrics: list[dict], plots_dir: Path):
    """Plot Phase 3 loss curves and exit accuracy progression."""
    if not metrics:
        return

    epochs = [m["epoch"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Predictor loss
    axes[0, 0].plot(epochs, [m["pred_loss"] for m in metrics], "b-", lw=1.5, label="Total Pred Loss")
    axes[0, 0].plot(epochs, [m["param_loss"] for m in metrics], "g--", lw=1, label="Param Loss")
    axes[0, 0].plot(epochs, [m["energy_loss"] for m in metrics], "r--", lw=1, label="Energy Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Predictor Losses")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Exit loss
    axes[0, 1].plot(epochs, [m["exit_loss"] for m in metrics], "purple", lw=1.5)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Exit Branch Loss (Cross-Entropy)")
    axes[0, 1].grid(True, alpha=0.3)

    # Exit accuracies over time
    exit_keys = [k for k in metrics[0].keys() if k.startswith("exit_acc_T")]
    colors = plt.cm.viridis(np.linspace(0, 1, len(exit_keys)))
    for i, key in enumerate(exit_keys):
        t_label = key.replace("exit_acc_", "")
        vals = [m.get(key, 0) for m in metrics]
        axes[1, 0].plot(epochs, vals, color=colors[i], lw=1.5, label=t_label)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_title("Exit Branch Accuracies Over Training")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3)

    # Average predicted timestep
    axes[1, 1].plot(epochs, [m["avg_t_predicted"] for m in metrics], "darkorange", lw=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Avg Predicted Timestep")
    axes[1, 1].set_title("Average T_optimal Predicted by Complexity Predictor")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Phase 3: Joint Training Progress", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "phase3_training_curves")


def plot_phase3_predictor_loss(metrics: list[dict], plots_dir: Path):
    """Separate detailed predictor loss plot."""
    if not metrics:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [m["pred_loss"] for m in metrics], "b-", lw=2, label="Total")
    ax.plot(epochs, [m["param_loss"] for m in metrics], "g-", lw=1.5, label="Parameter MSE")
    ax.plot(epochs, [m["energy_loss"] for m in metrics], "r-", lw=1.5, label="Energy Reg.")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Phase 3: Complexity Predictor Loss Breakdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, plots_dir, "phase3_predictor_loss")


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_exit_distribution_bar(exit_metrics: dict, plots_dir: Path):
    """Bar chart of sample distribution across exit timesteps."""
    if not exit_metrics:
        return
    timesteps = sorted(exit_metrics.keys())
    counts = [exit_metrics[t]["count"] for t in timesteps]
    fractions = [exit_metrics[t]["fraction"] * 100 for t in timesteps]
    accuracies = [exit_metrics[t]["accuracy"] for t in timesteps]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    bars = ax1.bar(
        [str(t) for t in timesteps], fractions,
        color="steelblue", edgecolor="white", alpha=0.8
    )
    ax1.set_xlabel("Exit Timestep")
    ax1.set_ylabel("Samples (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_title("Evaluation: Exit Point Distribution & Accuracy")

    # Add count labels on bars
    for bar, count, frac in zip(bars, counts, fractions):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{count}\n({frac:.1f}%)", ha="center", va="bottom", fontsize=9)

    # Secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot([str(t) for t in timesteps], accuracies, "ro-", lw=2, markersize=8, label="Accuracy")
    ax2.set_ylabel("Accuracy (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_exit_distribution")


def plot_exit_distribution_pie(exit_metrics: dict, plots_dir: Path):
    """Pie chart of exit distribution."""
    if not exit_metrics:
        return
    timesteps = sorted(exit_metrics.keys())
    fractions = [exit_metrics[t]["fraction"] for t in timesteps]
    labels = [f"T={t}\n({exit_metrics[t]['fraction']*100:.1f}%)" for t in timesteps]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(timesteps)))

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        fractions, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title("Exit Timestep Distribution (Adaptive Inference)")
    _save_fig(fig, plots_dir, "eval_exit_pie")


def plot_energy_savings_comparison(adaptive: dict, baseline: dict, plots_dir: Path):
    """Bar chart comparing adaptive vs baseline metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Accuracy comparison
    methods = ["Baseline\n(Fixed T_max)", "NeuroScale++\n(Adaptive)"]
    accs = [baseline["accuracy"], adaptive["accuracy"]]
    colors = ["#FF6B6B", "#4ECDC4"]
    axes[0].bar(methods, accs, color=colors, edgecolor="white", width=0.5)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_ylim(min(accs) - 5, max(accs) + 3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Avg timestep comparison
    avg_ts = [baseline["avg_timestep"], adaptive["avg_timestep"]]
    axes[1].bar(methods, avg_ts, color=colors, edgecolor="white", width=0.5)
    axes[1].set_ylabel("Average Timesteps")
    axes[1].set_title("Computational Cost")
    for i, v in enumerate(avg_ts):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Energy savings & speedup
    metrics_names = ["Energy\nSavings", "Speedup"]
    metric_vals = [adaptive["energy_savings"] * 100, adaptive["speedup"]]
    metric_colors = ["#45B7D1", "#96CEB4"]
    bars = axes[2].bar(metrics_names, metric_vals, color=metric_colors, edgecolor="white", width=0.5)
    axes[2].set_ylabel("Value")
    axes[2].set_title("Efficiency Gains")
    for bar, v in zip(bars, metric_vals):
        unit = "%" if "Savings" in bar.get_x().__class__.__name__ else "x"
        label = f"{v:.1f}%" if v < 10 else f"{v:.1f}x"
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("NeuroScale++ vs Baseline", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_energy_comparison")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    plots_dir: Path,
    class_names: Optional[list[str]] = None,
    max_classes: int = 20,
):
    """Plot confusion matrix heatmap."""
    num_classes = confusion_matrix.shape[0]

    # For large class counts, show only a subset or use smaller cells
    if num_classes > max_classes:
        # Show top confused classes instead
        fig, ax = plt.subplots(figsize=(8, 5))
        # Per-class accuracy
        class_acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1).clip(1)
        worst_classes = np.argsort(class_acc)[:max_classes]
        sub_cm = confusion_matrix[np.ix_(worst_classes, worst_classes)]
        im = ax.imshow(sub_cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix (Top {max_classes} Most Confused Classes)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.6), max(6, num_classes * 0.5)))
        im = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax, shrink=0.8)

        if class_names and len(class_names) <= max_classes:
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)

            # Add text annotations for small matrices
            if num_classes <= 10:
                for i in range(num_classes):
                    for j in range(num_classes):
                        ax.text(j, i, str(confusion_matrix[i, j]),
                                ha="center", va="center", fontsize=8,
                                color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_confusion_matrix")


def plot_confidence_histogram(
    confidences: np.ndarray,
    correct: np.ndarray,
    plots_dir: Path,
):
    """Plot confidence distribution split by correct/incorrect."""
    fig, ax = plt.subplots(figsize=(8, 5))

    correct_conf = confidences[correct.astype(bool)]
    incorrect_conf = confidences[~correct.astype(bool)]

    bins = np.linspace(0, 1, 40)
    ax.hist(correct_conf, bins=bins, alpha=0.7, color="green", label=f"Correct (n={len(correct_conf)})")
    if len(incorrect_conf) > 0:
        ax.hist(incorrect_conf, bins=bins, alpha=0.7, color="red", label=f"Incorrect (n={len(incorrect_conf)})")

    ax.set_xlabel("Confidence (Max Softmax Probability)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save_fig(fig, plots_dir, "eval_confidence_histogram")


def plot_timestep_vs_confidence(
    timesteps_used: np.ndarray,
    confidences: np.ndarray,
    correct: np.ndarray,
    plots_dir: Path,
):
    """Scatter plot of timestep used vs confidence, colored by correctness."""
    fig, ax = plt.subplots(figsize=(8, 5))

    correct_mask = correct.astype(bool)
    ax.scatter(
        timesteps_used[correct_mask], confidences[correct_mask],
        c="green", alpha=0.3, s=10, label="Correct"
    )
    if (~correct_mask).sum() > 0:
        ax.scatter(
            timesteps_used[~correct_mask], confidences[~correct_mask],
            c="red", alpha=0.5, s=20, marker="x", label="Incorrect"
        )

    ax.set_xlabel("Timesteps Used")
    ax.set_ylabel("Confidence")
    ax.set_title("Timestep vs Confidence (Colored by Correctness)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, plots_dir, "eval_timestep_vs_confidence")


def plot_per_class_accuracy(
    per_class_acc: np.ndarray,
    per_class_avg_t: np.ndarray,
    plots_dir: Path,
    class_names: Optional[list[str]] = None,
    top_n: int = 20,
):
    """Plot per-class accuracy and average timestep."""
    num_classes = len(per_class_acc)
    show_n = min(num_classes, top_n)

    # Sort by accuracy (show worst classes)
    sorted_idx = np.argsort(per_class_acc)
    worst_idx = sorted_idx[:show_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, show_n * 0.3)))

    y_pos = range(show_n)
    labels = [class_names[i] if class_names else f"Class {i}" for i in worst_idx]

    # Accuracy bar chart (horizontal)
    axes[0].barh(y_pos, per_class_acc[worst_idx], color="steelblue", edgecolor="white")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_title(f"Per-Class Accuracy (Bottom {show_n})")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Avg timestep bar chart
    axes[1].barh(y_pos, per_class_avg_t[worst_idx], color="coral", edgecolor="white")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Average Timesteps Used")
    axes[1].set_title(f"Per-Class Avg Timestep (Bottom {show_n} by Acc)")
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.suptitle("Per-Class Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_per_class_analysis")


def plot_r2_histogram(
    r2_scores: np.ndarray,
    plots_dir: Path,
):
    """Histogram of per-sample R² scores for scaling law fit quality.

    Shows how well the power-law model fits each sample's accuracy curve.
    R² near 1.0 = good fit; values below ~0.8 flag poor fits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(r2_scores, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(r2_scores.mean(), color="red", linestyle="--", lw=1.5,
                    label=f"Mean = {r2_scores.mean():.3f}")
    axes[0].axvline(np.median(r2_scores), color="orange", linestyle="--", lw=1.5,
                    label=f"Median = {np.median(r2_scores):.3f}")
    axes[0].axvline(0.8, color="gray", linestyle=":", lw=1.2, label="R²=0.8 threshold")
    axes[0].set_xlabel("R² (Coefficient of Determination)")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_title("Phase 2: Scaling Law Fit Quality (R²)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Cumulative distribution
    sorted_r2 = np.sort(r2_scores)
    cdf = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2)
    axes[1].plot(sorted_r2, cdf * 100, "b-", lw=2)
    axes[1].axvline(0.9, color="red", linestyle="--", lw=1.2,
                    label=f"≥0.9: {(r2_scores >= 0.9).mean()*100:.1f}%")
    axes[1].axvline(0.8, color="orange", linestyle="--", lw=1.2,
                    label=f"≥0.8: {(r2_scores >= 0.8).mean()*100:.1f}%")
    axes[1].set_xlabel("R²")
    axes[1].set_ylabel("Cumulative % of Samples")
    axes[1].set_title("R² Cumulative Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.05, 1.05)

    fig.suptitle("Phase 2: Power-Law Curve Fit Quality", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "phase2_r2_histogram")


def plot_ann_snn_accuracy_comparison(
    ann_acc: float,
    snn_fixed_acc: float,
    snn_adaptive_acc: float,
    plots_dir: Path,
    ann_top5: float = 0.0,
    snn_fixed_top5: float = 0.0,
    snn_adaptive_top5: float = 0.0,
):
    """Bar chart showing ANN → SNN → Adaptive accuracy gap.

    This is the core paper table figure: shows conversion accuracy loss
    and how adaptive inference compares to both the ANN and fixed-T SNN.
    """
    has_top5 = ann_top5 > 0

    if has_top5:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes_raw = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes_raw]

    methods = ["ANN\n(Baseline)", "SNN\n(Fixed T_max)", "SNN\n(Adaptive)"]
    top1_accs = [ann_acc, snn_fixed_acc, snn_adaptive_acc]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    ax = axes[0]
    bars = ax.bar(methods, top1_accs, color=colors, edgecolor="white", width=0.5, zorder=3)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("ANN → SNN Conversion Accuracy")
    y_min = max(0, min(top1_accs) - 5)
    y_max = max(top1_accs) + 4
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)

    for bar, val in zip(bars, top1_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate accuracy gaps with arrows
    if snn_fixed_acc > 0:
        gap1 = ann_acc - snn_fixed_acc
        ax.annotate("", xy=(1, snn_fixed_acc), xytext=(0, ann_acc),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax.text(0.5, (ann_acc + snn_fixed_acc) / 2, f"−{gap1:.2f}%",
                ha="center", va="center", fontsize=9, color="red",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    if has_top5:
        top5_accs = [ann_top5, snn_fixed_top5, snn_adaptive_top5]
        ax2 = axes[1]
        bars2 = ax2.bar(methods, top5_accs, color=colors, edgecolor="white", width=0.5, zorder=3)
        ax2.set_ylabel("Top-5 Accuracy (%)")
        ax2.set_title("ANN → SNN Conversion Top-5 Accuracy")
        y_min2 = max(0, min(top5_accs) - 3)
        y_max2 = min(100, max(top5_accs) + 3)
        ax2.set_ylim(y_min2, y_max2)
        ax2.grid(True, alpha=0.3, axis="y", zorder=0)
        for bar, val in zip(bars2, top5_accs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle("NeuroScale++: ANN → SNN Accuracy Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_ann_snn_accuracy_comparison")


def plot_sops_comparison(
    sops_dict: dict,
    plots_dir: Path,
):
    """Bar chart comparing Synaptic Operations (SOPs) for adaptive vs fixed-T SNN.

    SOPs is the standard energy metric in SNN literature — each bar represents
    total synaptic addition events per sample, normalized to show reduction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    methods = ["SNN\n(Fixed T_max)", "NeuroScale++\n(Adaptive)"]
    sops_vals = [
        sops_dict.get("baseline_sops_per_sample", 0),
        sops_dict.get("adaptive_sops_per_sample", 0),
    ]
    colors = ["#FF5722", "#4CAF50"]

    # SOPs per sample bar chart
    bars = axes[0].bar(methods, sops_vals, color=colors, edgecolor="white", width=0.5, zorder=3)
    axes[0].set_ylabel("SOPs per Sample")
    axes[0].set_title("Synaptic Operations (SOPs) per Sample")
    axes[0].grid(True, alpha=0.3, axis="y", zorder=0)
    for bar, val in zip(bars, sops_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:.2e}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Reduction breakdown (pie / gauge-style bar)
    reduction_pct = sops_dict.get("sops_reduction", 0) * 100
    kept_pct = 100 - reduction_pct
    wedge_colors = ["#4CAF50", "#E0E0E0"]
    axes[1].pie(
        [reduction_pct, kept_pct],
        labels=[f"Saved\n{reduction_pct:.1f}%", f"Used\n{kept_pct:.1f}%"],
        colors=wedge_colors,
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title(f"SOPs Reduction\n(NeuroScale++ vs Fixed T_max)")

    fig.suptitle("Energy Efficiency: Synaptic Operations (SOPs)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_sops_comparison")


def plot_ece_reliability_diagram(
    ece_dict: dict,
    plots_dir: Path,
):
    """ECE reliability diagram (calibration curve).

    A perfectly calibrated model lies on the diagonal.
    Bars above diagonal = overconfident; below = underconfident.
    """
    bin_accs = ece_dict["bin_accs"]
    bin_confs = ece_dict["bin_confs"]
    bin_counts = ece_dict["bin_counts"]
    bin_edges = ece_dict["bin_edges"]
    ece = ece_dict["ece"]
    mce = ece_dict.get("mce", 0.0)

    n_bins = len(bin_accs)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Reliability diagram
    ax = axes[0]
    # Gap bars (overconfidence = red, underconfidence = blue)
    for b in range(n_bins):
        if bin_counts[b] == 0:
            continue
        conf = bin_confs[b]
        acc = bin_accs[b]
        # Draw accuracy bar
        ax.bar(bin_centers[b], acc, width=bin_width * 0.8,
               color="#4CAF50", alpha=0.7, edgecolor="white", zorder=3)
        # Draw gap shading
        gap_color = "#FF5722" if conf > acc else "#2196F3"
        ax.bar(bin_centers[b], conf - acc, bottom=acc, width=bin_width * 0.8,
               color=gap_color, alpha=0.4, edgecolor="white", zorder=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=5)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram\nECE={ece:.4f}  MCE={mce:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Sample count per bin
    ax2 = axes[1]
    bar_colors = ["#4CAF50" if c > 0 else "#E0E0E0" for c in bin_counts]
    ax2.bar(bin_centers, bin_counts, width=bin_width * 0.8,
            color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Confidence Bin Sample Counts")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Model Calibration (Expected Calibration Error)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_ece_reliability_diagram")


def plot_predictor_t_error(
    t_error_dict: dict,
    timesteps_used: np.ndarray,
    plots_dir: Path,
):
    """Visualise how well the predictor assigns the correct timestep.

    Shows: (1) error distribution bar chart broken down by over/under/exact,
    (2) histogram of absolute error magnitude.
    """
    mae = t_error_dict.get("mae", 0.0)
    rmse = t_error_dict.get("rmse", 0.0)
    exact = t_error_dict.get("exact_match_rate", 0.0) * 100
    over = t_error_dict.get("over_allocation_rate", 0.0) * 100
    under = t_error_dict.get("under_allocation_rate", 0.0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Stacked bar: exact / over / under allocation rates
    categories = ["Exact Match\n(T_pred = T_opt)",
                  "Over-allocated\n(T_pred > T_opt)",
                  "Under-allocated\n(T_pred < T_opt)"]
    values = [exact, over, under]
    bar_colors = ["#4CAF50", "#FF9800", "#F44336"]
    bars = axes[0].bar(categories, values, color=bar_colors, edgecolor="white", width=0.5, zorder=3)
    axes[0].set_ylabel("% of Samples")
    axes[0].set_title(f"Predictor Timestep Allocation\nMAE={mae:.2f}  RMSE={rmse:.2f}")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3, axis="y", zorder=0)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    # Timestep distribution used
    unique_ts, counts = np.unique(timesteps_used, return_counts=True)
    fracs = counts / counts.sum() * 100
    bar_colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(unique_ts)))
    axes[1].bar([str(int(t)) for t in unique_ts], fracs,
                color=bar_colors2, edgecolor="white", alpha=0.85, zorder=3)
    axes[1].set_xlabel("Timestep Used")
    axes[1].set_ylabel("% of Samples")
    axes[1].set_title("Timestep Distribution at Inference")
    axes[1].grid(True, alpha=0.3, axis="y", zorder=0)
    for i, (t, f) in enumerate(zip(unique_ts, fracs)):
        axes[1].text(i, f + 0.5, f"{f:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Complexity Predictor: Timestep Assignment Quality", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "eval_predictor_t_error")


def plot_accuracy_vs_energy_tradeoff(
    exit_metrics: dict,
    max_timestep: int,
    plots_dir: Path,
):
    """Plot accuracy vs energy savings tradeoff at each exit point."""
    if not exit_metrics:
        return
    timesteps = sorted(exit_metrics.keys())
    accuracies = [exit_metrics[t]["accuracy"] for t in timesteps]
    energy_savings_vals = [1.0 - (t / max_timestep) for t in timesteps]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        [e * 100 for e in energy_savings_vals], accuracies,
        c=timesteps, cmap="coolwarm", s=200, edgecolors="black", zorder=5
    )

    # Connect points with a line
    ax.plot([e * 100 for e in energy_savings_vals], accuracies, "k--", alpha=0.4)

    # Label each point
    for t, e, a in zip(timesteps, energy_savings_vals, accuracies):
        ax.annotate(f"T={t}", (e * 100, a), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Energy Savings (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Energy Savings Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Timestep")
    _save_fig(fig, plots_dir, "eval_accuracy_energy_tradeoff")


# ═══════════════════════════════════════════════════════════════════════════
# MASTER PLOT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_plots(results_manager, config: dict):
    """Generate ALL plots from the results manager data.

    Call this after all phases are complete. It reads metrics from the
    ResultsManager and generates every possible plot.

    Args:
        results_manager: The ResultsManager instance with all recorded metrics.
        config: Experiment configuration.
    """
    plots_dir = results_manager.get_plots_dir()
    print(f"\nGenerating plots in: {plots_dir}")

    # ── Phase 1 plots ─────────────────────────────────────────────────────
    phase1_metrics = results_manager.get_phase1_metrics()
    if phase1_metrics:
        print("  Generating Phase 1 plots...")
        plot_phase1_training_curves(phase1_metrics, plots_dir)
        plot_phase1_loss_separate(phase1_metrics, plots_dir)
        plot_phase1_accuracy_separate(phase1_metrics, plots_dir)

    # ── Phase 2 plots ─────────────────────────────────────────────────────
    phase2_data = results_manager.get_phase2_metrics()
    if phase2_data and "timesteps" in phase2_data:
        print("  Generating Phase 2 plots...")
        timesteps = np.array(phase2_data["timesteps"])
        mean_acc = np.array(phase2_data["mean_accuracy_per_t"])
        std_acc = np.array(phase2_data["std_accuracy_per_t"])
        plot_phase2_accuracy_vs_timestep(timesteps, mean_acc, std_acc, plots_dir)

        # R² fit quality (available after Phase 2 with new code)
        if "_r2_scores" in phase2_data:
            plot_r2_histogram(phase2_data["_r2_scores"], plots_dir)

    # ── Phase 3 plots ─────────────────────────────────────────────────────
    phase3_metrics = results_manager.get_phase3_metrics()
    if phase3_metrics:
        print("  Generating Phase 3 plots...")
        plot_phase3_training_curves(phase3_metrics, plots_dir)
        plot_phase3_predictor_loss(phase3_metrics, plots_dir)

    # ── Evaluation plots ──────────────────────────────────────────────────
    eval_data = results_manager.get_eval_metrics()
    if eval_data:
        print("  Generating evaluation plots...")

        adaptive = eval_data.get("adaptive", {})
        baseline = eval_data.get("baseline", {})
        ann_acc = eval_data.get("ann_acc", 0.0)
        ann_top5 = eval_data.get("ann_top5_acc", 0.0)

        if "exit_distribution" in eval_data:
            plot_exit_distribution_bar(eval_data["exit_distribution"], plots_dir)
            plot_exit_distribution_pie(eval_data["exit_distribution"], plots_dir)
            max_t = config["snn"]["max_timestep"]
            plot_accuracy_vs_energy_tradeoff(eval_data["exit_distribution"], max_t, plots_dir)

        if adaptive and baseline:
            plot_energy_savings_comparison(adaptive, baseline, plots_dir)

            # ANN → SNN accuracy comparison (the paper table figure)
            plot_ann_snn_accuracy_comparison(
                ann_acc=ann_acc,
                snn_fixed_acc=baseline.get("accuracy", 0.0),
                snn_adaptive_acc=adaptive.get("accuracy", 0.0),
                plots_dir=plots_dir,
                ann_top5=ann_top5,
                snn_fixed_top5=baseline.get("top5_accuracy", 0.0),
                snn_adaptive_top5=adaptive.get("top5_accuracy", 0.0),
            )

        # SOPs comparison
        if "sops" in eval_data:
            plot_sops_comparison(eval_data["sops"], plots_dir)

        # ECE reliability diagram
        if "ece" in eval_data:
            plot_ece_reliability_diagram(eval_data["ece"], plots_dir)

        # Predictor T-error
        if "predictor_t_error" in eval_data and eval_data["predictor_t_error"]:
            # Need raw timesteps_used — try loading from per-sample CSV if not in memory
            t_err = eval_data["predictor_t_error"]
            timesteps_used_arr = _load_timesteps_from_rm(results_manager)
            if timesteps_used_arr is not None:
                plot_predictor_t_error(t_err, timesteps_used_arr, plots_dir)

        if "confusion_matrix" in eval_data:
            plot_confusion_matrix(eval_data["confusion_matrix"], plots_dir)

    # Count generated plots
    png_files = list(plots_dir.glob("*.png"))
    pdf_files = list(plots_dir.glob("*.pdf"))
    print(f"  Generated {len(png_files)} PNG + {len(pdf_files)} PDF plots.")
    print(f"  Plots saved to: {plots_dir}")


def _load_timesteps_from_rm(results_manager) -> Optional[np.ndarray]:
    """Load timesteps_used array from adaptive_results.csv if available."""
    import csv as _csv
    csv_path = results_manager.eval_dir / "adaptive_results.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            return np.array([int(row["timestep_used"]) for row in reader])
    except Exception:
        return None


def generate_phase2_plots_from_file(
    profiling_path: str,
    plots_dir: Path,
    config: dict,
):
    """Generate Phase 2 plots from saved profiling results file.

    Useful for regenerating plots without re-running profiling.
    """
    data = np.load(profiling_path)
    timesteps = data["timesteps"]
    performances = data["performances"]
    alphas = data["alphas"]
    betas = data["betas"]
    gammas = data["gammas"]

    mean_acc = performances.mean(axis=0)
    std_acc = performances.std(axis=0)

    print("  Phase 2 plots from saved profiling data...")
    plot_phase2_accuracy_vs_timestep(timesteps, mean_acc, std_acc, plots_dir)
    plot_phase2_difficulty_histogram(
        _compute_t_optimals(alphas, betas, gammas, config),
        config["snn"]["exit_points"],
        plots_dir,
    )
    plot_phase2_scaling_law_samples(
        timesteps, performances, alphas, betas, gammas, plots_dir
    )
    plot_phase2_parameter_distributions(alphas, betas, gammas, plots_dir)

    # R² histogram (only in new-format .npz files)
    if "r2_scores" in data:
        plot_r2_histogram(data["r2_scores"], plots_dir)


def generate_eval_plots_from_csv(eval_dir: Path, plots_dir: Path, config: dict):
    """Regenerate evaluation plots from saved CSVs.

    Useful for re-plotting with different styles without re-running eval.
    """
    import csv as csv_mod

    # Load adaptive results
    results_path = eval_dir / "adaptive_results.csv"
    if results_path.exists():
        with open(results_path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)

        predictions = np.array([int(r["prediction"]) for r in rows])
        targets = np.array([int(r["target"]) for r in rows])
        correct = np.array([int(r["correct"]) for r in rows])
        confidences = np.array([float(r["confidence"]) for r in rows])
        timesteps_used = np.array([int(r["timestep_used"]) for r in rows])

        plot_confidence_histogram(confidences, correct, plots_dir)
        plot_timestep_vs_confidence(timesteps_used, confidences, correct, plots_dir)


def _compute_t_optimals(alphas, betas, gammas, config):
    """Compute T_optimal array from parameters."""
    target = config["scaling_law"]["target_accuracy"]
    max_t = config["snn"]["max_timestep"]
    t_optimals = []
    for a, b, g in zip(alphas, betas, gammas):
        if a < 1e-6 or (target - g) <= 0:
            t_optimals.append(max_t if a < 1e-6 else 1)
        else:
            t = ((target - g) / a) ** (1.0 / max(b, 0.01))
            t_optimals.append(int(np.clip(np.ceil(t), 1, max_t)))
    return np.array(t_optimals)
