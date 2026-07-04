"""Results Manager — centralized tracking, CSV export, and organized output.

Handles:
- Per-epoch metric logging to CSV
- Summary statistics generation
- Organized directory structure for all outputs
- JSON metadata for experiment reproducibility

Output structure:
    results/<dataset>_<timestamp>/
    ├── phase1/
    │   ├── training_log.csv         # epoch, train_loss, train_acc, test_loss, test_acc, lr, top5_acc
    │   └── summary.json             # best_acc, top5_acc, total_epochs, model_params, etc.
    ├── phase2/
    │   ├── profiling_stats.csv      # per-timestep accuracy statistics
    │   ├── sample_difficulty.csv    # alpha, beta, gamma, t_optimal, r2, mse per sample
    │   ├── fit_quality.csv          # r2/mse summary statistics
    │   └── summary.json
    ├── phase3/
    │   ├── training_log.csv         # epoch, pred_loss, exit_loss, exit_accs, avg_t
    │   └── summary.json
    ├── evaluation/
    │   ├── adaptive_results.csv     # per-sample: prediction, confidence, timestep_used, correct
    │   ├── exit_distribution.csv    # per-exit: timestep, count, fraction, accuracy
    │   ├── comparison.csv           # adaptive vs baseline + ANN baseline metrics
    │   ├── sops.csv                 # synaptic operations breakdown
    │   ├── ece.csv                  # per-bin calibration data for reliability diagram
    │   ├── predictor_t_error.csv    # predictor timestep error statistics
    │   └── summary.json             # all key numbers in one place
    ├── plots/                       # All generated plots (PNG + PDF)
    └── experiment_config.json       # Full config dump for reproducibility
"""

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


class ResultsManager:
    """Centralized experiment results tracking and export.

    Usage:
        rm = ResultsManager(config, results_dir="./results")
        rm.log_phase1_epoch(epoch, train_loss, train_acc, test_loss, test_acc, lr)
        ...
        rm.save_phase1_summary(best_acc=95.2, total_epochs=100)
        rm.finalize()
    """

    def __init__(self, config: dict, results_dir: str = "./results"):
        """
        Args:
            config: Full experiment configuration dict.
            results_dir: Base directory for all results.
        """
        self.config = config
        dataset_name = config["dataset"]["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{dataset_name}_{timestamp}"

        # Create directory structure
        self.base_dir = Path(results_dir) / self.experiment_name
        self.phase1_dir = self.base_dir / "phase1"
        self.phase2_dir = self.base_dir / "phase2"
        self.phase3_dir = self.base_dir / "phase3"
        self.eval_dir = self.base_dir / "evaluation"
        self.plots_dir = self.base_dir / "plots"

        for d in [self.phase1_dir, self.phase2_dir, self.phase3_dir, self.eval_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save experiment config
        self._save_config()

        # CSV writers (initialized lazily)
        self._phase1_csv: Optional[csv.writer] = None
        self._phase1_file = None
        self._phase3_csv: Optional[csv.writer] = None
        self._phase3_file = None

        # In-memory metric storage for plotting
        self.phase1_metrics: list[dict] = []
        self.phase2_metrics: dict[str, Any] = {}
        self.phase3_metrics: list[dict] = []
        self.eval_metrics: dict[str, Any] = {}

        # ANN accuracy stored at Phase 1 for ANN→SNN gap reporting
        self._ann_best_acc: float = 0.0
        self._ann_best_top5_acc: float = 0.0

    def _save_config(self):
        """Save full config as JSON for reproducibility."""
        config_path = self.base_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    # ──────────────────────────── Phase 1: ANN Training ────────────────────────────

    def log_phase1_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
        lr: float,
        test_top5_acc: float = 0.0,
    ):
        """Log one epoch of Phase 1 training."""
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 4),
            "test_top5_acc": round(test_top5_acc, 4),
            "lr": round(lr, 8),
        }
        self.phase1_metrics.append(row)

        # Write to CSV
        if self._phase1_csv is None:
            self._phase1_file = open(self.phase1_dir / "training_log.csv", "w", newline="")
            self._phase1_csv = csv.DictWriter(self._phase1_file, fieldnames=list(row.keys()))
            self._phase1_csv.writeheader()
        self._phase1_csv.writerow(row)
        self._phase1_file.flush()

    def save_phase1_summary(
        self,
        best_acc: float,
        best_epoch: int,
        total_epochs: int,
        model_name: str,
        total_params: int,
        training_time_seconds: float = 0.0,
        best_top5_acc: float = 0.0,
    ):
        """Save Phase 1 summary."""
        summary = {
            "phase": "Phase 1 - ANN Pretraining",
            "model": model_name,
            "dataset": self.config["dataset"]["name"],
            "best_test_accuracy": round(best_acc, 4),
            "best_test_top5_accuracy": round(best_top5_acc, 4),
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "training_time_seconds": round(training_time_seconds, 1),
            "batch_size": self.config["model"]["batch_size"],
            "learning_rate": self.config["model"]["lr"],
            "lr_schedule": self.config["model"].get("lr_schedule", "cosine"),
            "optimizer": "SGD",
            "weight_decay": self.config["model"].get("weight_decay", 1e-4),
        }
        with open(self.phase1_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Store ANN accuracy so evaluation phase can compute the ANN→SNN gap
        self._ann_best_acc = best_acc
        self._ann_best_top5_acc = best_top5_acc

        if self._phase1_file:
            self._phase1_file.close()
            self._phase1_file = None
            self._phase1_csv = None

    # ──────────────────────────── Phase 2: SNN Profiling ────────────────────────────

    def save_phase2_profiling_stats(
        self,
        timesteps: np.ndarray,
        mean_acc_per_t: np.ndarray,
        std_acc_per_t: np.ndarray,
    ):
        """Save per-timestep profiling statistics."""
        path = self.phase2_dir / "profiling_stats.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "mean_accuracy", "std_accuracy", "samples_correct_pct"])
            for t, mean_a, std_a in zip(timesteps, mean_acc_per_t, std_acc_per_t):
                writer.writerow([int(t), round(float(mean_a), 6), round(float(std_a), 6), round(float(mean_a) * 100, 2)])

        self.phase2_metrics["timesteps"] = timesteps.tolist()
        self.phase2_metrics["mean_accuracy_per_t"] = mean_acc_per_t.tolist()
        self.phase2_metrics["std_accuracy_per_t"] = std_acc_per_t.tolist()

    def save_phase2_sample_difficulty(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        gammas: np.ndarray,
        t_optimals: np.ndarray,
        labels: np.ndarray,
        r2_scores: Optional[np.ndarray] = None,
        fit_mse: Optional[np.ndarray] = None,
    ):
        """Save per-sample difficulty parameters to CSV."""
        has_fit = r2_scores is not None and fit_mse is not None
        path = self.phase2_dir / "sample_difficulty.csv"
        header = ["sample_idx", "label", "alpha", "beta", "gamma", "t_optimal"]
        if has_fit:
            header += ["r2", "fit_mse"]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(alphas)):
                row = [
                    i, int(labels[i]),
                    round(float(alphas[i]), 6),
                    round(float(betas[i]), 6),
                    round(float(gammas[i]), 6),
                    int(t_optimals[i]),
                ]
                if has_fit:
                    row += [round(float(r2_scores[i]), 6), round(float(fit_mse[i]), 8)]
                writer.writerow(row)

        self.phase2_metrics["num_samples"] = len(alphas)
        self.phase2_metrics["alpha_mean"] = float(alphas.mean())
        self.phase2_metrics["alpha_std"] = float(alphas.std())
        self.phase2_metrics["beta_mean"] = float(betas.mean())
        self.phase2_metrics["beta_std"] = float(betas.std())
        self.phase2_metrics["gamma_mean"] = float(gammas.mean())
        self.phase2_metrics["gamma_std"] = float(gammas.std())
        self.phase2_metrics["t_optimal_mean"] = float(t_optimals.mean())
        self.phase2_metrics["t_optimal_median"] = float(np.median(t_optimals))

        if has_fit:
            self.phase2_metrics["r2_mean"] = float(r2_scores.mean())
            self.phase2_metrics["r2_median"] = float(np.median(r2_scores))
            self.phase2_metrics["r2_std"] = float(r2_scores.std())
            self.phase2_metrics["r2_below_0.8_pct"] = float((r2_scores < 0.8).mean() * 100)
            self.phase2_metrics["fit_mse_mean"] = float(fit_mse.mean())
            # Store arrays for plotting
            self.phase2_metrics["_r2_scores"] = r2_scores
            self.phase2_metrics["_fit_mse"] = fit_mse

    def save_phase2_fit_quality(
        self,
        r2_scores: np.ndarray,
        fit_mse: np.ndarray,
        ann_baseline_acc: float,
        snn_max_t_acc: float,
    ):
        """Save curve-fit quality summary and ANN→SNN accuracy gap."""
        path = self.phase2_dir / "fit_quality.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["r2_mean", round(float(r2_scores.mean()), 6)])
            writer.writerow(["r2_median", round(float(np.median(r2_scores)), 6)])
            writer.writerow(["r2_std", round(float(r2_scores.std()), 6)])
            writer.writerow(["r2_min", round(float(r2_scores.min()), 6)])
            writer.writerow(["r2_pct_above_0.9", round(float((r2_scores >= 0.9).mean() * 100), 2)])
            writer.writerow(["r2_pct_above_0.8", round(float((r2_scores >= 0.8).mean() * 100), 2)])
            writer.writerow(["fit_mse_mean", round(float(fit_mse.mean()), 8)])
            writer.writerow(["fit_mse_median", round(float(np.median(fit_mse)), 8)])
            writer.writerow(["ann_baseline_acc", round(float(ann_baseline_acc), 4)])
            writer.writerow(["snn_max_t_acc", round(float(snn_max_t_acc), 4)])
            writer.writerow(["ann_snn_acc_gap", round(float(ann_baseline_acc - snn_max_t_acc), 4)])

        self.phase2_metrics["ann_baseline_acc"] = float(ann_baseline_acc)
        self.phase2_metrics["snn_max_t_acc"] = float(snn_max_t_acc)
        self.phase2_metrics["ann_snn_acc_gap"] = float(ann_baseline_acc - snn_max_t_acc)

    def save_phase2_summary(
        self,
        num_samples: int,
        num_thresholds: int,
        conversion_time_seconds: float = 0.0,
        profiling_time_seconds: float = 0.0,
    ):
        """Save Phase 2 summary."""
        summary = {
            "phase": "Phase 2 - SNN Profiling & Curve Fitting",
            "num_samples_profiled": num_samples,
            "num_layer_thresholds": num_thresholds,
            "timesteps_evaluated": self.config["snn"]["timesteps"],
            "conversion_method": self.config["conversion"]["method"],
            "percentile": self.config["conversion"]["percentile"],
            "conversion_time_seconds": round(conversion_time_seconds, 1),
            "profiling_time_seconds": round(profiling_time_seconds, 1),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in self.phase2_metrics.items()},
        }
        with open(self.phase2_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ──────────────────────────── Phase 3: Joint Training ────────────────────────────

    def log_phase3_epoch(
        self,
        epoch: int,
        pred_loss: float,
        param_loss: float,
        energy_loss: float,
        exit_loss: float,
        avg_t_predicted: float,
        exit_accuracies: dict[int, float],
    ):
        """Log one epoch of Phase 3 training."""
        row = {
            "epoch": epoch,
            "pred_loss": round(pred_loss, 6),
            "param_loss": round(param_loss, 6),
            "energy_loss": round(energy_loss, 6),
            "exit_loss": round(exit_loss, 6),
            "avg_t_predicted": round(avg_t_predicted, 2),
        }
        for t, acc in sorted(exit_accuracies.items()):
            row[f"exit_acc_T{t}"] = round(acc, 4)

        self.phase3_metrics.append(row)

        # Write to CSV
        if self._phase3_csv is None:
            self._phase3_file = open(self.phase3_dir / "training_log.csv", "w", newline="")
            self._phase3_csv = csv.DictWriter(self._phase3_file, fieldnames=list(row.keys()))
            self._phase3_csv.writeheader()
        self._phase3_csv.writerow(row)
        self._phase3_file.flush()

    def save_phase3_summary(
        self,
        best_exit_accuracies: dict[int, float],
        best_avg_t: float,
        total_epochs: int,
        predictor_params: int,
        exit_branch_params: int,
        training_time_seconds: float = 0.0,
    ):
        """Save Phase 3 summary."""
        summary = {
            "phase": "Phase 3 - Joint Training (Predictor + Exit Branches)",
            "total_epochs": total_epochs,
            "best_avg_t_predicted": round(best_avg_t, 2),
            "best_exit_accuracies": {str(k): round(v, 4) for k, v in sorted(best_exit_accuracies.items())},
            "predictor_parameters": predictor_params,
            "exit_branch_parameters": exit_branch_params,
            "predictor_lr": self.config["predictor"]["lr"],
            "exit_timesteps": self.config["snn"]["exit_points"],
            "target_accuracy": self.config["scaling_law"]["target_accuracy"],
            "training_time_seconds": round(training_time_seconds, 1),
        }
        with open(self.phase3_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self._phase3_file:
            self._phase3_file.close()
            self._phase3_file = None
            self._phase3_csv = None

    # ──────────────────────────── Evaluation ────────────────────────────

    def save_eval_per_sample(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidences: np.ndarray,
        timesteps_used: np.ndarray,
    ):
        """Save per-sample evaluation results."""
        correct = (predictions == targets).astype(int)
        path = self.eval_dir / "adaptive_results.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "target", "prediction", "correct", "confidence", "timestep_used"])
            for i in range(len(predictions)):
                writer.writerow([
                    i, int(targets[i]), int(predictions[i]), int(correct[i]),
                    round(float(confidences[i]), 4), int(timesteps_used[i]),
                ])

        self.eval_metrics["per_sample_saved"] = True
        self.eval_metrics["total_samples"] = len(predictions)

    def save_eval_exit_distribution(self, exit_metrics: dict[int, dict]):
        """Save exit distribution CSV."""
        path = self.eval_dir / "exit_distribution.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "count", "fraction", "accuracy"])
            for t, info in sorted(exit_metrics.items()):
                writer.writerow([
                    int(t), int(info["count"]),
                    round(float(info["fraction"]), 4),
                    round(float(info["accuracy"]), 4),
                ])
        self.eval_metrics["exit_distribution"] = exit_metrics

    def save_eval_comparison(
        self,
        adaptive_results: dict,
        baseline_results: dict,
        ann_acc: float = 0.0,
        ann_top5_acc: float = 0.0,
    ):
        """Save adaptive vs baseline comparison CSV, including ANN baseline."""
        path = self.eval_dir / "comparison.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "ann_baseline", "snn_fixed_t", "snn_adaptive", "improvement_vs_fixed"])
            rows = [
                ("top1_accuracy (%)",
                 ann_acc,
                 baseline_results["accuracy"],
                 adaptive_results["accuracy"],
                 adaptive_results["accuracy"] - baseline_results["accuracy"]),
                ("top5_accuracy (%)",
                 ann_top5_acc,
                 baseline_results.get("top5_accuracy", 0.0),
                 adaptive_results.get("top5_accuracy", 0.0),
                 adaptive_results.get("top5_accuracy", 0.0) - baseline_results.get("top5_accuracy", 0.0)),
                ("avg_timestep",
                 0.0,
                 baseline_results["avg_timestep"],
                 adaptive_results["avg_timestep"],
                 baseline_results["avg_timestep"] - adaptive_results["avg_timestep"]),
                ("energy_savings (%)",
                 0.0, 0.0,
                 adaptive_results["energy_savings"] * 100,
                 adaptive_results["energy_savings"] * 100),
                ("speedup (x)",
                 0.0, 1.0,
                 adaptive_results["speedup"],
                 adaptive_results["speedup"] - 1.0),
                ("ann_snn_acc_gap (%)",
                 ann_acc,
                 ann_acc - baseline_results["accuracy"],
                 ann_acc - adaptive_results["accuracy"],
                 0.0),
            ]
            for row in rows:
                writer.writerow([row[0]] + [round(v, 4) for v in row[1:]])

        self.eval_metrics["adaptive"] = adaptive_results
        self.eval_metrics["baseline"] = baseline_results
        self.eval_metrics["ann_acc"] = ann_acc
        self.eval_metrics["ann_top5_acc"] = ann_top5_acc

    def save_eval_sops(self, sops_dict: dict):
        """Save Synaptic Operations (SOPs) breakdown CSV."""
        path = self.eval_dir / "sops.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in sops_dict.items():
                writer.writerow([k, round(float(v), 4)])
        self.eval_metrics["sops"] = sops_dict

    def save_eval_ece(self, ece_dict: dict):
        """Save per-bin ECE data for reliability diagram + summary."""
        # Summary row
        path = self.eval_dir / "ece.csv"
        bin_accs = ece_dict["bin_accs"]
        bin_confs = ece_dict["bin_confs"]
        bin_counts = ece_dict["bin_counts"]
        bin_edges = ece_dict["bin_edges"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_lower", "bin_upper", "bin_conf", "bin_acc", "bin_count", "gap"])
            for b in range(len(bin_accs)):
                gap = abs(bin_accs[b] - bin_confs[b])
                writer.writerow([
                    round(float(bin_edges[b]), 4),
                    round(float(bin_edges[b + 1]), 4),
                    round(float(bin_confs[b]), 6),
                    round(float(bin_accs[b]), 6),
                    int(bin_counts[b]),
                    round(float(gap), 6),
                ])
        self.eval_metrics["ece"] = ece_dict

    def save_eval_predictor_t_error(self, t_error_dict: dict):
        """Save predictor timestep error statistics CSV."""
        path = self.eval_dir / "predictor_t_error.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            scalar_keys = ["mae", "rmse", "exact_match_rate", "over_allocation_rate",
                           "under_allocation_rate", "mean_error"]
            for k in scalar_keys:
                if k in t_error_dict:
                    writer.writerow([k, round(float(t_error_dict[k]), 6)])
        self.eval_metrics["predictor_t_error"] = t_error_dict

    def save_eval_summary(self, adaptive_results: dict, baseline_results: dict):
        """Save evaluation summary JSON."""
        ann_acc = self.eval_metrics.get("ann_acc", 0.0)
        ann_top5 = self.eval_metrics.get("ann_top5_acc", 0.0)
        sops = self.eval_metrics.get("sops", {})
        ece_data = self.eval_metrics.get("ece", {})
        t_err = self.eval_metrics.get("predictor_t_error", {})

        summary = {
            "phase": "Evaluation - Adaptive Inference",
            "dataset": self.config["dataset"]["name"],
            # ANN baseline (Phase 1 reference)
            "ann_baseline": {
                "top1_accuracy": round(ann_acc, 4),
                "top5_accuracy": round(ann_top5, 4),
            },
            # SNN fixed-T (T_max for all samples)
            "snn_fixed_t_baseline": {
                "top1_accuracy": round(baseline_results["accuracy"], 4),
                "top5_accuracy": round(baseline_results.get("top5_accuracy", 0.0), 4),
                "avg_timestep": round(baseline_results["avg_timestep"], 2),
            },
            # NeuroScale++ adaptive
            "adaptive": {
                "top1_accuracy": round(adaptive_results["accuracy"], 4),
                "top5_accuracy": round(adaptive_results.get("top5_accuracy", 0.0), 4),
                "avg_timestep": round(adaptive_results["avg_timestep"], 2),
                "energy_savings": round(adaptive_results["energy_savings"], 4),
                "speedup": round(adaptive_results["speedup"], 2),
                "avg_confidence": round(adaptive_results.get("avg_confidence", 0), 4),
                "timestep_std": round(adaptive_results.get("timestep_std", 0), 2),
            },
            # Key comparison numbers (the paper table)
            "comparison": {
                "ann_to_snn_acc_drop": round(ann_acc - baseline_results["accuracy"], 4),
                "snn_adaptive_vs_fixed_acc_drop": round(
                    baseline_results["accuracy"] - adaptive_results["accuracy"], 4),
                "ann_to_adaptive_acc_drop": round(ann_acc - adaptive_results["accuracy"], 4),
                "energy_savings_pct": round(adaptive_results["energy_savings"] * 100, 2),
                "speedup_factor": round(adaptive_results["speedup"], 2),
                "timestep_reduction": round(
                    baseline_results["avg_timestep"] - adaptive_results["avg_timestep"], 2),
            },
            # Energy (SOPs)
            "sops": {
                "adaptive_per_sample": round(sops.get("adaptive_sops_per_sample", 0), 2),
                "baseline_per_sample": round(sops.get("baseline_sops_per_sample", 0), 2),
                "sops_reduction_pct": round(sops.get("sops_reduction", 0) * 100, 2),
            },
            # Calibration
            "calibration": {
                "ece": round(ece_data.get("ece", 0), 6),
                "mce": round(ece_data.get("mce", 0), 6),
            },
            # Predictor quality
            "predictor_t_error": {
                "mae": round(t_err.get("mae", 0), 4),
                "rmse": round(t_err.get("rmse", 0), 4),
                "exact_match_rate": round(t_err.get("exact_match_rate", 0), 4),
                "over_allocation_rate": round(t_err.get("over_allocation_rate", 0), 4),
                "under_allocation_rate": round(t_err.get("under_allocation_rate", 0), 4),
            },
            "exit_points": self.config["snn"]["exit_points"],
            "max_timestep": self.config["snn"]["max_timestep"],
            "target_accuracy": self.config["scaling_law"]["target_accuracy"],
        }
        with open(self.eval_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ──────────────────────────── Per-class metrics ────────────────────────────

    def save_per_class_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timesteps_used: np.ndarray,
        num_classes: int,
        class_names: Optional[list[str]] = None,
    ):
        """Save per-class accuracy and average timestep."""
        path = self.eval_dir / "per_class_metrics.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "accuracy", "avg_timestep", "num_samples"])
            for c in range(num_classes):
                mask = targets == c
                if mask.sum() == 0:
                    continue
                class_acc = (predictions[mask] == targets[mask]).mean() * 100
                class_avg_t = timesteps_used[mask].mean()
                class_name = class_names[c] if class_names else f"class_{c}"
                writer.writerow([
                    c, class_name,
                    round(float(class_acc), 2),
                    round(float(class_avg_t), 2),
                    int(mask.sum()),
                ])

    # ──────────────────────────── Confusion matrix data ────────────────────────────

    def save_confusion_matrix(self, predictions: np.ndarray, targets: np.ndarray, num_classes: int):
        """Save raw confusion matrix as CSV."""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(targets, predictions):
            cm[t][p] += 1

        path = self.eval_dir / "confusion_matrix.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["true\\pred"] + [str(i) for i in range(num_classes)]
            writer.writerow(header)
            for i in range(num_classes):
                writer.writerow([str(i)] + [str(cm[i][j]) for j in range(num_classes)])

        self.eval_metrics["confusion_matrix"] = cm

    # ──────────────────────────── Utilities ────────────────────────────

    def get_plots_dir(self) -> Path:
        """Return the plots directory path."""
        return self.plots_dir

    def get_phase1_metrics(self) -> list[dict]:
        """Return phase 1 metrics for plotting."""
        return self.phase1_metrics

    def get_phase3_metrics(self) -> list[dict]:
        """Return phase 3 metrics for plotting."""
        return self.phase3_metrics

    def get_phase2_metrics(self) -> dict:
        """Return phase 2 metrics for plotting."""
        return self.phase2_metrics

    def get_eval_metrics(self) -> dict:
        """Return evaluation metrics for plotting."""
        return self.eval_metrics

    def finalize(self):
        """Close any open file handles and write final summary."""
        if self._phase1_file:
            self._phase1_file.close()
        if self._phase3_file:
            self._phase3_file.close()

        # Write a top-level experiment summary
        final_summary = {
            "experiment_name": self.experiment_name,
            "dataset": self.config["dataset"]["name"],
            "model": self.config["model"]["ann"],
            "timestamp": datetime.now().isoformat(),
            "results_directory": str(self.base_dir),
            "phases_completed": [],
        }
        if self.phase1_metrics:
            final_summary["phases_completed"].append("phase1")
        if self.phase2_metrics:
            final_summary["phases_completed"].append("phase2")
        if self.phase3_metrics:
            final_summary["phases_completed"].append("phase3")
        if self.eval_metrics:
            final_summary["phases_completed"].append("evaluation")

        with open(self.base_dir / "experiment_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)

        print(f"\nResults saved to: {self.base_dir}")
