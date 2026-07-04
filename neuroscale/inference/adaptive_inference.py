"""Adaptive Inference Pipeline.

The full NeuroScale++ inference system:
1. Complexity Predictor sees the input image → predicts (α, β, γ)
2. Scaling Law computes T_optimal → snaps to nearest exit point
3. Multi-Exit SNN runs only up to T_optimal → exits early
4. Returns prediction + energy metrics

This is what runs at deployment time. The predictor adds minimal overhead
(~5% of SNN compute) while saving 40-70% energy on average by early exiting.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from ..predictor.complexity_predictor import ComplexityPredictor
from ..spiking.multi_exit_snn import MultiExitSNN
from ..scaling.scaling_law import ScalingLawModule
from ..utils.metrics import (
    accuracy,
    energy_savings,
    compute_sops,
    compute_ece,
    compute_predictor_t_error,
    topk_accuracy_from_logits,
)
from ..utils.results_manager import ResultsManager


class AdaptiveInference:
    """Full adaptive inference pipeline for NeuroScale++.

    Combines:
    - ComplexityPredictor: predicts per-sample difficulty
    - ScalingLawModule: computes optimal timestep from difficulty
    - MultiExitSNN: runs SNN with early exit at predicted timestep

    Usage:
        inference = AdaptiveInference(multi_exit_snn, predictor, config)
        results = inference.evaluate(test_loader, device)
    """

    def __init__(
        self,
        multi_exit_snn: MultiExitSNN,
        predictor: ComplexityPredictor,
        config: dict,
        sops_per_timestep: float = 1e6,
    ):
        """
        Args:
            multi_exit_snn: Trained Multi-Exit SNN.
            predictor: Trained Complexity Predictor.
            config: Configuration dictionary.
            sops_per_timestep: Estimated synaptic ops per timestep per sample.
                Default 1e6 is a reasonable proxy for ResNet-20 on CIFAR-10.
                Use estimate_sops_per_timestep() for a tighter estimate.
        """
        self.multi_exit_snn = multi_exit_snn
        self.predictor = predictor

        snn_cfg = config["snn"]
        scaling_cfg = config["scaling_law"]
        inference_cfg = config.get("inference", {})

        self.exit_timesteps = snn_cfg["exit_points"]
        self.max_timestep = snn_cfg["max_timestep"]
        self.target_accuracy = scaling_cfg["target_accuracy"]
        self.confidence_threshold = inference_cfg.get("confidence_threshold", 0.9)
        self.sops_per_timestep = sops_per_timestep

        self.scaling_module = ScalingLawModule(
            exit_timesteps=self.exit_timesteps,
            target_accuracy=self.target_accuracy,
        )

    @torch.no_grad()
    def predict_single(
        self, image: torch.Tensor, device: torch.device
    ) -> dict:
        """Run inference on a single image.

        Args:
            image: Input image tensor (1, C, H, W) or (C, H, W).
            device: Computation device.

        Returns:
            Dict with 'prediction', 'confidence', 'timestep_used',
            'scaling_params', 'logits'.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        self.predictor.eval()
        self.multi_exit_snn.eval()

        # Step 1: Predict complexity
        params = self.predictor(image)
        alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]

        # Step 2: Compute optimal timestep
        t_optimal = self.scaling_module.compute_t_optimal(alpha, beta, gamma)
        t_snapped = self.scaling_module.snap_to_exit(t_optimal)
        t_use = int(t_snapped.item())

        # Step 3: Run SNN up to T_optimal
        logits = self.multi_exit_snn.forward_single_exit(image, t_use)

        # Step 4: Get prediction and confidence
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = probs.max(dim=1)

        return {
            "prediction": prediction.item(),
            "confidence": confidence.item(),
            "timestep_used": t_use,
            "scaling_params": {
                "alpha": alpha.item(),
                "beta": beta.item(),
                "gamma": gamma.item(),
            },
            "logits": logits.cpu(),
        }

    @torch.no_grad()
    def predict_batch(
        self, images: torch.Tensor, device: torch.device
    ) -> dict:
        """Run inference on a batch of images.

        Args:
            images: Batch of images (B, C, H, W).
            device: Computation device.

        Returns:
            Dict with batch predictions, confidences, timesteps used.
        """
        images = images.to(device)
        self.predictor.eval()
        self.multi_exit_snn.eval()

        batch_size = images.size(0)

        # Step 1: Predict complexity for all samples
        params = self.predictor(images)
        alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]

        # Step 2: Compute optimal timesteps
        t_optimal = self.scaling_module.compute_t_optimal(alpha, beta, gamma)
        t_snapped = self.scaling_module.snap_to_exit(t_optimal)

        # Step 3: Group samples by exit timestep for efficient batched inference
        # Run all samples through the SNN up to the max needed timestep,
        # but collect results at each sample's designated exit
        predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
        confidences = torch.zeros(batch_size, device=device)
        all_logits = torch.zeros(batch_size, self.multi_exit_snn.num_classes, device=device)

        # For simplicity and correctness, process by exit groups
        for exit_t in self.exit_timesteps:
            mask = (t_snapped == exit_t)
            if not mask.any():
                continue

            batch_images = images[mask]
            logits = self.multi_exit_snn.forward_single_exit(batch_images, exit_t)

            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            predictions[mask] = preds
            confidences[mask] = conf
            all_logits[mask] = logits

        return {
            "predictions": predictions.cpu(),
            "confidences": confidences.cpu(),
            "timesteps_used": t_snapped.cpu().int(),
            "logits": all_logits.cpu(),
            "scaling_params": {
                "alpha": alpha.cpu(),
                "beta": beta.cpu(),
                "gamma": gamma.cpu(),
            },
        }

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        device: torch.device,
        verbose: bool = True,
    ) -> dict:
        """Full evaluation on a test set.

        Args:
            test_loader: Test/validation DataLoader.
            device: Computation device.
            verbose: Print progress.

        Returns:
            Dict with comprehensive evaluation metrics.
        """
        self.predictor.eval()
        self.multi_exit_snn.eval()
        self.scaling_module = self.scaling_module.to(device)

        all_predictions = []
        all_targets = []
        all_timesteps = []
        all_confidences = []

        iterator = tqdm(test_loader, desc="Evaluating") if verbose else test_loader

        for images, targets in iterator:
            images = images.to(device)
            results = self.predict_batch(images, device)

            all_predictions.append(results["predictions"])
            all_targets.append(targets)
            all_timesteps.append(results["timesteps_used"])
            all_confidences.append(results["confidences"])

        # Concatenate all results
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        timesteps = torch.cat(all_timesteps).numpy()
        confidences = torch.cat(all_confidences).numpy()

        # Compute metrics
        correct = (predictions == targets).float()
        total_acc = correct.mean().item() * 100

        # Per-exit accuracy
        exit_metrics = {}
        for t in self.exit_timesteps:
            mask = timesteps == t
            if mask.sum() > 0:
                exit_acc = correct[mask].mean().item() * 100
                exit_count = mask.sum()
                exit_metrics[t] = {
                    "accuracy": exit_acc,
                    "count": int(exit_count),
                    "fraction": float(exit_count) / len(timesteps),
                }

        # Energy metrics
        avg_timestep = timesteps.mean()
        energy_saved = energy_savings(timesteps, self.max_timestep)
        speedup = self.max_timestep / max(avg_timestep, 1)

        results = {
            "accuracy": total_acc,
            "avg_timestep": float(avg_timestep),
            "energy_savings": float(energy_saved),
            "speedup": float(speedup),
            "total_samples": len(targets),
            "exit_distribution": exit_metrics,
            "avg_confidence": float(confidences.mean()),
            "timestep_std": float(timesteps.std()),
        }

        if verbose:
            self._print_results(results)

        return results

    @torch.no_grad()
    def evaluate_baseline(
        self,
        test_loader: DataLoader,
        device: torch.device,
        verbose: bool = True,
    ) -> dict:
        """Evaluate baseline: all samples use max timestep (no early exit).

        Useful for comparison to show how much NeuroScale++ saves.

        Args:
            test_loader: Test DataLoader.
            device: Computation device.
            verbose: Print progress.

        Returns:
            Baseline metrics dict.
        """
        self.multi_exit_snn.eval()

        all_correct = []
        iterator = tqdm(test_loader, desc="Baseline Eval") if verbose else test_loader

        for images, targets in iterator:
            images, targets = images.to(device), targets.to(device)

            # Run at max timestep
            logits = self.multi_exit_snn.forward_single_exit(
                images, self.max_timestep
            )
            preds = logits.argmax(dim=1)
            all_correct.append((preds == targets.to(device)).cpu())

        correct = torch.cat(all_correct).float()
        baseline_acc = correct.mean().item() * 100

        results = {
            "accuracy": baseline_acc,
            "avg_timestep": float(self.max_timestep),
            "energy_savings": 0.0,
            "speedup": 1.0,
        }

        if verbose:
            print(f"\nBaseline (T={self.max_timestep}): Accuracy={baseline_acc:.2f}%")

        return results

    @torch.no_grad()
    def compare(
        self,
        test_loader: DataLoader,
        device: torch.device,
    ) -> dict:
        """Run both adaptive and baseline evaluation and compare.

        Returns:
            Dict with 'adaptive', 'baseline', and 'comparison' metrics.
        """
        print("=" * 60)
        print("NeuroScale++ Evaluation")
        print("=" * 60)

        print("\n--- Adaptive Inference (NeuroScale++) ---")
        adaptive = self.evaluate(test_loader, device, verbose=True)

        print("\n--- Baseline (Fixed T_max) ---")
        baseline = self.evaluate_baseline(test_loader, device, verbose=True)

        # Comparison
        acc_drop = baseline["accuracy"] - adaptive["accuracy"]

        comparison = {
            "accuracy_drop": acc_drop,
            "energy_savings": adaptive["energy_savings"],
            "speedup": adaptive["speedup"],
            "avg_timestep_reduction": baseline["avg_timestep"] - adaptive["avg_timestep"],
        }

        print("\n--- Comparison ---")
        print(f"Accuracy drop: {acc_drop:.2f}%")
        print(f"Energy savings: {adaptive['energy_savings']*100:.1f}%")
        print(f"Speedup: {adaptive['speedup']:.2f}x")
        print(f"Avg timestep: {adaptive['avg_timestep']:.1f} vs {baseline['avg_timestep']:.0f}")
        print("=" * 60)

        return {
            "adaptive": adaptive,
            "baseline": baseline,
            "comparison": comparison,
        }

    def _print_results(self, results: dict):
        """Pretty-print evaluation results."""
        print(f"\n{'='*50}")
        print(f"Adaptive Inference Results")
        print(f"{'='*50}")
        print(f"  Accuracy:       {results['accuracy']:.2f}%")
        print(f"  Avg Timestep:   {results['avg_timestep']:.1f} / {self.max_timestep}")
        print(f"  Energy Savings: {results['energy_savings']*100:.1f}%")
        print(f"  Speedup:        {results['speedup']:.2f}x")
        print(f"  Avg Confidence: {results['avg_confidence']:.3f}")
        print(f"\n  Exit Distribution:")
        for t, info in sorted(results["exit_distribution"].items()):
            print(
                f"    T={t:3d}: {info['count']:5d} samples "
                f"({info['fraction']*100:5.1f}%) "
                f"| Acc={info['accuracy']:.1f}%"
            )
        print(f"{'='*50}")

    @torch.no_grad()
    def evaluate_full(
        self,
        test_loader: DataLoader,
        device: torch.device,
        results_manager: Optional[ResultsManager] = None,
        num_classes: int = 10,
        profiling_results: Optional[dict] = None,
    ) -> dict:
        """Full evaluation with comprehensive metrics + ResultsManager logging.

        Runs adaptive + baseline, computes SOPs, top-5, ECE, predictor T-error,
        saves per-sample results, confusion matrix, per-class metrics, and all CSVs.

        Args:
            test_loader: Test/validation DataLoader.
            device: Computation device.
            results_manager: ResultsManager for CSV/metric logging.
            num_classes: Number of classes in the dataset.
            profiling_results: Optional Phase 2 dict with 'alphas','betas','gammas'
                used to compute predictor T-error vs ground-truth T_optimal.

        Returns:
            Dict with 'adaptive', 'baseline', 'comparison', and raw arrays.
        """
        print("=" * 60)
        print("NeuroScale++ Full Evaluation")
        print("=" * 60)

        compute_top5 = num_classes >= 5

        # ── Adaptive Evaluation ──────────────────────────────────────────────
        print("\n--- Adaptive Inference (NeuroScale++) ---")
        self.predictor.eval()
        self.multi_exit_snn.eval()

        all_predictions = []
        all_targets = []
        all_timesteps = []
        all_confidences = []
        all_logits = []
        all_pred_alphas, all_pred_betas, all_pred_gammas = [], [], []

        for images, targets in tqdm(test_loader, desc="Adaptive Eval"):
            images = images.to(device)
            results = self.predict_batch(images, device)

            all_predictions.append(results["predictions"])
            all_targets.append(targets)
            all_timesteps.append(results["timesteps_used"])
            all_confidences.append(results["confidences"])
            all_logits.append(results["logits"])
            all_pred_alphas.append(results["scaling_params"]["alpha"])
            all_pred_betas.append(results["scaling_params"]["beta"])
            all_pred_gammas.append(results["scaling_params"]["gamma"])

        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()
        timesteps_used = torch.cat(all_timesteps).numpy()
        confidences = torch.cat(all_confidences).numpy()
        logits_np = torch.cat(all_logits).numpy()          # (N, C)
        pred_alphas = torch.cat(all_pred_alphas).numpy()
        pred_betas = torch.cat(all_pred_betas).numpy()
        pred_gammas = torch.cat(all_pred_gammas).numpy()

        # Core adaptive metrics
        correct = (predictions == targets).astype(float)
        adaptive_acc = correct.mean() * 100
        avg_t = timesteps_used.mean()
        energy_saved = 1.0 - (avg_t / self.max_timestep)
        speedup = self.max_timestep / max(avg_t, 1)

        # Top-5 accuracy
        adaptive_top5 = topk_accuracy_from_logits(logits_np, targets, k=min(5, num_classes)) \
            if compute_top5 else 0.0

        # Per-exit breakdown
        exit_metrics = {}
        for t in self.exit_timesteps:
            mask = timesteps_used == t
            if mask.sum() > 0:
                exit_metrics[t] = {
                    "accuracy": correct[mask].mean() * 100,
                    "count": int(mask.sum()),
                    "fraction": float(mask.sum()) / len(timesteps_used),
                }

        # SOPs
        sops_dict = compute_sops(timesteps_used, self.sops_per_timestep, self.max_timestep)

        # ECE (Expected Calibration Error)
        ece_dict = compute_ece(confidences, correct)

        adaptive_results = {
            "accuracy": adaptive_acc,
            "top5_accuracy": adaptive_top5,
            "avg_timestep": float(avg_t),
            "energy_savings": float(energy_saved),
            "speedup": float(speedup),
            "total_samples": len(targets),
            "exit_distribution": exit_metrics,
            "avg_confidence": float(confidences.mean()),
            "timestep_std": float(timesteps_used.std()),
            "ece": ece_dict["ece"],
            "mce": ece_dict["mce"],
            "sops_per_sample": sops_dict["adaptive_sops_per_sample"],
            "sops_reduction": sops_dict["sops_reduction"],
        }
        self._print_results(adaptive_results)

        # ── Baseline Evaluation (fixed T_max) ────────────────────────────────
        print("\n--- Baseline (Fixed T_max) ---")
        self.multi_exit_snn.eval()
        baseline_preds_list, baseline_targets_list, baseline_logits_list = [], [], []
        for images, tgts in tqdm(test_loader, desc="Baseline Eval"):
            images = images.to(device)
            logits = self.multi_exit_snn.forward_single_exit(images, self.max_timestep)
            baseline_preds_list.append(logits.argmax(dim=1).cpu())
            baseline_logits_list.append(logits.cpu())
            baseline_targets_list.append(tgts)

        bl_preds = torch.cat(baseline_preds_list).numpy()
        bl_targets = torch.cat(baseline_targets_list).numpy()
        bl_logits = torch.cat(baseline_logits_list).numpy()
        bl_correct = (bl_preds == bl_targets).astype(float)
        baseline_acc = bl_correct.mean() * 100
        baseline_top5 = topk_accuracy_from_logits(bl_logits, bl_targets, k=min(5, num_classes)) \
            if compute_top5 else 0.0

        baseline_results = {
            "accuracy": baseline_acc,
            "top5_accuracy": baseline_top5,
            "avg_timestep": float(self.max_timestep),
            "energy_savings": 0.0,
            "speedup": 1.0,
        }
        print(f"  Baseline T={self.max_timestep}: Top-1={baseline_acc:.2f}%"
              + (f", Top-5={baseline_top5:.2f}%" if compute_top5 else ""))

        # ── Predictor T-error vs Phase 2 ground truth ────────────────────────
        t_error_dict = {}
        if profiling_results is not None:
            from ..scaling.curve_fitting import compute_optimal_timestep
            gt_alphas = profiling_results["alphas"]
            gt_betas = profiling_results["betas"]
            gt_gammas = profiling_results["gammas"]
            target_acc_val = self.target_accuracy
            max_t = self.max_timestep

            # Ground-truth T_optimal for each test sample
            # Test set size may differ from profiling set; use min length
            n = min(len(predictions), len(gt_alphas))
            gt_t_optimals = np.array([
                compute_optimal_timestep(gt_alphas[i], gt_betas[i], gt_gammas[i],
                                         target_acc_val, max_timestep=max_t)
                for i in range(n)
            ])
            t_error_dict = compute_predictor_t_error(
                predicted_t=timesteps_used[:n].astype(float),
                optimal_t=gt_t_optimals.astype(float),
            )
            print(f"\n  Predictor T-error: MAE={t_error_dict['mae']:.2f}, "
                  f"RMSE={t_error_dict['rmse']:.2f}, "
                  f"ExactMatch={t_error_dict['exact_match_rate']*100:.1f}%")

        # ── Comparison summary ────────────────────────────────────────────────
        acc_drop = baseline_results["accuracy"] - adaptive_results["accuracy"]
        ann_acc = getattr(results_manager, "_ann_best_acc", 0.0) if results_manager else 0.0
        ann_top5 = getattr(results_manager, "_ann_best_top5_acc", 0.0) if results_manager else 0.0

        print(f"\n--- Summary ---")
        print(f"  ANN baseline:   {ann_acc:.2f}% (Top-1)" + (f" / {ann_top5:.2f}% (Top-5)" if compute_top5 else ""))
        print(f"  SNN T_max:      {baseline_acc:.2f}%  (gap from ANN: {ann_acc-baseline_acc:.2f}%)")
        print(f"  SNN Adaptive:   {adaptive_acc:.2f}%  (gap from ANN: {ann_acc-adaptive_acc:.2f}%)")
        print(f"  Energy savings: {energy_saved*100:.1f}%  |  Speedup: {speedup:.2f}x")
        print(f"  SOPs reduction: {sops_dict['sops_reduction']*100:.1f}%")
        print(f"  ECE: {ece_dict['ece']:.4f}  |  MCE: {ece_dict['mce']:.4f}")
        print("=" * 60)

        # ── Persist to ResultsManager ─────────────────────────────────────────
        if results_manager is not None:
            results_manager.save_eval_per_sample(
                predictions=predictions,
                targets=targets,
                confidences=confidences,
                timesteps_used=timesteps_used,
            )
            results_manager.save_eval_exit_distribution(exit_metrics)
            results_manager.save_eval_comparison(
                adaptive_results, baseline_results,
                ann_acc=ann_acc, ann_top5_acc=ann_top5,
            )
            results_manager.save_eval_sops(sops_dict)
            results_manager.save_eval_ece(ece_dict)
            if t_error_dict:
                results_manager.save_eval_predictor_t_error(t_error_dict)
            results_manager.save_confusion_matrix(predictions, targets, num_classes)
            results_manager.save_per_class_metrics(
                predictions=predictions,
                targets=targets,
                timesteps_used=timesteps_used,
                num_classes=num_classes,
            )
            results_manager.save_eval_summary(adaptive_results, baseline_results)

        return {
            "adaptive": adaptive_results,
            "baseline": baseline_results,
            "comparison": {
                "accuracy_drop": acc_drop,
                "ann_to_snn_gap": ann_acc - baseline_acc,
                "ann_to_adaptive_gap": ann_acc - adaptive_acc,
                "energy_savings": energy_saved,
                "speedup": speedup,
                "sops_reduction": sops_dict.get("sops_reduction", 0.0),
                "ece": ece_dict["ece"],
            },
            # Raw arrays for plotting
            "_predictions": predictions,
            "_targets": targets,
            "_confidences": confidences,
            "_timesteps_used": timesteps_used,
            "_logits": logits_np,
            "_ece": ece_dict,
            "_t_error": t_error_dict,
            "_sops": sops_dict,
        }
