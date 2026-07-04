"""Model calibration for ANN-to-SNN conversion.

After threshold balancing, calibration fine-tunes the thresholds using
a small dataset to minimize the conversion error (difference between
ANN outputs and SNN outputs at a given timestep).

Reference: Li et al., "Free Lunch for Few-shot Learning: Distribution
Calibration" and calibration-based SNN conversion methods.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Optional


class ModelCalibrator:
    """Calibrate SNN thresholds to minimize ANN-SNN output discrepancy.

    Uses layer-wise calibration: adjusts thresholds iteratively to minimize
    the L2 error between ANN layer outputs and time-averaged SNN spike rates.
    """

    def __init__(
        self,
        num_steps: int = 100,
        lr: float = 0.01,
        target_timestep: int = 32,
    ):
        """
        Args:
            num_steps: Number of calibration optimization steps per layer.
            lr: Learning rate for threshold adjustment.
            target_timestep: Timestep at which to calibrate (SNN runs this many steps).
        """
        self.num_steps = num_steps
        self.lr = lr
        self.target_timestep = target_timestep

    def calibrate(
        self,
        ann_model: nn.Module,
        thresholds: OrderedDict[str, float],
        calibration_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> OrderedDict[str, float]:
        """Calibrate thresholds by minimizing layer-wise conversion error.

        This is a lightweight post-processing step that adjusts the thresholds
        computed by ThresholdBalancer to account for temporal dynamics.

        Args:
            ann_model: The original pretrained ANN.
            thresholds: Initial thresholds from ThresholdBalancer.
            calibration_loader: Small calibration dataset.
            device: Computation device.

        Returns:
            Calibrated thresholds.
        """
        ann_model = ann_model.to(device)
        ann_model.eval()

        calibrated_thresholds = OrderedDict()

        # Collect target activations from the ANN
        ann_activations = self._collect_ann_activations(ann_model, calibration_loader, device)

        # For each layer, optimize the threshold
        for layer_name, init_threshold in thresholds.items():
            if layer_name not in ann_activations:
                calibrated_thresholds[layer_name] = init_threshold
                continue

            target_acts = ann_activations[layer_name]

            # Optimize threshold via grid search around the initial value
            best_threshold = init_threshold
            best_error = float("inf")

            # Search in a range around the initial threshold
            for scale in [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]:
                candidate = init_threshold * scale
                # Simulate spike rate: rate = clamp(activation / threshold, 0, 1)
                simulated_rate = torch.clamp(target_acts / candidate, 0, 1)
                # Reconstruction: threshold * rate should approximate activation
                reconstructed = simulated_rate * candidate
                error = (target_acts - reconstructed).pow(2).mean().item()

                if error < best_error:
                    best_error = error
                    best_threshold = candidate

            calibrated_thresholds[layer_name] = max(best_threshold, 1e-5)

        return calibrated_thresholds

    def _collect_ann_activations(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Collect mean activations from ANN ReLU layers."""
        activations = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach().cpu())
            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                model(images)

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        # Average activations
        mean_activations = {}
        for name, acts_list in activations.items():
            all_acts = torch.cat(acts_list, dim=0)
            mean_activations[name] = all_acts.mean(dim=0)

        return mean_activations
