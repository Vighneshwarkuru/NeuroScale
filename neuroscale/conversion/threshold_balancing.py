"""Threshold balancing for ANN-to-SNN conversion.

The key idea: set each spiking neuron's firing threshold equal to the maximum
(or a high percentile of) activations observed at that layer in the original ANN.
This ensures the SNN's firing rates can faithfully represent the ANN's activation values.

Reference: Diehl et al., "Fast-classifying, high-accuracy spiking deep networks
through weight and threshold balancing" (IJCNN 2015).
"""

import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict


class ThresholdBalancer:
    """Compute per-layer firing thresholds from ANN activations.

    For each ReLU layer in the ANN, we record the maximum activation
    (or a percentile) over a calibration dataset. These values become
    the firing thresholds for the corresponding spiking neurons.
    """

    def __init__(self, percentile: float = 99.9):
        """
        Args:
            percentile: Percentile of activations to use as threshold.
                        99.9 works better than 100 (max) to avoid outlier sensitivity.
        """
        self.percentile = percentile
        self.thresholds: OrderedDict[str, float] = OrderedDict()
        self._hooks = []
        self._activations: dict[str, list[torch.Tensor]] = {}

    def compute_thresholds(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> OrderedDict[str, float]:
        """Compute firing thresholds by observing ANN activations.

        Args:
            model: Pretrained ANN model.
            calibration_loader: DataLoader with calibration samples (no augmentation).
            device: Device to run on.

        Returns:
            OrderedDict mapping layer names to threshold values.
        """
        model = model.to(device)
        model.eval()

        # Register hooks on all ReLU layers (and the layers before them)
        self._activations = {}
        self._hooks = []
        self._register_hooks(model)

        # Collect activations
        with torch.no_grad():
            for images, _ in calibration_loader:
                images = images.to(device)
                model(images)

        # Compute percentile thresholds
        self.thresholds = OrderedDict()
        for name, acts in self._activations.items():
            all_acts = torch.cat(acts, dim=0)
            if self.percentile >= 100.0:
                threshold = all_acts.max().item()
            else:
                threshold = torch.quantile(
                    all_acts.flatten().float(),
                    self.percentile / 100.0
                ).item()
            # Avoid zero thresholds
            self.thresholds[name] = max(threshold, 1e-5)

        # Clean up hooks
        self._remove_hooks()

        return self.thresholds

    def _register_hooks(self, model: nn.Module):
        """Register forward hooks on ReLU/activation layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook function that records activations."""
        def hook_fn(module, input, output):
            if name not in self._activations:
                self._activations[name] = []
            # Store max per channel to save memory
            if output.dim() == 4:
                # Conv layer: store max per spatial location across batch
                max_act = output.detach().cpu()
            else:
                max_act = output.detach().cpu()
            self._activations[name].append(max_act)
        return hook_fn

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    def get_threshold(self, layer_name: str) -> float:
        """Get threshold for a specific layer."""
        if layer_name not in self.thresholds:
            raise KeyError(f"No threshold computed for layer: {layer_name}")
        return self.thresholds[layer_name]
