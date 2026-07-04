"""ANN-to-SNN Converter.

Orchestrates the full conversion pipeline:
1. Analyze ANN architecture
2. Compute firing thresholds via threshold balancing
3. Optionally calibrate thresholds
4. Replace ReLU with spiking neurons
5. Return a ready-to-run SNN model

The conversion preserves weights — only activations change from continuous
(ReLU) to temporal spike trains (IF/LIF neurons).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Optional
import copy

from .threshold_balancing import ThresholdBalancer
from .calibration import ModelCalibrator


class ANNtoSNNConverter:
    """Convert a pretrained ANN to an equivalent SNN.

    The converter replaces all ReLU activations with Integrate-and-Fire (IF)
    neurons, using the computed thresholds. Weights are directly transferred.
    """

    def __init__(
        self,
        percentile: float = 99.9,
        calibrate: bool = True,
        calibration_steps: int = 100,
    ):
        """
        Args:
            percentile: Percentile for threshold balancing (99.9 recommended).
            calibrate: Whether to apply post-balancing calibration.
            calibration_steps: Number of calibration optimization steps.
        """
        self.percentile = percentile
        self.calibrate = calibrate
        self.calibration_steps = calibration_steps
        self.thresholds: OrderedDict[str, float] = OrderedDict()

    def convert(
        self,
        ann_model: nn.Module,
        calibration_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[nn.Module, OrderedDict[str, float]]:
        """Convert ANN to SNN.

        Args:
            ann_model: Pretrained ANN model.
            calibration_loader: DataLoader for calibration (no augmentation).
            device: Computation device.

        Returns:
            Tuple of (snn_model_weights, thresholds).
            The actual SNN model construction happens in the spiking module
            since it needs spiking neuron layers. Here we compute the thresholds
            and prepare the weight transfer.
        """
        ann_model = ann_model.to(device)
        ann_model.eval()

        # Step 1: Compute thresholds via threshold balancing
        balancer = ThresholdBalancer(percentile=self.percentile)
        self.thresholds = balancer.compute_thresholds(
            ann_model, calibration_loader, device
        )

        # Step 2: Optional calibration
        if self.calibrate:
            calibrator = ModelCalibrator(num_steps=self.calibration_steps)
            self.thresholds = calibrator.calibrate(
                ann_model, self.thresholds, calibration_loader, device
            )

        # Step 3: Prepare the model for SNN conversion
        # Deep copy to avoid modifying the original
        snn_base = copy.deepcopy(ann_model)

        # Normalize weights by thresholds (weight normalization for SNN)
        snn_base = self._normalize_weights(snn_base)

        return snn_base, self.thresholds

    def _normalize_weights(self, model: nn.Module) -> nn.Module:
        """Normalize weights layer-by-layer for SNN conversion.

        For each layer, divide weights by the input layer's threshold
        and multiply by the current layer's threshold. This ensures
        spike rates correctly represent activation magnitudes.
        """
        threshold_list = list(self.thresholds.values())

        layer_idx = 0
        prev_threshold = 1.0  # Input layer has no threshold

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if layer_idx < len(threshold_list):
                    current_threshold = threshold_list[layer_idx]

                    # Normalize: w_snn = w_ann * (prev_threshold / current_threshold)
                    with torch.no_grad():
                        module.weight.data *= (prev_threshold / current_threshold)
                        if module.bias is not None:
                            module.bias.data /= current_threshold

                    prev_threshold = current_threshold
                    layer_idx += 1

        return model

    def get_thresholds(self) -> OrderedDict[str, float]:
        """Return computed thresholds."""
        return self.thresholds

    def get_threshold_list(self) -> list[float]:
        """Return thresholds as a list (ordered by layer depth)."""
        return list(self.thresholds.values())
