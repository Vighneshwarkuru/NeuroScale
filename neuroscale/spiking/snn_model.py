"""SNN Model wrapper.

Wraps a converted ANN into a full SNN that runs over multiple timesteps.
Handles:
- Spike encoding of continuous inputs
- Running the network for T timesteps
- Accumulating output spike counts for classification
- Resetting neuron states between samples
"""

import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict

from .neurons import IFNeuron, LIFNeuron
from .layers import (
    SpikingConv2d, SpikingLinear, SpikingBatchNorm2d,
    SpikingMaxPool2d, SpikingAvgPool2d
)


class SNNModel(nn.Module):
    """Spiking Neural Network model converted from an ANN.

    Runs the network over multiple timesteps, using rate coding:
    - Input: repeated at each timestep (direct input injection)
    - Internal: neurons integrate and fire based on IF/LIF dynamics
    - Output: accumulated over timesteps (spike count / T = firing rate)
    """

    def __init__(
        self,
        ann_model: nn.Module,
        thresholds: list[float],
        max_timestep: int = 64,
        neuron_type: str = "if",
    ):
        """
        Args:
            ann_model: The ANN model with normalized weights (from converter).
            thresholds: List of firing thresholds per layer.
            max_timestep: Maximum timesteps to simulate.
            neuron_type: 'if' or 'lif'.
        """
        super().__init__()
        self.max_timestep = max_timestep
        self.neuron_type = neuron_type
        self.thresholds = thresholds

        # Build spiking layers from ANN architecture
        self.spiking_layers = self._build_spiking_layers(ann_model)

        # Copy the final classification layer (no spiking - accumulates)
        self.classifier = self._extract_classifier(ann_model)

    def _build_spiking_layers(self, ann_model: nn.Module) -> nn.ModuleList:
        """Convert ANN layers to spiking equivalents."""
        spiking_layers = nn.ModuleList()
        threshold_idx = 0

        for name, module in ann_model.named_modules():
            if isinstance(module, nn.Conv2d):
                threshold = self.thresholds[threshold_idx] if threshold_idx < len(self.thresholds) else 1.0
                spiking_conv = SpikingConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    bias=module.bias is not None,
                    threshold=threshold,
                    neuron_type=self.neuron_type,
                )
                # Transfer weights
                spiking_conv.conv.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    spiking_conv.conv.bias.data = module.bias.data.clone()
                spiking_layers.append(spiking_conv)
                threshold_idx += 1

            elif isinstance(module, nn.MaxPool2d):
                spiking_layers.append(
                    SpikingMaxPool2d(kernel_size=module.kernel_size, stride=module.stride)
                )

            elif isinstance(module, nn.AdaptiveAvgPool2d):
                output_size = module.output_size
                if isinstance(output_size, tuple):
                    output_size = output_size[0]
                spiking_layers.append(SpikingAvgPool2d(output_size=output_size))

            elif isinstance(module, nn.BatchNorm2d):
                spiking_bn = SpikingBatchNorm2d(module.num_features)
                spiking_bn.bn.load_state_dict(module.state_dict())
                spiking_layers.append(spiking_bn)

        return spiking_layers

    def _extract_classifier(self, ann_model: nn.Module) -> nn.Module:
        """Extract the final linear classifier layer."""
        # Find the last Linear layer
        last_linear = None
        for module in ann_model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is not None:
            classifier = nn.Linear(
                last_linear.in_features, last_linear.out_features,
                bias=last_linear.bias is not None
            )
            classifier.weight.data = last_linear.weight.data.clone()
            if last_linear.bias is not None:
                classifier.bias.data = last_linear.bias.data.clone()
            return classifier

        raise ValueError("No Linear layer found in ANN model")

    def reset_neurons(self):
        """Reset all neuron membrane potentials."""
        for layer in self.spiking_layers:
            if hasattr(layer, "reset"):
                layer.reset()

    def forward(self, x: torch.Tensor, timesteps: Optional[int] = None) -> torch.Tensor:
        """Run SNN for T timesteps.

        Args:
            x: Input image (batch, channels, H, W) - continuous valued.
            timesteps: Number of timesteps to simulate. If None, uses max_timestep.

        Returns:
            Output logits (batch, num_classes) - averaged over timesteps.
        """
        if timesteps is None:
            timesteps = self.max_timestep

        self.reset_neurons()

        # Accumulate output over timesteps
        output_sum = None

        for t in range(timesteps):
            # Direct input injection: same input at every timestep
            spike = x

            # Pass through spiking layers
            for layer in self.spiking_layers:
                spike = layer(spike)

            # Flatten for classifier
            flat = spike.view(spike.size(0), -1)

            # Classify (no spiking on output - accumulate)
            out = self.classifier(flat)

            if output_sum is None:
                output_sum = out
            else:
                output_sum = output_sum + out

        # Average over timesteps
        return output_sum / timesteps

    def forward_at_timesteps(
        self, x: torch.Tensor, checkpoint_timesteps: list[int]
    ) -> dict[int, torch.Tensor]:
        """Run SNN and return outputs at specific timestep checkpoints.

        Used for profiling (Phase 2) and multi-exit inference.

        Args:
            x: Input image (batch, channels, H, W).
            checkpoint_timesteps: List of timesteps to record outputs at.
                                  Must be sorted ascending.

        Returns:
            Dict mapping timestep -> output logits at that point.
        """
        self.reset_neurons()

        outputs = {}
        output_sum = None
        max_t = max(checkpoint_timesteps)

        for t in range(1, max_t + 1):
            spike = x

            for layer in self.spiking_layers:
                spike = layer(spike)

            flat = spike.view(spike.size(0), -1)
            out = self.classifier(flat)

            if output_sum is None:
                output_sum = out
            else:
                output_sum = output_sum + out

            if t in checkpoint_timesteps:
                outputs[t] = output_sum / t

        return outputs
