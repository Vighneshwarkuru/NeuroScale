"""Spiking equivalents of standard ANN layers.

These layers wrap standard Conv2d/Linear/BatchNorm with spiking neurons,
allowing direct weight transfer from the converted ANN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .neurons import IFNeuron, LIFNeuron


class SpikingConv2d(nn.Module):
    """Spiking convolutional layer.

    Applies convolution to spike inputs and integrates into IF/LIF neurons.
    BatchNorm is folded into the convolution weights (absorbed during conversion).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        threshold: float = 1.0,
        neuron_type: str = "if",
        beta: float = 0.9,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

        if neuron_type == "if":
            self.neuron = IFNeuron(threshold=threshold)
        elif neuron_type == "lif":
            self.neuron = LIFNeuron(threshold=threshold, beta=beta)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

    def reset(self):
        """Reset neuron states."""
        self.neuron.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for one timestep.

        Args:
            x: Spike input (batch, in_channels, H, W).

        Returns:
            Output spikes (batch, out_channels, H', W').
        """
        out = self.conv(x)
        out = self.neuron(out)
        return out


class SpikingLinear(nn.Module):
    """Spiking fully-connected layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 1.0,
        neuron_type: str = "if",
        beta: float = 0.9,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if neuron_type == "if":
            self.neuron = IFNeuron(threshold=threshold)
        elif neuron_type == "lif":
            self.neuron = LIFNeuron(threshold=threshold, beta=beta)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

    def reset(self):
        """Reset neuron states."""
        self.neuron.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for one timestep.

        Args:
            x: Spike input (batch, in_features).

        Returns:
            Output spikes (batch, out_features).
        """
        out = self.linear(x)
        out = self.neuron(out)
        return out


class SpikingBatchNorm2d(nn.Module):
    """BatchNorm for spiking networks.

    In converted SNNs, BatchNorm is typically folded into preceding Conv weights.
    This layer handles the case where BN is kept separate (threshold-dependent BN).
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        # In SNN mode, BN operates on the accumulated membrane potential
        # rather than individual spikes
        self.bn.eval()  # Always in eval mode for SNN inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization.

        Args:
            x: Input tensor (batch, channels, H, W).

        Returns:
            Normalized tensor.
        """
        return self.bn(x)


class SpikingMaxPool2d(nn.Module):
    """Max pooling for spike trains.

    For binary spikes, max pooling passes through a spike if any input
    in the pooling window spiked. This is equivalent to OR pooling.
    """

    def __init__(self, kernel_size: int, stride: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply max pooling to spikes."""
        return self.pool(x)


class SpikingAvgPool2d(nn.Module):
    """Average pooling for spike trains.

    Averages spike counts over the pooling window. Output is no longer
    strictly binary but represents a local spike rate.
    """

    def __init__(self, kernel_size: int = None, output_size: int = 1):
        super().__init__()
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size)
        else:
            self.pool = nn.AdaptiveAvgPool2d((output_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply average pooling to spikes."""
        return self.pool(x)
