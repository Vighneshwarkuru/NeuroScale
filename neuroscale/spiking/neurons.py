"""Spiking neuron models for SNN inference.

Implements Integrate-and-Fire (IF) and Leaky Integrate-and-Fire (LIF) neurons
that replace ReLU activations in the converted SNN.

Key behavior:
- Neurons accumulate input over timesteps into a membrane potential.
- When potential exceeds threshold, the neuron fires (emits spike = 1).
- After firing, potential is reset (subtraction reset, not hard reset).
"""

import torch
import torch.nn as nn
from typing import Optional


class IFNeuron(nn.Module):
    """Integrate-and-Fire neuron.

    Dynamics:
        V[t] = V[t-1] + X[t]
        S[t] = 1 if V[t] >= threshold, else 0
        V[t] = V[t] - threshold * S[t]  (soft reset)

    This is the standard neuron for ANN-to-SNN conversion because IF neurons
    with rate coding can exactly represent ReLU activations given enough timesteps.
    """

    def __init__(self, threshold: float = 1.0):
        """
        Args:
            threshold: Firing threshold voltage.
        """
        super().__init__()
        self.threshold = threshold
        self.membrane_potential: Optional[torch.Tensor] = None

    def reset(self):
        """Reset membrane potential (call at start of each new input)."""
        self.membrane_potential = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one timestep of input.

        Args:
            x: Input current for this timestep (batch, channels, [height, width]).

        Returns:
            Binary spike tensor (same shape as x).
        """
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x)

        # Integrate
        self.membrane_potential = self.membrane_potential + x

        # Fire
        spikes = (self.membrane_potential >= self.threshold).float()

        # Reset (soft reset: subtract threshold)
        self.membrane_potential = self.membrane_potential - spikes * self.threshold

        return spikes


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron.

    Dynamics:
        V[t] = beta * V[t-1] + X[t]
        S[t] = 1 if V[t] >= threshold, else 0
        V[t] = V[t] - threshold * S[t]  (soft reset)

    The leak factor (beta) causes membrane potential to decay over time,
    making LIF neurons more biologically realistic but requiring more timesteps
    for accurate rate coding.
    """

    def __init__(self, threshold: float = 1.0, beta: float = 0.9):
        """
        Args:
            threshold: Firing threshold voltage.
            beta: Membrane potential decay factor (0 < beta < 1). Higher = less leak.
        """
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.membrane_potential: Optional[torch.Tensor] = None

    def reset(self):
        """Reset membrane potential."""
        self.membrane_potential = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one timestep of input.

        Args:
            x: Input current for this timestep.

        Returns:
            Binary spike tensor.
        """
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x)

        # Leak + Integrate
        self.membrane_potential = self.beta * self.membrane_potential + x

        # Fire
        spikes = (self.membrane_potential >= self.threshold).float()

        # Soft reset
        self.membrane_potential = self.membrane_potential - spikes * self.threshold

        return spikes
