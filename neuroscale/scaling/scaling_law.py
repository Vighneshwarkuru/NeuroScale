"""Scaling Law Module.

Provides a differentiable scaling law model that can be used during
joint training (Phase 3). Also includes utilities for computing energy
metrics and analyzing the scaling behavior of the system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ScalingLawModule(nn.Module):
    """Differentiable power-law scaling model.

    Used during training to:
    1. Compute expected performance at predicted timesteps
    2. Provide gradient signal for the complexity predictor
    3. Balance accuracy vs. energy in the loss function

    The scaling law: Performance(T) = α × T^β + γ
    """

    def __init__(
        self,
        exit_timesteps: list[int],
        target_accuracy: float = 0.95,
        energy_weight: float = 0.5,
    ):
        """
        Args:
            exit_timesteps: Valid exit timesteps (e.g., [4, 8, 16, 32, 64]).
            target_accuracy: Target accuracy threshold for T_optimal.
            energy_weight: Weight for energy loss vs. accuracy loss.
        """
        super().__init__()
        self.exit_timesteps = sorted(exit_timesteps)
        self.target_accuracy = target_accuracy
        self.energy_weight = energy_weight
        self.max_timestep = max(exit_timesteps)

        # Register exit timesteps as buffer
        self.register_buffer(
            "exit_t", torch.tensor(exit_timesteps, dtype=torch.float32)
        )

    def predicted_performance(
        self, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute predicted performance at given timesteps.

        Args:
            alpha: Amplitude (batch,).
            beta: Exponent (batch,).
            gamma: Baseline (batch,).
            t: Timesteps (batch,) or scalar.

        Returns:
            Predicted performance (batch,).
        """
        return alpha * t.pow(beta) + gamma

    def compute_t_optimal(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optimal timestep from scaling law parameters (differentiable).

        Solves: target = α × T^β + γ  →  T = ((target - γ) / α)^(1/β)

        Args:
            alpha: Amplitude (batch,).
            beta: Exponent (batch,).
            gamma: Baseline (batch,).

        Returns:
            Optimal timestep (batch,) as continuous float (for gradient flow).
        """
        numerator = (self.target_accuracy - gamma).clamp(min=1e-6)
        ratio = (numerator / alpha.clamp(min=1e-6)).clamp(min=1e-6)
        inv_beta = 1.0 / beta.clamp(min=0.01)
        t_optimal = ratio.pow(inv_beta)

        # Clamp to valid range
        t_optimal = t_optimal.clamp(min=self.exit_timesteps[0], max=self.max_timestep)
        return t_optimal

    def compute_loss(
        self,
        predicted_params: dict[str, torch.Tensor],
        target_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute training loss for the scaling law predictor.

        Combines:
        1. Parameter regression loss (MSE on α, β, γ)
        2. Timestep prediction loss (how close is predicted T to actual T)
        3. Energy regularization (penalize large predicted timesteps)

        Args:
            predicted_params: Dict with 'alpha', 'beta', 'gamma' from predictor.
            target_params: Dict with 'alpha', 'beta', 'gamma' from curve fitting.

        Returns:
            Dict with 'total', 'param_loss', 'timestep_loss', 'energy_loss'.
        """
        # Parameter regression loss
        param_loss = (
            (predicted_params["alpha"] - target_params["alpha"]).pow(2).mean()
            + (predicted_params["beta"] - target_params["beta"]).pow(2).mean()
            + (predicted_params["gamma"] - target_params["gamma"]).pow(2).mean()
        ) / 3.0

        # Timestep prediction loss
        pred_t = self.compute_t_optimal(
            predicted_params["alpha"],
            predicted_params["beta"],
            predicted_params["gamma"],
        )
        target_t = self.compute_t_optimal(
            target_params["alpha"],
            target_params["beta"],
            target_params["gamma"],
        )
        # Use log-space MSE for timesteps (since they span orders of magnitude)
        timestep_loss = (pred_t.log() - target_t.log()).pow(2).mean()

        # Energy regularization: penalize high predicted timesteps
        # Normalized by max_timestep so it's in [0, 1]
        energy_loss = (pred_t / self.max_timestep).mean()

        # Total loss
        total = param_loss + timestep_loss + self.energy_weight * energy_loss

        return {
            "total": total,
            "param_loss": param_loss,
            "timestep_loss": timestep_loss,
            "energy_loss": energy_loss,
        }

    def compute_energy_metrics(
        self, timesteps_used: torch.Tensor
    ) -> dict[str, float]:
        """Compute energy-related metrics.

        Args:
            timesteps_used: Actual timesteps used per sample (batch,).

        Returns:
            Dict with energy metrics.
        """
        avg_t = timesteps_used.float().mean().item()
        max_t = float(self.max_timestep)

        return {
            "avg_timestep": avg_t,
            "max_timestep": max_t,
            "energy_savings": 1.0 - (avg_t / max_t),
            "speedup": max_t / max(avg_t, 1.0),
        }

    def snap_to_exit(self, t_continuous: torch.Tensor) -> torch.Tensor:
        """Snap continuous timestep predictions to valid exit points.

        Args:
            t_continuous: Continuous T values (batch,).

        Returns:
            Snapped to nearest valid exit (batch,), rounding up.
        """
        # (batch, 1) vs (1, num_exits)
        t_expanded = t_continuous.unsqueeze(1)
        exits = self.exit_t.unsqueeze(0)

        # Find smallest exit >= t for each sample
        valid = exits >= t_expanded
        max_exit = self.exit_t[-1]

        # Replace invalid positions with large value, then take min
        masked = exits.masked_fill(~valid, max_exit + 1)
        snapped = masked.min(dim=1).values

        # Any that had no valid exit get max
        snapped = snapped.clamp(max=max_exit)

        return snapped
