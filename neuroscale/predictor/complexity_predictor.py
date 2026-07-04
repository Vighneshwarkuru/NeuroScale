"""Complexity Predictor Network.

A lightweight CNN that takes a raw input image and predicts per-sample
scaling law parameters (α, β, γ). These parameters define the power-law
relationship between SNN timesteps and accuracy for that specific sample:

    Performance(T) = α × T^β + γ

From these parameters, we derive T_optimal — the minimum timesteps needed
to reach a target accuracy level for that sample.

Design goals:
- Must be very cheap (< 5% of the SNN's compute)
- Operates on the original image (not SNN features)
- Trained with supervision from Phase 2 profiling data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ComplexityPredictor(nn.Module):
    """Lightweight CNN that predicts sample complexity (α, β, γ).

    Architecture:
    - A few conv layers to extract spatial features
    - Global average pooling
    - MLP head to predict 3 scaling law parameters

    The network is intentionally small — it needs to run before the SNN
    and its overhead must be negligible compared to the savings from
    early exiting.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        hidden_dims: list[int] = [64, 128, 64],
        num_params: int = 3,
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB).
            image_size: Input spatial size (32 for CIFAR, 224 for ImageNet).
            hidden_dims: Hidden layer dimensions for the MLP head.
            num_params: Number of scaling law parameters to predict (α, β, γ = 3).
        """
        super().__init__()
        self.image_size = image_size
        self.num_params = num_params

        # Feature extractor: lightweight convolutions
        if image_size <= 32:
            # CIFAR-sized inputs
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 16x16

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 8x8

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            )
            feature_dim = 128
        else:
            # ImageNet-sized inputs
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),  # 56x56

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),  # 28x28

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),  # 14x14

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),  # 7x7

                nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            )
            feature_dim = 256

        # MLP head to predict scaling law parameters
        mlp_layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim

        mlp_layers.append(nn.Linear(in_dim, num_params))
        self.head = nn.Sequential(*mlp_layers)

        # Parameter-specific output activations
        # α > 0 (amplitude), β > 0 (exponent), γ ∈ [0, 1] (asymptote)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict scaling law parameters for each input sample.

        Args:
            x: Input images (batch, channels, H, W).

        Returns:
            Dictionary with keys 'alpha', 'beta', 'gamma', each (batch,).
            Also includes 'raw' with the raw 3-parameter output (batch, 3).
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        raw_params = self.head(features)

        # Apply constraints to ensure physically meaningful parameters:
        # α > 0: amplitude of the scaling law (use softplus)
        alpha = F.softplus(raw_params[:, 0])

        # β ∈ (0, 1]: exponent (use sigmoid to bound, then scale)
        # β represents diminishing returns — should be fractional
        beta = torch.sigmoid(raw_params[:, 1])

        # γ ∈ [0, 1]: baseline/asymptote (use sigmoid)
        gamma = torch.sigmoid(raw_params[:, 2])

        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "raw": raw_params,
        }

    def predict_optimal_timestep(
        self,
        x: torch.Tensor,
        target_accuracy: float = 0.95,
        min_timestep: int = 2,
        max_timestep: int = 64,
        exit_timesteps: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Predict the optimal (minimum) timestep for each sample.

        Given the predicted scaling law parameters, compute the minimum T
        such that Performance(T) >= target_accuracy.

        From: Performance(T) = α × T^β + γ
        Solving for T: T = ((target - γ) / α) ^ (1/β)

        Args:
            x: Input images (batch, channels, H, W).
            target_accuracy: Target performance level (0 to 1).
            min_timestep: Minimum allowed timestep.
            max_timestep: Maximum allowed timestep.
            exit_timesteps: If provided, snap to nearest valid exit point.

        Returns:
            Optimal timestep per sample (batch,), as float tensor.
        """
        params = self.forward(x)
        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]

        # Solve: target = α × T^β + γ  →  T = ((target - γ) / α) ^ (1/β)
        numerator = (target_accuracy - gamma).clamp(min=1e-6)
        ratio = (numerator / alpha.clamp(min=1e-6))

        # T = ratio ^ (1/β)
        # Clamp ratio to avoid negative/zero before power
        ratio = ratio.clamp(min=1e-6)
        inv_beta = (1.0 / beta.clamp(min=0.01))
        t_optimal = ratio.pow(inv_beta)

        # Clamp to valid range
        t_optimal = t_optimal.clamp(min=min_timestep, max=max_timestep)

        # Snap to nearest exit point if provided
        if exit_timesteps is not None:
            t_optimal = self._snap_to_exits(t_optimal, exit_timesteps)

        return t_optimal

    def _snap_to_exits(
        self, t_optimal: torch.Tensor, exit_timesteps: list[int]
    ) -> torch.Tensor:
        """Snap continuous T values to the nearest valid exit timestep (rounding up).

        Args:
            t_optimal: Continuous timestep values (batch,).
            exit_timesteps: Valid exit points (sorted ascending).

        Returns:
            Snapped timesteps (batch,).
        """
        exits = torch.tensor(exit_timesteps, dtype=t_optimal.dtype, device=t_optimal.device)

        # For each sample, find the smallest exit >= t_optimal
        # Shape: (batch, num_exits)
        t_expanded = t_optimal.unsqueeze(1)
        exits_expanded = exits.unsqueeze(0)

        # Mask: exits >= t_optimal
        valid = exits_expanded >= t_expanded

        # For each row, find the first valid exit (smallest valid)
        # If no valid exit, use the maximum
        max_exit = exits[-1]
        snapped = torch.where(
            valid.any(dim=1),
            exits_expanded.masked_fill(~valid, max_exit + 1).min(dim=1).values,
            max_exit.expand(t_optimal.size(0)),
        )

        return snapped
