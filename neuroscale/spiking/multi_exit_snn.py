"""Multi-Exit SNN with early-exit branches.

Extends the base SNNModel with intermediate classifiers at different timestep
checkpoints. During inference, the network can exit early once the complexity
predictor determines sufficient timesteps have been reached.

Exit points (e.g., T=4, 8, 16, 32, 64) each have a lightweight classifier
that reads the accumulated spike rates up to that point.
"""

import torch
import torch.nn as nn
from typing import Optional

from .snn_model import SNNModel


class ExitBranch(nn.Module):
    """Lightweight early-exit classifier branch.

    Takes the accumulated spike output at a given timestep and produces
    class logits. Uses a small MLP to allow slightly different decision
    boundaries at different timesteps.
    """

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 128):
        """
        Args:
            in_features: Flattened feature dimension from the SNN backbone.
            num_classes: Number of output classes.
            hidden_dim: Hidden layer size in the exit branch MLP.
        """
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flattened features (batch, in_features).

        Returns:
            Class logits (batch, num_classes).
        """
        return self.branch(x)


class MultiExitSNN(nn.Module):
    """SNN with multiple early-exit points at different timesteps.

    Architecture:
    - Base SNN backbone (converted from ANN)
    - Exit branches at specified timestep checkpoints
    - During training: all exits produce logits (multi-loss training)
    - During inference: exits based on complexity predictor's T_optimal

    The backbone runs timestep-by-timestep, and at each checkpoint we
    tap into the accumulated output to make a classification decision.
    """

    def __init__(
        self,
        snn_model: SNNModel,
        exit_timesteps: list[int],
        num_classes: int,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 128,
    ):
        """
        Args:
            snn_model: Base SNN model (already converted from ANN).
            exit_timesteps: List of timesteps where exits are placed (e.g., [4, 8, 16, 32, 64]).
            num_classes: Number of output classes.
            feature_dim: Dimension of flattened features. Auto-detected if None.
            hidden_dim: Hidden dimension for exit branch MLPs.
        """
        super().__init__()
        self.snn = snn_model
        self.exit_timesteps = sorted(exit_timesteps)
        self.num_classes = num_classes
        self.max_timestep = max(exit_timesteps)

        # Determine feature dimension from the SNN's classifier
        if feature_dim is None:
            feature_dim = snn_model.classifier.in_features

        # Create exit branches for each checkpoint
        self.exit_branches = nn.ModuleDict({
            str(t): ExitBranch(feature_dim, num_classes, hidden_dim)
            for t in self.exit_timesteps
        })

        # The final exit can share weights with the original classifier
        # or be its own branch. We keep it separate for flexibility.

    def forward(
        self, x: torch.Tensor, target_timestep: Optional[int] = None
    ) -> dict[int, torch.Tensor]:
        """Run SNN and return predictions at all exit points.

        In training mode, returns logits at ALL exit timesteps (for multi-loss).
        In eval mode with target_timestep, exits early at the specified timestep.

        Args:
            x: Input images (batch, channels, H, W).
            target_timestep: If set in eval mode, only run up to this timestep.

        Returns:
            Dict mapping timestep -> class logits (batch, num_classes).
        """
        self.snn.reset_neurons()

        outputs = {}
        spike_sum = None

        # Determine how far to run
        if target_timestep is not None and not self.training:
            run_until = target_timestep
        else:
            run_until = self.max_timestep

        for t in range(1, run_until + 1):
            # Run one timestep through SNN backbone
            spike = x
            for layer in self.snn.spiking_layers:
                spike = layer(spike)

            # Accumulate spike output
            flat = spike.view(spike.size(0), -1)
            if spike_sum is None:
                spike_sum = flat
            else:
                spike_sum = spike_sum + flat

            # At exit checkpoints, compute logits
            if t in self.exit_timesteps:
                # Normalize by timesteps elapsed
                features = spike_sum / t
                exit_logits = self.exit_branches[str(t)](features)
                outputs[t] = exit_logits

                # In eval mode, exit early if we've reached target
                if not self.training and target_timestep is not None and t >= target_timestep:
                    break

        return outputs

    def forward_single_exit(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        """Run SNN up to a specific timestep and return only that exit's prediction.

        Used during inference when the complexity predictor has determined T_optimal.

        Args:
            x: Input images (batch, channels, H, W).
            timestep: The timestep to exit at. Will be rounded up to the nearest exit point.

        Returns:
            Class logits (batch, num_classes) from the chosen exit.
        """
        # Round up to nearest exit point
        exit_t = self._nearest_exit(timestep)

        outputs = self.forward(x, target_timestep=exit_t)
        return outputs[exit_t]

    def forward_with_confidence(
        self, x: torch.Tensor, confidence_threshold: float = 0.9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run SNN with confidence-based early exit.

        Exits at the first timestep where the softmax confidence exceeds
        the threshold. Fallback: use the final exit.

        Args:
            x: Input images (batch, channels, H, W).
            confidence_threshold: Exit when max softmax prob exceeds this.

        Returns:
            Tuple of (predictions, exit_timesteps_per_sample).
        """
        batch_size = x.size(0)
        self.snn.reset_neurons()

        # Track which samples have exited
        exited = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        final_logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        exit_times = torch.full((batch_size,), self.max_timestep, device=x.device)

        spike_sum = None

        for t in range(1, self.max_timestep + 1):
            spike = x
            for layer in self.snn.spiking_layers:
                spike = layer(spike)

            flat = spike.view(batch_size, -1)
            if spike_sum is None:
                spike_sum = flat
            else:
                spike_sum = spike_sum + flat

            if t in self.exit_timesteps:
                features = spike_sum / t
                logits = self.exit_branches[str(t)](features)

                # Check confidence for samples that haven't exited yet
                probs = torch.softmax(logits, dim=1)
                confidence = probs.max(dim=1).values

                # Exit samples that are confident enough
                should_exit = (confidence >= confidence_threshold) & (~exited)
                final_logits[should_exit] = logits[should_exit]
                exit_times[should_exit] = t
                exited = exited | should_exit

                # If all samples have exited, stop
                if exited.all():
                    break

        # For samples that never exceeded threshold, use final output
        if not exited.all():
            remaining = ~exited
            final_logits[remaining] = logits[remaining]  # Use last computed logits

        return final_logits, exit_times

    def _nearest_exit(self, timestep: int) -> int:
        """Find the nearest exit point >= the given timestep."""
        for t in self.exit_timesteps:
            if t >= timestep:
                return t
        return self.exit_timesteps[-1]

    def get_exit_timesteps(self) -> list[int]:
        """Return the list of exit timesteps."""
        return self.exit_timesteps
