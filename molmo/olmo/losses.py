"""Flow matching loss and trajectory prediction head for continuous trajectory prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelConfig


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0
) -> torch.Tensor:
    """
    Computes sine-cosine positional embedding vectors for scalar positions.
    
    Args:
        pos: (batch_size,) tensor of positions
        embedding_dim: dimension of embedding (must be even)
        min_period: minimum period for sinusoid
        max_period: maximum period for sinusoid
        
    Returns:
        (batch_size, embedding_dim) tensor of position embeddings
    """
    assert (pos <= 1).all(), "Positions must be less than 1"
    assert (pos >= 0).all(), "Positions must be greater than 0"
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum('i,j->ij', pos, 1.0 / period * 2 * math.pi)
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class FlowMatchingTrajectoryLoss(nn.Module):
    """
    Flow matching loss for continuous trajectory prediction.
    
    Uses affine (Conditional OTP) flow matching to train a model to predict
    the velocity field of trajectories interpolating between noise and ground truth.
    """

    def __init__(self, velocity_weighting: str = "none"):
        """
        Args:
            velocity_weighting: How to weight the loss. Options:
                - "none": uniform weighting
                - "t": weight by time (weight = t) to emphasize later steps
                - "inverse_t": weight by 1-t to emphasize earlier steps
        """
        super().__init__()
        assert velocity_weighting in ["none", "t", "inverse_t"]
        self.velocity_weighting = velocity_weighting

    def forward(
        self,
        predicted_velocity: torch.Tensor,
        trajectory_target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        We use the DIFFUSION CONVENTION (OpenPI style), which is backwards from standard flow matching papers:
        - t=1: noise (starting point for inference)
        - t=0: target (ending point for inference)
        
        Path definition (Conditional OTP):
            X_t = t * X_0 + (1-t) * X_1
            where X_0 = noise, X_1 = target
            
            At t=0: X_0 = target (data)
            At t=1: X_1 = noise (noise)
        
        Velocity (what we train to predict):
            dX_t/dt = X_0 - X_1 = noise - target
        
        Inference uses NEGATIVE time steps (t: 1 → 0):
            dt = -1.0 / num_steps
            x_t = x_t + dt * v_t  (moves backward in time)
        
        Alternative (standard flow matching papers):
            Would use X_t = (1-t) * noise + t * target with dX_t/dt = target - noise
            and integrate forward (t: 0 → 1). This is mathematically equivalent but
            uses opposite time direction. We follow OpenPI/diffusion convention for compatibility.

        Args:
            predicted_velocity: Model prediction of velocity, shape (batch_size, action_horizon, action_dim)
            trajectory_target: Ground truth trajectory (X_1), shape (batch_size, action_horizon, action_dim)
            t: Time samples in [0, 1], shape (batch_size,)
            noise: Noise samples (X_0), shape (batch_size, action_horizon, action_dim)

        Returns:
            Scalar loss value
        """
        # For Conditional OTP with diffusion convention, the true velocity is simply X_0 - X_1
        # which is noise - target. This is because:
        # X_t = t * noise + (1-t) * target
        # dX_t/dt = noise - target (constant velocity field for linear path)
        true_velocity = noise - trajectory_target

        # MSE loss between predicted and true velocity
        velocity_loss = F.mse_loss(predicted_velocity, true_velocity, reduction="none")

        # Apply time weighting if specified
        if self.velocity_weighting == "t":
            # Expand t to match loss dimensions
            weights = t.view(-1, *([1] * (velocity_loss.dim() - 1)))
            velocity_loss = velocity_loss * weights
        elif self.velocity_weighting == "inverse_t":
            # Weight earlier steps more heavily
            weights = (1.0 - t).view(-1, *([1] * (velocity_loss.dim() - 1)))
            velocity_loss = velocity_loss * weights

        # Check if loss is reasonable before returning
        mean_loss = velocity_loss.mean()
        
        # If loss is extremely large (> 1e6), something is wrong
        if mean_loss > 1e6:
            import warnings
            warnings.warn(
                f"Flow matching loss is extremely large ({mean_loss.item():.2e}). "
                f"This usually indicates:\n"
                f"  1. Very large velocity predictions (check model initialization)\n"
                f"  2. Unnormalized trajectory data (should be in [-10, 10] range ideally)\n"
                f"  3. Learning rate too high\n"
                f"Clipping loss to 1e6 to prevent inf, but you should investigate the root cause."
            )
            # mean_loss = torch.clamp(mean_loss, max=1e6)
        
        return mean_loss

