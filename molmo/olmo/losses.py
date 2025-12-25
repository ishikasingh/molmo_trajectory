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
    either the velocity field or the clean target (x0) directly.
    
    Supports two prediction types:
    - "velocity": Model predicts the velocity field v = noise - target
    - "x0": Model predicts x0 (clean target) directly
    """

    def __init__(self, prediction_type: str = "velocity"):
        """
        Args:
            prediction_type: "velocity" or "x0"
                - "velocity": Model predicts velocity field (noise - target)
                - "x0": Model predicts clean target x0 directly
        """
        super().__init__()
        if prediction_type not in ("velocity", "x0"):
            raise ValueError(f"prediction_type must be 'velocity' or 'x0', got {prediction_type}")
        self.prediction_type = prediction_type

    def forward(
        self,
        model_output: torch.Tensor,
        trajectory_target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        action_dim_mask: torch.Tensor = None,
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
        
        For velocity prediction:
            Model predicts: v = noise - target
            Loss: MSE(v_pred, noise - target)
        
        For x0 prediction:
            Model predicts: x0 = target (clean data)
            Loss: MSE(x0_pred, target)
            During inference, velocity is computed as: v = (x_t - x0_pred) / t
        
        Inference uses NEGATIVE time steps (t: 1 → 0):
            dt = -1.0 / num_steps
            x_t = x_t + dt * v_t  (moves backward in time)

        Args:
            model_output: Model prediction, shape (batch_size, action_horizon, max_action_dim)
                - For velocity prediction: predicted velocity
                - For x0 prediction: predicted clean target x0
                May be padded to max_action_dim in multi-expert mode
            trajectory_target: Ground truth trajectory (X_1), shape (batch_size, action_horizon, max_action_dim)
                May be padded to max_action_dim in multi-expert mode
            t: Time samples in [0, 1], shape (batch_size,)
            noise: Noise samples (X_0), shape (batch_size, action_horizon, max_action_dim)
                May be padded to max_action_dim in multi-expert mode
            action_dim_mask: Optional, shape (batch_size,) - valid action_dim for each sample
                If provided, only computes loss for valid dimensions (for multi-expert mode)

        Returns:
            Scalar loss value
        """
        if self.prediction_type == "velocity":
            # For velocity prediction, the true velocity is noise - target
            # X_t = t * noise + (1-t) * target
            # dX_t/dt = noise - target (constant velocity field for linear path)
            true_velocity = noise - trajectory_target
            loss = F.mse_loss(model_output, true_velocity, reduction="none")
        else:  # x0 prediction
            # For x0 prediction, we directly regress to the clean target
            # The model predicts x0 (the denoised/clean trajectory)
            loss = F.mse_loss(model_output, trajectory_target, reduction="none")

        # Handle multi-expert mode with different action dimensions per sample
        if action_dim_mask is not None:
            # Create mask: (batch_size, action_horizon, max_action_dim)
            batch_size, action_horizon, max_action_dim = loss.shape
            mask = torch.zeros_like(loss)
            for i in range(batch_size):
                valid_dim = action_dim_mask[i].item()
                mask[i, :, :valid_dim] = 1.0
            
            # Apply mask and compute mean over valid dimensions only
            masked_loss = loss * mask
            mean_loss = masked_loss.sum() / mask.sum()
        else:
            mean_loss = loss.mean()
        
        # If loss is extremely large (> 1e6), something is wrong
        if mean_loss > 1e6:
            import warnings
            warnings.warn(
                f"Flow matching loss is extremely large ({mean_loss.item():.2e}). "
                f"This usually indicates:\n"
                f"  1. Very large predictions (check model initialization)\n"
                f"  2. Unnormalized trajectory data (should be in [-10, 10] range ideally)\n"
                f"  3. Learning rate too high\n"
                f"Clipping loss to 1e6 to prevent inf, but you should investigate the root cause."
            )
            # mean_loss = torch.clamp(mean_loss, max=1e6)
        
        return mean_loss

