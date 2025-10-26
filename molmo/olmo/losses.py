"""Flow matching loss and trajectory prediction head for continuous trajectory prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


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

        # Return mean loss
        return velocity_loss.mean()


class FlowMatchingHead(nn.Module):
    """
    Flow matching head that processes noisy action tokens through the main transformer.
    
    This module:
    1. Embeds noisy actions into the model's embedding space
    2. Embeds timestep information
    3. Combines action and timestep embeddings
    4. These are concatenated with text/image tokens and passed through the transformer
    5. Outputs velocity predictions from the action token representations
    
    The key difference from cross-attention: action tokens are processed THROUGH the 
    transformer alongside text/image tokens, not in a separate module.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        action_horizon: int,
        use_adarms: bool = False,
    ):
        """
        Args:
            d_model: Model dimension (same as LLM)
            action_dim: Dimension of each action vector
            action_horizon: Number of action steps to predict
            use_adarms: Deprecated - kept for backward compatibility but not used
        """
        super().__init__()
        
        self.d_model = d_model
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        # Note: use_adarms is deprecated. We only support MLP-style mixing which works
        # with standard transformers without requiring conditional layer norms.
        
        # Project actions to model dimension
        self.action_in_proj = nn.Linear(action_dim, d_model)
        
        # MLP-style timestep conditioning (Pi0)
        # Concatenate action + time embeddings and mix with MLP
        self.action_time_mlp_in = nn.Linear(2 * d_model, d_model)
        self.action_time_mlp_out = nn.Linear(d_model, d_model)
        
        # Project action outputs to velocity
        self.action_out_proj = nn.Linear(d_model, action_dim)
    
    def embed_actions_with_time(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed noisy actions and timestep information.
        
        Args:
            noisy_actions: (batch_size, action_horizon, action_dim)
            timestep: (batch_size,) scalar time values in [0, 1]
            
        Returns:
            action_tokens: (batch_size, action_horizon, d_model)
            
        Note: We use MLP-style mixing (Pi0) which works with existing transformers.
              AdaRMS conditioning (Pi0.5) would require conditional layer norms.
        """
        # Embed actions
        action_tokens = self.action_in_proj(noisy_actions)  # (B, H, D)
        
        # Embed timestep using sinusoidal encoding
        time_emb = posemb_sincos(timestep, self.d_model)  # (B, D)
        
        # Mix timestep with action embeddings using MLP
        # Expand time embedding to match action sequence length
        time_tokens = time_emb.unsqueeze(1).expand(-1, self.action_horizon, -1)  # (B, H, D)
        
        # Concatenate and process through MLPs
        action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)  # (B, H, 2D)
        action_time_tokens = F.silu(self.action_time_mlp_in(action_time_tokens))
        action_expert_tokens = F.silu(self.action_time_mlp_out(action_time_tokens))
        
        return action_expert_tokens
    
    def predict_velocity(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project action hidden states to velocity predictions.
        
        Args:
            action_hidden_states: (batch_size, action_horizon, d_model)
            
        Returns:
            velocity: (batch_size, action_horizon, action_dim)
        """
        return self.action_out_proj(action_hidden_states)


def make_flow_matching_attention_mask(
    prefix_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    prefix_ar_mask: Optional[torch.Tensor] = None,
    suffix_ar_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create attention mask for flow matching that prevents text/image tokens 
    from attending to action tokens.
    
    Based on openpi's make_attn_mask function. The ar_mask controls which tokens 
    can attend to which:
    - ar_mask[i] = False: token i is in bidirectional block with previous tokens
    - ar_mask[i] = True: token i starts new causal block, previous tokens can't attend to it
    
    For flow matching:
    - Prefix (text/images): all False (fully bidirectional)
    - Suffix (actions): [True, False, False, ...] 
      * First action token True -> prevents prefix from attending
      * Rest False -> allows causal attention within actions
    
    Args:
        prefix_mask: (batch_size, prefix_len) - True for valid prefix tokens
        suffix_mask: (batch_size, suffix_len) - True for valid suffix tokens
        prefix_ar_mask: (prefix_len,) - ar mask for prefix (typically all False)
        suffix_ar_mask: (suffix_len,) - ar mask for suffix (typically [True, False, ...])
        
    Returns:
        attention_mask: (batch_size, total_len, total_len) boolean mask
            True = can attend, False = cannot attend
    """
    batch_size = prefix_mask.shape[0]
    prefix_len = prefix_mask.shape[1]
    suffix_len = suffix_mask.shape[1]
    total_len = prefix_len + suffix_len
    device = prefix_mask.device
    
    # Concatenate masks
    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)  # (B, total_len)
    
    # Create ar_mask if not provided
    if prefix_ar_mask is None:
        prefix_ar_mask = torch.zeros(prefix_len, dtype=torch.bool, device=device)
    if suffix_ar_mask is None:
        # Default: first action token prevents prefix attention, rest allow causal
        suffix_ar_mask = torch.zeros(suffix_len, dtype=torch.bool, device=device)
        suffix_ar_mask[0] = True
    
    # Concatenate ar_masks (these are 1D, broadcast to batch)
    ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)  # (total_len,)
    ar_mask = ar_mask.unsqueeze(0).expand(batch_size, -1)  # (B, total_len)
    
    # Compute cumulative sum for causal masking
    cumsum = torch.cumsum(ar_mask.float(), dim=1)  # (B, total_len)
    
    # Token i can attend to token j if cumsum[j] <= cumsum[i]
    # This creates the block structure
    attn_mask = cumsum.unsqueeze(1) >= cumsum.unsqueeze(2)  # (B, total_len, total_len)
    
    # Also respect input_mask (padding)
    valid_mask = input_mask.unsqueeze(1) & input_mask.unsqueeze(2)  # (B, total_len, total_len)
    
    return attn_mask & valid_mask
