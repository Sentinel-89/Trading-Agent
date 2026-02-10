# ============================================================
# rl/ppo_actor_critic_continuous.py
#
# PPO Actor–Critic (continuous logits)
#
# Continuous-action network for portfolio allocation:
# - Actor outputs a diagonal Gaussian over *logits* (unbounded)
# - Env maps logits -> weights via softmax (and can enforce top-k)
# - Critic outputs scalar V(s)
#
# NOTE:
# - `action_mask` is accepted for API compatibility but ignored
#   (continuous logits are always valid; masking is handled in env)
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class DiagGaussian:
    """Minimal diagonal Gaussian wrapper.

    Matches the discrete PPO interface expectations:
    - sample / log_prob / entropy
    - log_prob and entropy return shape (B,) by summing over dims
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = torch.exp(self.log_std)
        self.base = Normal(self.mean, self.std)

    def sample(self) -> torch.Tensor:
        return self.base.sample()

    def rsample(self) -> torch.Tensor:
        return self.base.rsample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Joint log-probability under a diagonal Gaussian.

        Supports:
          - batched actions: (B, A)  -> returns (B,)
          - single-step action: (A,) -> returns scalar ()
          - scalar action with batch: (B,) when A==1 -> returns (B,)
        """
        # Unbatched single action vector (A,)
        if action.dim() == 1 and self.mean.dim() == 1:
            return self.base.log_prob(action).sum()

        # Batched scalar actions for A==1: (B,) -> (B,1)
        if action.dim() == 1 and self.mean.dim() == 2:
            action = action.unsqueeze(-1)

        # Batched actions: (B, A)
        return self.base.log_prob(action).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Joint entropy.

        Returns:
          - (B,) for batched distributions
          - scalar () for unbatched distributions
        """
        ent = self.base.entropy()
        # Unbatched: (A,) -> scalar
        if ent.dim() == 1 and self.mean.dim() == 1:
            return ent.sum()
        return ent.sum(dim=-1)


class PPOActorCriticContinuous(nn.Module):
    """PPO Actor–Critic network operating on full observation vectors.

    Design assumptions:
    - Input is a flat observation vector (full state)
    - Continuous action is a vector of logits (unbounded)
      (env converts logits -> long-only weights)
    - Shared backbone with separate actor and critic heads
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ------------------------------------------------------------------
        # Shared feature extractor
        #
        # Purpose:
        # - Mix and transform the full observation representation
        # - Provide a common representation for actor and critic
        #
        # Notes:
        # - Small MLP by design (same style as discrete PPO)
        # ------------------------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Policy heads (actor)
        #
        # Outputs parameters of a diagonal Gaussian over logits:
        #   action ~ Normal(mu, std)
        #
        # The environment is responsible for mapping:
        #   logits -> softmax -> weights
        # ------------------------------------------------------------------
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # ------------------------------------------------------------------
        # Value head (critic)
        #
        # Outputs a scalar state-value estimate V(s)
        # ------------------------------------------------------------------
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """Forward pass through backbone and both heads.

        Args:
            obs: Tensor of shape (batch_size, obs_dim)

        Returns:
            mu:    Mean logits for Gaussian policy (batch_size, action_dim)
            log_s: Log-std logits for Gaussian policy (batch_size, action_dim)
            value: State value estimate (batch_size,)
        """
        features = self.backbone(obs)

        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        value = self.value_head(features).squeeze(-1)

        return mu, log_std, value

    def get_dist_and_value(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        """Get policy distribution and value without sampling."""
        mu, log_std, value = self.forward(obs)
        dist = DiagGaussian(mu, log_std)
        return dist, value

    def act(self, obs: torch.Tensor, action_mask=None, deterministic: bool = False):
        """Single-step action selection (matches discrete API)."""
        dist, value = self.get_dist_and_value(obs, action_mask)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def act_batch(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None, deterministic: bool = False):
        """Sample actions for rollout collection.

        Args:
            obs: Tensor of shape (batch_size, obs_dim)
            action_mask: accepted for compatibility (ignored)

        Returns:
            action:   (batch_size, action_dim) float
            log_prob: (batch_size,) float
            value:    (batch_size,) float
        """
        dist, value = self.get_dist_and_value(obs, action_mask)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_mask=None):
        """Compute log_probs/entropy/value for PPO updates (mirrors discrete)."""
        dist, value = self.get_dist_and_value(obs, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value
