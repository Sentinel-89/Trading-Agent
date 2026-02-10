# Neural PPO Network only!  (no storage of state transitions, no PPO update logic -> see other files)

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOActorCritic(nn.Module):
    """
    PPO Actor–Critic network operating on full observation vectors.

    Design assumptions:
    - Input is a 69-dim observation vector (full state)
    - Discrete action space: {Hold, Buy, Sell}
    - Shared backbone with separate policy and value heads
    """

    def __init__(self, obs_dim: int = 69, hidden_dim: int = 128):
        super().__init__()

        # ------------------------------------------------------------------
        # Shared feature extractor
        #
        # Purpose:
        # - Mix and transform the full observation representation
        # - Provide a common representation for actor and critic
        #
        # Notes:
        # - Small MLP by design
        # - ReLU chosen for PPO stability
        # ------------------------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Policy head (actor)
        #
        # Outputs unnormalized logits for a categorical distribution
        # over discrete actions:
        #   0 = HOLD
        #   1 = BUY
        #   2 = SELL
        # ------------------------------------------------------------------
        self.policy_head = nn.Linear(hidden_dim, 3)

        # ------------------------------------------------------------------
        # Value head (critic)
        #
        # Outputs a scalar state-value estimate V(s)
        # ------------------------------------------------------------------
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        Forward pass through backbone and both heads.

        Args:
            obs: Tensor of shape (batch_size, obs_dim)

        Returns:
            logits: Action logits for categorical policy
            value:  State value estimate (batch_size,)
        """
        features = self.backbone(obs)

        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return logits, value

    def _masked_dist(self, logits: torch.Tensor, action_mask: torch.Tensor | None):
        """
        Create a categorical distribution.
        If action_mask is None, behave like standard PPO (no masking but penalty for invalid ops).
        """
        if action_mask is None:
            return Categorical(logits=logits)

        mask = action_mask.to(torch.bool)
        masked_logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=masked_logits)

    def get_dist_and_value(self, obs: torch.Tensor, action_mask: torch.Tensor| None = None):
        """
        Get masked policy distribution and value without sampling.

        Useful for evaluation or custom action selection.
        """
        logits, value = self.forward(obs)
        dist = self._masked_dist(logits, action_mask)
        return dist, value

    def act(self, obs: torch.Tensor, action_mask=None, deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = self._masked_dist(logits, action_mask)

        if deterministic:
            if action_mask is None:
                action = torch.argmax(logits, dim=-1)
            else:
                masked_logits = logits.masked_fill(~action_mask.to(torch.bool), -1e9)
                action = torch.argmax(masked_logits, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def act_batch(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        """
        Sample an action from the current masked policy.

        Used during rollout collection.

        Args:
            obs: Tensor of shape (batch_size, obs_dim)
            action_mask: Tensor of shape (batch_size, num_actions), with 1 for valid and 0 for invalid actions

        Returns:
            action:   Sampled discrete action
            log_prob: Log-probability of sampled action
            value:    State value estimate
        """
        logits, value = self.forward(obs)

        # Categorical distribution for discrete PPO with masking
        dist = self._masked_dist(logits, action_mask)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_mask=None):
        logits, value = self.forward(obs)
        dist = self._masked_dist(logits, action_mask)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, value
