# Neural PPO Network only!  (no storage of state transitions, no PPO update logic -> for that see ppo_trainer.py!)

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOActorCritic(nn.Module):
    """
    PPO Actor–Critic network operating on a frozen GRU latent state.

    Design assumptions:
    - Input is a 64-dim latent vector produced by a frozen encoder
    - Discrete action space: {Hold, Buy, Sell}
    - Shared backbone with separate policy and value heads
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        # ------------------------------------------------------------------
        # Shared feature extractor
        #
        # Purpose:
        # - Mix and transform the GRU latent representation
        # - Provide a common representation for actor and critic
        #
        # Notes:
        # - Small MLP by design (latent already encodes structure)
        # - ReLU chosen for PPO stability
        # ------------------------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
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

    def forward(self, latent: torch.Tensor):
        """
        Forward pass through backbone and both heads.

        Args:
            latent: Tensor of shape (batch_size, latent_dim)

        Returns:
            logits: Action logits for categorical policy
            value:  State value estimate (batch_size,)
        """
        features = self.backbone(latent)

        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return logits, value

    def act(self, latent: torch.Tensor):
        """
        Sample an action from the current policy.

        Used during rollout collection.

        Args:
            latent: Tensor of shape (batch_size, latent_dim)

        Returns:
            action:   Sampled discrete action
            log_prob: Log-probability of sampled action
            value:    State value estimate
        """
        logits, value = self.forward(latent)

        # Categorical distribution for discrete PPO
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, latent: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions taken during rollout.

        Used during PPO update to compute:
        - log probabilities
        - entropy bonus
        - value estimates

        Args:
            latent:  Tensor of shape (batch_size, latent_dim)
            actions: Tensor of discrete actions taken

        Returns:
            log_probs: Log-probabilities of provided actions
            entropy:   Policy entropy (for regularization)
            value:     State value estimate
        """
        logits, value = self.forward(latent)

        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, value
