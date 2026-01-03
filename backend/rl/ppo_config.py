# backend/rl/ppo_config.py

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """
    Central configuration object for PPO + GAE.

    This file intentionally owns *all* RL hyperparameters.
    No magic numbers should appear inside trainers, buffers, or GAE code.
    """

    # Discounting
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO clipping
    clip_eps: float = 0.2

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # PPO update structure
    num_epochs: int = 10
    batch_size: int = 64
