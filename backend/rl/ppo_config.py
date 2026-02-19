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

    # Optional KL control (used by ppo_trainer_kl.py)
    # - If target_kl is None, KL early-stop is disabled
    target_kl: float | None = None
    kl_cutoff_multiplier: float = 1.5

    # Loss coefficients
    value_coef: float = 0.5
    # Entropy bonus
    # NOTE: continuous policies often need a smaller entropy_coef than discrete
    entropy_coef: float = 0.01

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # PPO update structure
    num_epochs: int = 10
    batch_size: int = 64


# ----------------------------
# Recommended presets
# ----------------------------

def make_discrete_config() -> PPOConfig:
    """Defaults are tuned for discrete policies (buy/hold/sell)."""
    return PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        # Enable KL early-stop to avoid occasional catastrophic PPO updates
        target_kl=0.01,
        kl_cutoff_multiplier=1.5,
        value_coef=0.5,
        entropy_coef=0.01,
        # Slightly lower LR is more stable for discrete action markets
        learning_rate=2e-4,
        max_grad_norm=0.5,
        # Fewer epochs reduces overfitting to one rollout (helps KL stability)
        num_epochs=6,
        # Larger batch reduces gradient noise (64 can be very spiky)
        batch_size=256,
    )


def make_continuous_config() -> PPOConfig:
    """More conservative/stable defaults for continuous portfolio-weight policies."""
    return PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        # Enable KL early-stop to prevent catastrophic updates
        target_kl=0.02,
        kl_cutoff_multiplier=1.5,
        value_coef=0.5,
        # Continuous often needs smaller entropy bonus
        entropy_coef=0.001,
        # Lower LR is usually more stable for continuous control
        learning_rate=1e-4,
        max_grad_norm=0.5,
        # Fewer epochs reduces overfitting to one rollout and helps KL stability
        num_epochs=5,
        # Slightly larger batch is usually betterfor  8*252=2016 steps/update
        batch_size=256,
    )
