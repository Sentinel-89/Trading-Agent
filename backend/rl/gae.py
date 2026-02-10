import torch
from typing import Optional


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    num_envs: int,
    last_values: Optional[torch.Tensor] = None,
):
    """
    Correct GAE for vectorized (async) environments.

    This implementation computes GAE *independently per environment*
    and is safe for multi-asset / multi-env PPO training.

    Expected inputs:
      - rewards: shape (T, N) or (T*N,)
      - values:  shape (T, N) or (T*N,)
      - dones:   shape (T, N) or (T*N,)
      - num_envs: number of parallel environments

    Advantages are NOT normalized here.
    """

    device = values.device

    rewards = rewards.to(device)
    values = values.to(device)
    dones = dones.to(device).float()

    # If flattened, reshape to (T, N)
    if rewards.dim() == 1:
        T = rewards.size(0) // num_envs
        rewards = rewards.view(T, num_envs)
        values = values.view(T, num_envs)
        dones = dones.view(T, num_envs)
    else:
        T = rewards.size(0)

    advantages = torch.zeros_like(rewards, device=device)
    returns = torch.zeros_like(rewards, device=device)

    # NOTE:
    # `dones[t]` is assumed to indicate whether the transition taken at step t
    # ended the episode (i.e., whether the *next* state is terminal).
    # Therefore, bootstrapping for step t must be gated by `dones[t]`.
    if last_values is None:
        last_values = torch.zeros(num_envs, device=device, dtype=values.dtype)
    else:
        last_values = last_values.to(device)

    # Compute GAE independently per environment
    for env in range(num_envs):
        gae = 0.0
        for t in reversed(range(T)):
            # Bootstrap value from the next state unless the episode ended at step t.
            if t == T - 1:
                next_value = last_values[env]
            else:
                next_value = values[t + 1, env]

            next_non_terminal = 1.0 - dones[t, env]

            delta = rewards[t, env] + gamma * next_value * next_non_terminal - values[t, env]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t, env] = gae

        returns[:, env] = advantages[:, env] + values[:, env]

    # Flatten back to (T*N,)
    advantages = advantages.view(-1)
    returns = returns.view(-1)

    return advantages, returns
