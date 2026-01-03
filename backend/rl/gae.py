import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    """
    Generalized Advantage Estimation (GAE), single-environment version.

    Computes advantage estimates and value targets (returns) from
    rollout data collected using the current policy.

    This implementation assumes:
      - rewards, values, dones are 1D tensors of equal length
      - values[t] corresponds to V(s_t)
      - dones[t] indicates whether the episode terminated at step t
    """

    # Number of timesteps in rollout */
    T = rewards.size(0)

    # Storage for advantages */
    advantages = torch.zeros(T, dtype=torch.float32)

    # Running advantage accumulator (A_{t+1}) */
    gae = 0.0

    # Iterate backwards through time */
    for t in reversed(range(T)):

        # If episode ended at t, there is no bootstrap value */
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        # Mask is zero if episode terminated at t */
        not_done = 1.0 - dones[t]

        # Temporal-Difference residual (delta_t) */
        delta = rewards[t] + gamma * next_value * not_done - values[t]

        # GAE recursion */
        gae = delta + gamma * gae_lambda * not_done * gae

        advantages[t] = gae

    # Returns are advantages plus baseline values */
    returns = advantages + values

    return advantages, returns
