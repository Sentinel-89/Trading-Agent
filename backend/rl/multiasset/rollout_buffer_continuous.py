# ============================================================
# rl/rollout_buffer_continuous.py
#
# PPO Rollout Buffer (Continuous Actions)
#
# Same responsibilities as the discrete rollout buffer, but:
# - actions are float tensors with shape (N, action_dim)
# - get_tensors(flatten=True) returns actions as (T×N, action_dim)
# ============================================================

import torch


class RolloutBufferContinuous:
    """
    PPO Rollout Buffer (Vectorized Environments, Continuous Actions)

    Stores on-policy rollout data collected from N parallel environments
    over T timesteps, then flattens (T, N) → (T×N) for PPO updates.

    Responsibilities:
      - Store batched rollout data
      - Preserve done masks for GAE
      - Provide flattened tensors for PPO

    Non-responsibilities:
      - No advantage computation
      - No normalization
      - No PPO optimization
    """

    def __init__(self):
        self.observations = []
        self.action_masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.observations.clear()
        self.action_masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def add_batch(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        """
        Add one timestep of data from N parallel environments.

        Shapes:
          observations : (N, obs_dim)
          action_masks : (N, action_dim)
          actions      : (N, action_dim)   <-- continuous vector actions
          log_probs    : (N,)
          values       : (N,)
          rewards      : (N,)
          dones        : (N,)
        """

        # Basic shape checks (fail fast)
        assert observations.dim() == 2, f"observations must be (N, D), got {tuple(observations.shape)}"
        assert action_masks.dim() == 2, f"action_masks must be (N, A), got {tuple(action_masks.shape)}"
        assert actions.dim() == 2, f"actions must be (N, A), got {tuple(actions.shape)}"

        N = observations.shape[0]
        assert action_masks.shape[0] == N, "action_masks batch size mismatch"
        assert actions.shape[0] == N, "actions batch size mismatch"

        # Force scalar vectors to shape (N,)
        log_probs = log_probs.view(-1)
        values = values.view(-1)
        rewards = rewards.view(-1)
        dones = dones.view(-1)

        assert log_probs.shape[0] == N, "log_probs batch size mismatch"
        assert values.shape[0] == N, "values batch size mismatch"
        assert rewards.shape[0] == N, "rewards batch size mismatch"
        assert dones.shape[0] == N, "dones batch size mismatch"

        self.observations.append(observations.detach().to(device="cpu", dtype=torch.float32))
        self.action_masks.append(action_masks.detach().to(device="cpu", dtype=torch.float32))
        self.actions.append(actions.detach().to(device="cpu", dtype=torch.float32))
        self.log_probs.append(log_probs.detach().to(device="cpu", dtype=torch.float32))
        self.values.append(values.detach().to(device="cpu", dtype=torch.float32))
        self.rewards.append(rewards.detach().to(device="cpu", dtype=torch.float32))
        self.dones.append(dones.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0))

    def get_tensors(self, flatten: bool = True):
        """
        Stack rollout data into tensors.

        If flatten=True:
          (T, N, ...) → (T×N, ...)

        Returns:
          dict of tensors ready for PPO update.
        """

        observations = torch.stack(self.observations)  # (T, N, D)
        action_masks = torch.stack(self.action_masks)  # (T, N, A)
        actions = torch.stack(self.actions)            # (T, N, A)
        log_probs = torch.stack(self.log_probs)        # (T, N)
        values = torch.stack(self.values)              # (T, N)
        rewards = torch.stack(self.rewards)            # (T, N)
        dones = torch.stack(self.dones)                # (T, N)

        if flatten:
            T, N, A = actions.shape

            observations = observations.reshape(T * N, -1)
            action_masks = action_masks.reshape(T * N, -1)
            actions = actions.reshape(T * N, A)
            log_probs = log_probs.reshape(T * N)
            values = values.reshape(T * N)
            rewards = rewards.reshape(T * N)
            dones = dones.reshape(T * N)

        return {
            "observations": observations,
            "action_masks": action_masks,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
        }
