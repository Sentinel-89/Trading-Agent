import torch



class RolloutBuffer:
    """
    PPO Rollout Buffer (Vectorized Environments)

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
        self.action_masks = []          # mask actually used by the policy
        self.true_action_masks = []     # always-correct env constraint mask (optional)
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.observations.clear()
        self.action_masks.clear()
        self.true_action_masks.clear()
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
        true_action_masks: torch.Tensor | None = None,
    ):
        """
        Add one timestep of data from N parallel environments.

        Shapes:
          observations : (N, obs_dim)
          action_masks : (N, action_dim)
          actions      : (N,)
          log_probs    : (N,)
          values       : (N,)
          rewards      : (N,)
          dones        : (N,)
          true_action_masks : (N, action_dim)  (optional; env constraint mask)
        """

        self.observations.append(observations.detach().to(dtype=torch.float32))
        self.action_masks.append(action_masks.detach().to(dtype=torch.float32))
        if true_action_masks is not None:
            self.true_action_masks.append(true_action_masks.detach().to(dtype=torch.float32))
        self.actions.append(actions.detach())  # keep integer dtype
        self.log_probs.append(log_probs.detach().to(dtype=torch.float32))
        self.values.append(values.detach().to(dtype=torch.float32))
        self.rewards.append(rewards.detach().to(dtype=torch.float32))
        self.dones.append(dones.detach().to(dtype=torch.float32))

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
        true_action_masks = None
        if len(self.true_action_masks) > 0:
            true_action_masks = torch.stack(self.true_action_masks)  # (T, N, A)
        actions = torch.stack(self.actions)            # (T, N)
        log_probs = torch.stack(self.log_probs)        # (T, N)
        values = torch.stack(self.values)               # (T, N)
        rewards = torch.stack(self.rewards)             # (T, N)
        dones = torch.stack(self.dones)                 # (T, N)

        if flatten:
            T, N = actions.shape

            observations = observations.reshape(T * N, -1)
            action_masks = action_masks.reshape(T * N, -1)
            if true_action_masks is not None:
                true_action_masks = true_action_masks.reshape(T * N, -1)
            actions = actions.reshape(T * N)
            log_probs = log_probs.reshape(T * N)
            values = values.reshape(T * N)
            rewards = rewards.reshape(T * N)
            dones = dones.reshape(T * N)

        out = {
            "observations": observations,
            "action_masks": action_masks,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
        }

        if true_action_masks is not None:
            out["true_action_masks"] = true_action_masks

        return out
