import torch


class RolloutBuffer:
    """
    PPO Rollout Buffer (Parallel-Environment, v5)

    It acts as a temporary container between:
      - Environment rollouts
      - PPO optimization (policy + value updates)

    Responsibilities:
      - Preserve temporal order
      - Preserve episode boundaries (done flags)
      - Provide stacked tensors for PPO loss computation

    Non-responsibilities:
      - No advantage computation
      - No normalization
      - No model updates
    """

    def __init__(self):
        """
        Initialize empty storage lists.
        Data is appended step-by-step during environment interaction.
        """

        # Latent observations produced by the frozen GRU encoder 
        self.observations = []

        # Actions sampled from the policy */
        self.actions = []

        self.action_masks = []          # mask actually used by the policy
        self.true_action_masks = [] 

        # Log-probabilities of sampled actions (for PPO ratio) 
        self.log_probs = []

        # Value function estimates at each step 
        self.values = []

        # Scalar rewards returned by the environment 
        self.rewards = []

        # Episode termination flags 
        self.dones = []

    def clear(self):
        """
        Clear all stored rollout data.
        Called after each PPO update cycle.
        """

        self.observations.clear()
        self.action_masks.clear()
        self.true_action_masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def add(
        self,
        observations: torch.Tensor,
        action: torch.Tensor,
        action_masks: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        true_action_masks: torch.Tensor | None = None,

    ):
        """
        Add one environment step to the rollout buffer.

        Parameters:
        - latent   : GRU latent state used as policy input (shape: [latent_dim])
        - action   : Action sampled from policy (scalar tensor)
        - log_prob : Log-probability of the sampled action
        - value    : Value function estimate for the current state
        - reward   : Environment reward (float)
        - done     : Episode termination flag
        """

        # Store detached tensors to avoid backprop through rollout collection
        self.observations.append(observations.detach().to(dtype=torch.float32))
        self.action_masks.append(action_masks.detach().to(dtype=torch.float32))
        if true_action_masks is not None:
            self.true_action_masks.append(true_action_masks.detach().to(dtype=torch.float32))
        self.actions.append(action.detach())  # keep integer dtype
        self.log_probs.append(log_prob.detach().to(dtype=torch.float32))
        self.values.append(value.detach().to(dtype=torch.float32))
        self.rewards.append(reward.detach().to(dtype=torch.float32))
        self.dones.append(done.detach().to(dtype=torch.float32))

    def size(self) -> int:
        """
        Return the number of stored timesteps in the buffer.
        """

        return len(self.rewards)

    def get_tensors(self, flatten: bool = True):


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
