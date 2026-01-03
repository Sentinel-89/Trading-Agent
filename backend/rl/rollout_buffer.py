import torch


class RolloutBuffer:
    """
    PPO Rollout Buffer (Single-Environment, v1)

    This buffer stores a fixed-length sequence of on-policy interaction data
    collected from exactly one environment (TradingEnv v1).

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
        self.latents = []

        # Actions sampled from the policy */
        self.actions = []

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

        self.latents.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def add(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
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
        self.latents.append(latent.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())

        # Rewards and done flags are stored as raw Python scalars
        self.rewards.append(reward)
        self.dones.append(done)

    def size(self) -> int:
        """
        Return the number of stored timesteps in the buffer.
        """

        return len(self.rewards)

    def get_tensors(self):
        """
        Stack stored rollout data into tensors.

        Returns:
        A dictionary containing:
          - latents   : Tensor[T, latent_dim]
          - actions   : Tensor[T]
          - log_probs : Tensor[T]
          - values    : Tensor[T]
          - rewards   : Tensor[T]
          - dones     : Tensor[T]
        """

        # Stack tensors along time dimension */
        latents = torch.stack(self.latents)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        # Convert rewards and dones to tensors */
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        return {
            "latents": latents,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
        }
