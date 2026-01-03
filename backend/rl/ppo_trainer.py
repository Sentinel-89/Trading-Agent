import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical


def ppo_update(
    policy,
    optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    num_epochs: int,
    batch_size: int,
):
    """
    Perform PPO updates using rollout data from a single environment.

    Returns diagnostic statistics for smoke testing.
    """

    # ============================================================
    # Advantage normalization (recommended for PPO stability)
    # ============================================================

    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )

    # ============================================================
    # Prepare mini-batch loader
    # ============================================================

    dataset = TensorDataset(
        observations,
        actions,
        old_log_probs,
        advantages,
        returns,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # ============================================================
    # Accumulators for diagnostics
    # ============================================================

    policy_losses = []
    value_losses = []
    entropies = []

    # ============================================================
    # PPO optimization epochs
    # ============================================================

    for _ in range(num_epochs):

        for obs_b, act_b, old_logp_b, adv_b, ret_b in loader:

            logits, value = policy(obs_b)

            dist = Categorical(logits=logits)

            new_logp = dist.log_prob(act_b)

            ratio = torch.exp(new_logp - old_logp_b)

            unclipped = ratio * adv_b
            clipped = torch.clamp(
                ratio,
                1.0 - clip_eps,
                1.0 + clip_eps,
            ) * adv_b

            policy_loss = -torch.mean(
                torch.min(unclipped, clipped)
            )

            value_loss = F.mse_loss(
                value.squeeze(-1),
                ret_b,
            )

            entropy = torch.mean(dist.entropy())

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------------------------------------------
            # Collect diagnostics
            # ----------------------------------------------------

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

    # ============================================================
    # Return averaged diagnostics (smoke test contract)
    # ============================================================
    # That means you collect:
    # one policy_loss
    # one value_loss
    # one entropy
    # for every minibatch update across all epochs.
    # i.e. they are averaged over all minibatch gradient steps across all PPO epochs for a single rollout

    return {
        "policy_loss": float(sum(policy_losses) / len(policy_losses)),
        "value_loss": float(sum(value_losses) / len(value_losses)),
        "entropy": float(sum(entropies) / len(entropies)),
    }
