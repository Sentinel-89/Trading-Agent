# ============================================================
# rl/ppo_trainer_kl.py
#
# PPO update helper (with KL + clipfrac diagnostics)
#
# Notes:
# - Compatible with discrete (Categorical) and continuous policies
# - dist.log_prob(actions) may return (B,) or (B, action_dim)
# - Optional KL early-stop for stability
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================
# Helpers
# ============================

def _reduce_logprob(logp: torch.Tensor) -> torch.Tensor:
    """Ensure log_prob is shape (B,)."""
    if logp.ndim == 1:
        return logp
    if logp.ndim == 2:
        return logp.sum(dim=-1)
    raise ValueError(f"Unsupported log_prob shape: {tuple(logp.shape)}")


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """1
    Shape-safe: flattens to (B,) to avoid accidental broadcasting.
    """
    y_true = y_true.detach().view(-1)
    y_pred = y_pred.detach().view(-1)
    var_y = torch.var(y_true, unbiased=False)
    if var_y.item() < 1e-12:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / (var_y + 1e-12))


def ppo_update(
    policy,
    optimizer,
    observations: torch.Tensor,
    action_masks: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    num_epochs: int,
    batch_size: int,
    max_grad_norm: float,
    *,
    target_kl: Optional[float] = None,
    kl_early_stop: bool = True,
    kl_cutoff_multiplier: float = 1.5,
) -> Dict[str, Any]:
    """
    PPO update with KL/clipfrac diagnostics and optional KL early-stop.

    Approx KL is computed as mean(old_logp - new_logp).

    Early stopping:
      if target_kl is set and approx_kl exceeds target_kl*multiplier,
      the update stops early for stability.
    """
    assert torch.isfinite(advantages).all(), "Advantages contain NaN or Inf"

    # ============================
    # Advantage handling (parallel-safe normalization)
    # ============================
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ============================
    # Prepare mini-batch loader
    # ============================
    dataset = TensorDataset(
        observations,
        action_masks,
        actions,
        old_log_probs,
        advantages,
        returns,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ============================
    # Accumulators for diagnostics
    # ============================
    policy_losses = []
    value_losses = []
    entropies = []
    kls = []
    clipfracs = []

    early_stopped = False

    # ============================
    # PPO optimization epochs
    # ============================
    for _ in range(num_epochs):
        epoch_kls = []
        for obs_b, mask_b, act_b, old_logp_b, adv_b, ret_b in loader:
            dist, value = policy.get_dist_and_value(obs_b, mask_b)

            new_logp = _reduce_logprob(dist.log_prob(act_b))
            old_logp_b = _reduce_logprob(old_logp_b)

            ratio = torch.exp(new_logp - old_logp_b)

            unclipped = ratio * adv_b
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            value_loss = F.mse_loss(value.squeeze(-1), ret_b.squeeze(-1))

            ent = dist.entropy()
            # For continuous actions, entropy may be (B, action_dim); sum across dims per sample.
            entropy = ent.sum(dim=-1).mean() if ent.ndim == 2 else ent.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl = torch.mean(old_logp_b - new_logp)
                clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float())

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            kls.append(approx_kl.item())
            epoch_kls.append(approx_kl.item())
            clipfracs.append(clipfrac.item())

        # KL early-stop check at epoch boundary for smoother behavior
        if target_kl is not None and kl_early_stop and len(epoch_kls) > 0:
            mean_kl = float(sum(epoch_kls) / len(epoch_kls))
            if mean_kl > float(target_kl) * float(kl_cutoff_multiplier):
                early_stopped = True
                break

    # ============================
    # Return averaged diagnostics
    # ============================
    # explained variance (value function fit)
    # Recompute values for whole batch (cheap compared to full rollouts)
    with torch.no_grad():
        dist_all, v_all = policy.get_dist_and_value(observations, action_masks)
        ev = explained_variance(v_all.squeeze(-1), returns.squeeze(-1))

    def _avg(xs):
        return float(sum(xs) / max(1, len(xs)))

    out: Dict[str, Any] = {
        "policy_loss": _avg(policy_losses),
        "value_loss": _avg(value_losses),
        "entropy": _avg(entropies),
        "approx_kl": _avg(kls),
        "clipfrac": _avg(clipfracs),
        "explained_var": ev,
        "early_stopped": bool(early_stopped),
    }
    return out
