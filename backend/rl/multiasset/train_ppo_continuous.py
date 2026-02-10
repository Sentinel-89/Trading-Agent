# backend/rl/train_ppo_continuous.py

"""Phase-D PPO training - CONTINUOUS PORTFOLIO

Trains a PPO agent to output *continuous* portfolio allocations.

Core ideas:
- Multi-symbol portfolio environment (each env instance trades `num_assets` symbols).
- Continuous actions: the policy outputs logits; the env maps logits -> long-only weights via softmax.
- Optional top-k holdings enforced inside the environment.
- Vectorized rollout collection (AsyncVectorEnv).

Important slicing semantics (inside TradingEnvContinuous):
- `window_length` in TradingEnvContinuousConfig is a *warmup one
  It controls the earliest allowed start index for an episode., 30 in this case  
- `episode_length` controls how many tradable steps an episode lasts in training mode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from gymnasium.vector import AsyncVectorEnv

from backend.rl.gae import compute_gae
from backend.rl.multiasset.ppo_actor_critic_continuous import PPOActorCriticContinuous
from backend.rl.ppo_config import make_continuous_config
from backend.rl.ppo_trainer import ppo_update
from backend.rl.multiasset.rollout_buffer_continuous import RolloutBufferContinuous
from backend.rl.multiasset.trading_env_continuous import TradingEnvContinuous, TradingEnvContinuousConfig


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, torch (and cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# JSON utilities (safe logging)
# ============================================================

def _json_safe(x):
    """Convert numpy/torch values to JSON-serializable Python types."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist() if x.ndim > 0 else float(x.item())
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def write_log_json(path: str, payload: dict) -> None:
    """Atomic JSON write: write tmp then replace."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp_path, path)


# ============================================================
# Data loading
# ============================================================

def load_and_scale_universe(
    data_dir: str,
    scaler_path: str,
    feature_cols: List[str],
    date_start: str,
    date_end: str,
) -> Dict[str, pd.DataFrame]:
    """Load *_labeled.csv files, slice by date, preserve raw prices, and scale features.

    Returns:
        Dict[symbol, df] where df is aligned across all symbols (common date intersection)
        and uses a simple integer index for fast env access.

    Notes:
      - `Close_raw` (and `Open_raw` if present) are preserved for pricing/returns.
      - ONLY `feature_cols` are scaled.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    scaler = joblib.load(scaler_path)
    data_by_symbol: Dict[str, pd.DataFrame] = {}

    for fname in os.listdir(data_dir):
        if not fname.endswith("_labeled.csv"):
            continue

        symbol = fname.replace("_labeled.csv", "")
        df = pd.read_csv(os.path.join(data_dir, fname))

        # Parse/slice dates
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            df = df[(df["Date"] >= date_start) & (df["Date"] <= date_end)].reset_index(drop=True)
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.loc[date_start:date_end]

        # Preserve raw prices
        df["Open_raw"] = df["Open"].astype(float)
        df["Close_raw"] = df["Close"].astype(float)


        # Scale features (use numpy to avoid sklearn feature-name warnings)
        df[feature_cols] = scaler.transform(df.loc[:, feature_cols].to_numpy())

        data_by_symbol[symbol] = df

    if not data_by_symbol:
        raise ValueError(f"No *_labeled.csv files found in {data_dir}")

    # Align to common dates (intersection)
    common_idx = None
    for df in data_by_symbol.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) < 2:
        raise ValueError("Could not compute a non-empty intersection of dates")

    common_idx = common_idx.sort_values()

    for sym in list(data_by_symbol.keys()):
        data_by_symbol[sym] = data_by_symbol[sym].reindex(common_idx).dropna().reset_index(drop=True)

    return data_by_symbol


# ============================================================
# Training
# ============================================================

def train_phase_d_continuous(seed: int, output_dir: str, max_updates: int):
    """Main training loop for Phase-D continuous PPO."""

    # ----------------------------
    # Train interval
    # ----------------------------
    train_start = "2020-01-01"
    train_end = "2024-06-30"

    set_seed(seed)

    # PPO model on GPU if available; env-side frozen GRU should run on CPU for subprocess safety.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_device = "cpu"

    # Conservative defaults for continuous PPO
    ppo_cfg = make_continuous_config()

    # ----------------------------
    # Paths
    # ----------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_assets_dir = os.path.join(base_dir, "models", "checkpoints")
    data_dir = os.path.join(base_dir, "data", "processed")

    scaler_path = os.path.join(model_assets_dir, "scaler.pkl")
    gru_encoder_path = os.path.join(model_assets_dir, "gru_encoder.pt")

    if not os.path.exists(gru_encoder_path):
        raise FileNotFoundError(f"Missing GRU encoder checkpoint: {gru_encoder_path}")

    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----------------------------
    # Rollout settings
    # ----------------------------
    num_envs = 8
    rollout_steps = 252
    checkpoint_interval = 100

    # Frozen GRU params (must match checkpoint)
    gru_window = 30

    # ----------------------------
    # Features (MUST match scaler training)
    # ----------------------------
    feature_cols = [
        "Open",
        "Close",
        "RSI",
        "MACD",
        "MACD_Signal",
        "ATR",
        "SMA_50",
        "OBV",
        "ROC_10",
        "SMA_Ratio",
        "RealizedVol_20",
    ]

    # Load + align + scale
    data_by_symbol = load_and_scale_universe(
        data_dir=data_dir,
        scaler_path=scaler_path,
        feature_cols=feature_cols,
        date_start=train_start,
        date_end=train_end,
    )

    # ----------------------------
    # Environment config
    # ----------------------------
    env_cfg = TradingEnvContinuousConfig(
        # Warmup/burn-in for engineered features.
        window_length=30,
        episode_length=252,
        num_assets=14,
        include_cash=True,
        top_k=0,
        transaction_cost=0.0005,
        slippage=0.0,
        reward_mode="log_return",
        turnover_penalty=0.0,
    )

    if len(data_by_symbol) < env_cfg.num_assets:
        raise ValueError(
            f"Not enough symbols for num_assets={env_cfg.num_assets}. "
            f"Found only {len(data_by_symbol)} symbols after loading data."
        )

    # ----------------------------
    # Env factory (one env per subprocess)
    # ----------------------------
    def make_env(rank: int):
        def _init():
            return TradingEnvContinuous(
                data_by_symbol=data_by_symbol,
                feature_cols=feature_cols,
                config=env_cfg,
                seed=seed + rank,
                encoder_ckpt_path=gru_encoder_path,
                device=env_device,
                gru_window=gru_window,
            )

        return _init

    env = AsyncVectorEnv([make_env(i) for i in range(num_envs)])

    # ----------------------------
    # Model
    # ----------------------------
    obs_dim = int(env.single_observation_space.shape[0])
    action_dim = int(env.single_action_space.shape[0])

    expected_action_dim = env_cfg.num_assets + (1 if env_cfg.include_cash else 0)
    if action_dim != expected_action_dim:
        raise ValueError(f"action_dim mismatch: expected {expected_action_dim}, got {action_dim}")

    policy = PPOActorCriticContinuous(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.learning_rate)

    # Continuous env has no invalid actions; keep mask as all ones.
    action_masks = torch.ones((num_envs, action_dim), dtype=torch.float32, device=device)

    # ----------------------------
    # Logging
    # ----------------------------
    run_log = {
        "phase": "D_continuous",
        "seed": seed,
        "num_symbols": len(data_by_symbol),
        "feature_cols": feature_cols,
        "scaler_path": scaler_path,
        "gru_encoder_path": gru_encoder_path,
        "gru_window": gru_window,
        "env_device": env_device,
        "env_cfg": env_cfg.__dict__,
        "updates": [],
    }

    # ----------------------------
    # PPO loop
    # ----------------------------
    ppo_update_step = 0

    while ppo_update_step < max_updates:
        obs, infos = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        buffer = RolloutBufferContinuous()

        # Light diagnostics to catch pathological portfolio behavior
        turnover_sum = 0.0
        nonzero_sum = 0.0
        prev_w = None

        for _t in range(rollout_steps):
            with torch.no_grad():
                actions, logp, values = policy.act_batch(obs, action_masks)

            next_obs, rewards, terms, truncs, infos = env.step(actions.cpu().numpy())
            dones = np.logical_or(terms, truncs)

            # ---- diagnostics from infos (weights) ----
            weights = None
            if isinstance(infos, list):
                weights_list = [info.get("weights") for info in infos]
                if all(w is not None for w in weights_list):
                    weights = np.stack(weights_list)
            elif isinstance(infos, dict):
                weights = infos.get("weights")

            if weights is not None:
                # how many assets have non-trivial weight (exclude cash)
                nonzero = np.sum(weights[:, : env_cfg.num_assets] > 1e-12, axis=1)
                nonzero_sum += float(nonzero.mean())

                # average turnover between steps (L1 change)
                if prev_w is not None:
                    turnover_sum += float(np.abs(weights - prev_w).sum(axis=1).mean())
                prev_w = weights
            else:
                prev_w = None

            # Store transition
            buffer.add_batch(
                observations=obs,
                action_masks=action_masks,
                actions=actions,
                log_probs=logp,
                values=values,
                rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
                dones=torch.tensor(dones, dtype=torch.float32, device=device),
            )

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # Bootstrap value for GAE at last observation
        with torch.no_grad():
            _, last_values = policy.get_dist_and_value(obs, action_masks)
            last_values = last_values.squeeze(-1)

        rollout = buffer.get_tensors(flatten=True)

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
            num_envs=num_envs,
            last_values=last_values,
        )

        stats = ppo_update(
            policy,
            optim,
            rollout["observations"],
            rollout["action_masks"],
            rollout["actions"],
            rollout["log_probs"],
            advantages,
            returns,
            ppo_cfg.clip_eps,
            ppo_cfg.value_coef,
            ppo_cfg.entropy_coef,
            ppo_cfg.num_epochs,
            ppo_cfg.batch_size,
            ppo_cfg.max_grad_norm,
            target_kl=ppo_cfg.target_kl,
            kl_cutoff_multiplier=getattr(ppo_cfg, "kl_cutoff_multiplier", 1.5),
        )

        # ---- summary stats ----
        r = rollout["rewards"]
        v = rollout["values"]
        a_raw = advantages
        a_norm = (a_raw - a_raw.mean()) / (a_raw.std() + 1e-8)

        avg_nonzero = nonzero_sum / float(rollout_steps) if rollout_steps > 0 else 0.0
        avg_turnover = turnover_sum / float(max(1, rollout_steps - 1))

        r_env = r.view(-1, num_envs).sum(dim=0)

        print(
            f"[{ppo_update_step:04d}] "
            f"reward_mean={r.mean().item():+.4e} "
            f"reward_std={r.std().item():+.4e} "
            f"adv_raw_mean={a_raw.mean().item():+.4e} "
            f"adv_raw_std={a_raw.std().item():+.4e} "
            f"adv_norm_std={a_norm.std().item():+.4e} "
            f"|v|max={v.abs().max().item():+.4e} "
            f"|ret|max={returns.abs().max().item():+.4e} "
            f"entropy={stats['entropy']:+.4e} "
            f"kl={stats.get('approx_kl', 0.0):+.4e} "
            f"clipfrac={stats.get('clipfrac', 0.0):+.3f}\n"
            f"  nonzero_pos={avg_nonzero:.3f} turnover={avg_turnover:.4f}\n"
            f"  per_env_reward={r_env.cpu().numpy()}"
        )

        run_log["updates"].append(
            {
                "step": ppo_update_step,
                "reward_mean": r.mean().item(),
                "reward_std": r.std().item(),
                "adv_raw_mean": a_raw.mean().item(),
                "adv_raw_std": a_raw.std().item(),
                "adv_norm_mean": a_norm.mean().item(),
                "adv_norm_std": a_norm.std().item(),
                "values_abs_max": v.abs().max().item(),
                "returns_abs_max": returns.abs().max().item(),
                "entropy": float(stats["entropy"]),
                "approx_kl": float(stats.get("approx_kl", 0.0)),
                "clipfrac": float(stats.get("clipfrac", 0.0)),
                "nonzero_positions": float(avg_nonzero),
                "turnover": float(avg_turnover),
                "per_env_reward": r_env.detach().cpu().numpy().tolist(),
                "rollout_steps": rollout_steps,
                "num_envs": num_envs,
            }
        )

        write_log_json(os.path.join(seed_dir, "train_log.json"), run_log)

        # Checkpoint
        if (ppo_update_step + 1) % checkpoint_interval == 0:
            ckpt_payload = {
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "ppo_update_step": ppo_update_step,
                "seed": seed,
                "env_cfg": env_cfg.__dict__,
                "feature_cols": feature_cols,
                "scaler_path": scaler_path,
                "gru_encoder_path": gru_encoder_path,
                "gru_window": gru_window,
                "env_device": env_device,
                "last_update_metrics": {
                    "reward_mean": r.mean().item(),
                    "reward_std": r.std().item(),
                    "entropy": float(stats["entropy"]),
                    "approx_kl": float(stats.get("approx_kl", 0.0)),
                    "clipfrac": float(stats.get("clipfrac", 0.0)),
                    "nonzero_positions": float(avg_nonzero),
                    "turnover": float(avg_turnover),
                },
            }

            ckpt_path = os.path.join(ckpt_dir, f"ppo_step_{ppo_update_step:05d}.pt")
            torch.save(ckpt_payload, ckpt_path)

            latest_path = os.path.join(ckpt_dir, "latest.pt")
            torch.save(ckpt_payload, latest_path)

            print(f"[Checkpoint] Saved PPO checkpoint at step {ppo_update_step}")

        ppo_update_step += 1

    # Final checkpoint
    final_path = os.path.join(seed_dir, "final.pt")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "seed": seed,
            "ppo_update_step": ppo_update_step,
            "env_cfg": env_cfg.__dict__,
            "feature_cols": feature_cols,
            "scaler_path": scaler_path,
            "gru_encoder_path": gru_encoder_path,
            "gru_window": gru_window,
            "env_device": env_device,
        },
        final_path,
    )

    write_log_json(os.path.join(seed_dir, "train_log.json"), run_log)

    env.close()
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_updates", type=int, default=2000)

    default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
        "phase_d_continuous",
    )
    parser.add_argument("--output_dir", type=str, default=default_out)

    args = parser.parse_args()

    train_phase_d_continuous(seed=args.seed, output_dir=args.output_dir, max_updates=args.max_updates)