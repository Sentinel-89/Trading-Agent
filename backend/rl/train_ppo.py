# backend/rl/train_ppo_phase_d.py - - forked from "train_ppo_phase_c_multi_symbol.py"

"""
Discrete PPO training (deployment)

- Single Asset Model 
- Frozen GRU encoder
- Risk-aware reward (v4.1 and v5 unreliased small shaping)
- Training until convergence (no fixed episode budget)
- PPO checkpointing every 50 updates
"""

import os
import json
import argparse
import random
import torch
import joblib
import pandas as pd
import numpy as np

from backend.rl.trading_env_n import TradingEnv
from backend.rl.ppo_actor_critic import PPOActorCritic
from backend.rl.ppo_config import PPOConfig
from backend.rl.rollout_buffer import RolloutBuffer
from backend.rl.gae import compute_gae
from backend.rl.ppo_trainer import ppo_update

from gymnasium.vector import AsyncVectorEnv


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Phase-D training
# ============================================================

def train_phase_d(
    seed: int,
    output_dir: str,
    max_episodes: int,
    use_action_mask: bool = True,
):
    """
    Phase-D PPO training loop.

    Notes:
    - max_episodes is a safety cap only
    - One PPO update per episode
    """

    TRAIN_START = "2020-03-01"
    TRAIN_END = "2023-12-31"

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Config] use_action_mask={use_action_mask}")

    # ============================================================
    # PPO configuration (frozen)
    # ============================================================

    config = PPOConfig()

    # ============================================================
    # Paths
    # ============================================================

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SEED_DIR = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(SEED_DIR, exist_ok=True)

    # ---- Static model assets (inputs) --------------------------
    MODEL_ASSETS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
    ENCODER_CKPT_PATH = os.path.join(MODEL_ASSETS_DIR, "gru_encoder.pt")
    SCALER_PATH = os.path.join(MODEL_ASSETS_DIR, "scaler.pkl")

    # ---- PPO artifacts (outputs) -------------------------------
    PPO_CHECKPOINT_DIR = os.path.join(SEED_DIR, "checkpoints")
    os.makedirs(PPO_CHECKPOINT_DIR, exist_ok=True)

    CHECKPOINT_INTERVAL = 100
    ppo_update_step = 0

    # ---- Data --------------------------------------------------
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

    # ============================================================
    # Feature columns
    # ============================================================


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

    # ============================================================
    # Load scaler and data
    # ============================================================

    scaler = joblib.load(SCALER_PATH)
    data_by_symbol = {}

    for fname in os.listdir(DATA_DIR):
        if fname.endswith("_labeled.csv"):
            symbol = fname.replace("_labeled.csv", "")
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
            df = df[(df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)].reset_index(drop=True)
            df["Close_raw"] = df["Close"].astype(float)
            df["Open_raw"] = df["Open"].astype(float)
            df[feature_cols] = scaler.transform(df[feature_cols])
            data_by_symbol[symbol] = df

    # ============================================================
    # Multi-symbol universe (fixed)
    # ============================================================

    SYMBOLS = [
        "HDFCBANK", "ICICIBANK", "SBIN",
        "TCS", "INFY", "HCLTECH",
        "RELIANCE", "NTPC",
        "ITC", "HINDUNILVR",
        "MARUTI", "TMPV",
        "TATASTEEL", "JSWSTEEL",
    ]

    for s in SYMBOLS:
        assert s in data_by_symbol, f"Missing data for symbol: {s}"

    # ============================================================
    # Environment
    # ============================================================

    def make_env():
        def _init():
            # Each sub-env is one symbol episode; env reads options['symbol'] on reset.
            # Uses v5 reward, rolling_window 252, random_start.
            return TradingEnv(
                data_by_symbol=data_by_symbol,
                encoder_ckpt_path=ENCODER_CKPT_PATH,
                feature_cols=feature_cols,
                env_version="v4.1",
                episode_mode="rolling_window",
                window_length=252,
                random_start=True,
                device="cpu",
            )
        return _init

    NUM_ENVS = len(SYMBOLS)
    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = AsyncVectorEnv(env_fns)

    # ============================================================
    # Policy + optimizer
    # ============================================================

    obs_dim = env.single_observation_space.shape[0]
    policy = PPOActorCritic(obs_dim=obs_dim).to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
    )

    # ============================================================
    # Run-level log (diagnostic only)
    # ============================================================

    run_log = {
        "phase": "D",
        "seed": seed,
        "updates": [],
    }
    def _write_log_json(path: str, payload: dict) -> None:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    # ============================================================
    # Training loop
    # ============================================================

    for ep in range(max_episodes):

        ROLLOUT_STEPS = 252

        symbols = SYMBOLS.copy()
        random.shuffle(symbols)

        obs, infos = env.reset(options=[{"symbol": s} for s in symbols])
        assigned_symbols = symbols.copy()

        if isinstance(infos, dict):
            start_steps = infos.get("step", [None] * NUM_ENVS)
            true_action_masks_np = np.stack(infos["true_action_mask"]) if "true_action_mask" in infos else None
        else:
            start_steps = [info.get("step", None) for info in infos]
            true_action_masks_np = np.stack([info["true_action_mask"] for info in infos]) if (len(infos) > 0 and "true_action_mask" in infos[0]) else None

        # Stores per-step batched tensors for PPO: obs/mask/action/logp/value/reward/done.
        buffer = RolloutBuffer()

        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        if use_action_mask:
            if true_action_masks_np is None:
                raise RuntimeError("use_action_mask=True but env did not provide info['true_action_mask']")
            true_action_masks = torch.tensor(true_action_masks_np, dtype=torch.float32, device=device)
            action_masks = true_action_masks
        else:
            true_action_masks = None
            action_masks = None

        action_counts = torch.zeros(3, device=device)
        invalid_action_count = 0
        max_invalid_prob = float("nan")
        checked_invalid_prob = False

        # --- Action masking ---
        # Env must provide info["true_action_mask"] on reset() and step():
        #   shape: (num_envs, 3) after stacking
        #   order: [HOLD, BUY, SELL]
        # A value of 1.0 means the action is allowed; 0.0 means invalid.
        # When enabled, we pass the mask into the policy so the categorical distribution
        # assigns ~0 probability to invalid actions.
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                if use_action_mask and (not checked_invalid_prob) and hasattr(policy, "get_dist_and_value"):
                    try:
                        dist, _ = policy.get_dist_and_value(obs, action_masks)
                        if hasattr(dist, "probs"):
                            probs = dist.probs
                            invalid_prob = probs * (1.0 - true_action_masks)
                            max_invalid_prob = float(invalid_prob.max().item())
                    except Exception:
                        pass
                    checked_invalid_prob = True

                if use_action_mask:
                    actions, log_probs, values = policy.act_batch(obs, action_masks)
                else:
                    actions, log_probs, values = policy.act_batch(obs, None)

            #  if masking works, sampled actions should always be valid.
            if use_action_mask:
                chosen_valid = true_action_masks.gather(1, actions.view(-1, 1)).squeeze(1)
                step_invalid = int((chosen_valid < 0.5).sum().item())
                invalid_action_count += step_invalid
                if step_invalid > 0:
                    raise RuntimeError(
                        f"Action masking enabled but sampled {step_invalid} invalid actions in a step. "
                        f"max_invalid_prob={max_invalid_prob}"
                    )

            for a in actions:
                action_counts[a] += 1

            next_obs, rewards, terms, truncs, infos = env.step(actions.cpu().numpy())
            dones = np.logical_or(terms, truncs)

            # Update mask for next step using info returned by the env.
            if use_action_mask:
                if isinstance(infos, dict):
                    next_true_action_masks_np = np.stack(infos["true_action_mask"])
                else:
                    next_true_action_masks_np = np.stack([info["true_action_mask"] for info in infos])

                next_true_action_masks = torch.tensor(next_true_action_masks_np, dtype=torch.float32, device=device)
                next_action_masks = next_true_action_masks
            else:
                next_true_action_masks = None
                next_action_masks = None

            buffer.add_batch(
                observations=obs,
                action_masks=action_masks,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=torch.tensor(rewards, device=device),
                dones=torch.tensor(dones, device=device),
            )

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            true_action_masks = next_true_action_masks
            action_masks = next_action_masks

        # Flatten from (T, N, ...) to (T*N, ...) for minibatch.
        rollout = buffer.get_tensors(flatten=True)

        # Generalized advantage estimation over vectorized rollout;
        # uses dones to reset; returns are bootstrapped targets for value function.
        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            config.gamma,
            config.gae_lambda,
            num_envs=NUM_ENVS,
        )

        action_dist = (action_counts / action_counts.sum()).cpu().numpy()
        num_trades = int(action_counts[1].item() + action_counts[2].item())

        # PPO clipped objective, value loss, entropy bonus; multiple epochs over minibatches.
        # action_masks passed if treu so invalid actions are excluded 
        stats = ppo_update(
            policy,
            optimizer,
            rollout["observations"],
            rollout["action_masks"],
            rollout["actions"],
            rollout["log_probs"],
            advantages,
            returns,
            config.clip_eps,
            config.value_coef,
            config.entropy_coef,
            config.num_epochs,
            config.batch_size,
            config.max_grad_norm,
        )

        kl_val = stats.get("kl", stats.get("approx_kl", stats.get("mean_kl", None)))
        kl_val = float("nan") if kl_val is None else float(kl_val)

        mean_reward = rollout["rewards"].mean().item()
        mean_value = rollout["values"].mean().item()
        mean_adv = advantages.mean().item()
        r = rollout["rewards"]
        a = advantages

        run_log["updates"].append(
            {
                "ppo_update": ppo_update_step,
                "mean_reward_per_step": mean_reward,
                "entropy": float(stats["entropy"]),
                "kl": kl_val,
                "assigned_symbols": assigned_symbols,
                "num_trades": num_trades,
                "invalid_actions": int(invalid_action_count),
                "max_invalid_prob": float(max_invalid_prob),
                "action_counts": {
                    "HOLD": int(action_counts[0].item()),
                    "BUY": int(action_counts[1].item()),
                    "SELL": int(action_counts[2].item()),
                },
                "action_dist": {
                    "HOLD": float(action_dist[0]),
                    "BUY": float(action_dist[1]),
                    "SELL": float(action_dist[2]),
                },
            }
        )

        print(
            f"Ep {ep+1:05d} | "
            f"reward mean={r.mean():+.4e} std={r.std():.4e} "
            f"adv mean={a.mean():+.4e} std={a.std():.4e} "
            f"entropy={float(stats['entropy']):.4f} "
            f"kl={kl_val:.4e} "
            f"trades={num_trades} "
            f"invalid={invalid_action_count} "
            f"maxInvP={max_invalid_prob:.2e} "
            f"HOLD/BUY/SELL={action_dist[0]:.2f}/{action_dist[1]:.2f}/{action_dist[2]:.2f}"
        )

        # PPO checkpointing
        ppo_update_step += 1
        _write_log_json(os.path.join(SEED_DIR, "train_log.json"), run_log)

        if ppo_update_step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(
                PPO_CHECKPOINT_DIR,
                f"ppo_step_{ppo_update_step:05d}.pt"
            )

            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ppo_update_step": ppo_update_step,
                    "seed": seed,
                    "last_update_metrics": {
                    "mean_reward_per_step": mean_reward,
                    "mean_value": mean_value,
                    "mean_advantage": mean_adv,
                    "entropy": float(stats["entropy"]),
                    "kl": kl_val,
                    "num_trades": num_trades,
                    "action_dist": {
                        "HOLD": float(action_dist[0]),
                        "BUY": float(action_dist[1]),
                        "SELL": float(action_dist[2]),
                    },
                },
                },
                ckpt_path
            )

            print(f"[Checkpoint] Saved PPO checkpoint at step {ppo_update_step}")


    # ============================================================
    # Write diagnostic log
    # ============================================================

    with open(os.path.join(SEED_DIR, "train_log.json"), "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\n training finished.")
    print(f"[Final] Logs written to {SEED_DIR}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=1_000_000)
    parser.add_argument("--output_dir", type=str, default="backend/artifacts/phase_d")

    # Action mask toggle
    parser.add_argument("--use_action_mask", dest="use_action_mask", action="store_true", default=True)
    parser.add_argument("--no_action_mask", dest="use_action_mask", action="store_false")

    args = parser.parse_args()

    train_phase_d(
        seed=args.seed,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        use_action_mask=args.use_action_mask,
    )
