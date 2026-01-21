# backend/rl/train_ppo_phase_c_multi_symbol.py

"""
Phase-C PPO training (multi-symbol)

- One PPO update per episode
- Fixed rollout length (rolling window)
- One symbol sampled per episode
- JSON logging (one file per run)
"""

import os
import json
import argparse
import random
import torch
import joblib
import pandas as pd
import numpy as np

from backend.services.trading_env import TradingEnv
from backend.models.ppo_actor_critic import PPOActorCritic
from backend.rl.ppo_config import PPOConfig
from backend.rl.rollout_buffer import RolloutBuffer
from backend.rl.gae import compute_gae
from backend.rl.ppo_trainer import ppo_update


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
# Phase-C training
# ============================================================

def train_phase_c(
    env_version: str,
    num_episodes: int,
    seed: int,
    output_dir: str,
):
    """
    Phase-C PPO training (multi-symbol).

    - One PPO update per episode
    - Fixed rollout length (rolling window)
    - One symbol sampled per episode
    - JSON logging (one file per run)
    """

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # PPO configuration (FROZEN)
    # ============================================================

    config = PPOConfig()

    # ============================================================
    # Paths
    # ============================================================

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dynamically append the env_version subfolder under phase_c_multi/
    VERSIONED_OUTPUT_DIR = os.path.join(output_dir, env_version)

    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

    ENCODER_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gru_encoder.pt")
    SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")

    # ============================================================
    # Feature columns
    # ============================================================

    feature_cols = [
        "Open", "Close", "RSI", "MACD", "MACD_Signal",
        "ATR", "SMA_50", "SMA_Ratio", "OBV", "ROC_10",
        "RealizedVol_20",
    ]

    # ============================================================
    # Load scaler and data
    # ============================================================

    scaler = joblib.load(SCALER_PATH)
    data_by_symbol = {}

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        symbol = fname.replace("_labeled.csv", "")
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        df[feature_cols] = scaler.transform(df[feature_cols])
        data_by_symbol[symbol] = df

    # ============================================================
    # Multi-symbol universe (Axis B extension) - Full-Phase-C
    # ============================================================

    SYMBOLS = [
        # Banking & Finance
        "HDFCBANK", "ICICIBANK", "SBIN",
        # IT Services
        "TCS", "INFY", "HCLTECH",
        # Energy
        "RELIANCE", "NTPC",
        # FMCG
        "ITC", "HINDUNILVR",
        # Automobiles
        "MARUTI", "TMPV",
        # Metals
        "TATASTEEL", "JSWSTEEL",
    ]

    # Safety check
    for s in SYMBOLS:
        assert s in data_by_symbol, f"Missing data for symbol: {s}"

    # ============================================================
    # Environment
    # ============================================================

    env = TradingEnv(
        data_by_symbol=data_by_symbol,
        encoder_ckpt_path=ENCODER_CKPT_PATH,
        feature_cols=feature_cols,
        env_version=env_version,           # v3 or v4.1
        episode_mode="rolling_window",
        window_length=252,
        device=device,
    )

    # ============================================================
    # Policy + optimizer
    # ============================================================

    # Infer observation dimension via a dummy reset (i.e. observation dimensionality depends only on env_version)
    dummy_symbol = SYMBOLS[0]
    obs, _ = env.reset(symbol=dummy_symbol)

    policy = PPOActorCritic(latent_dim=obs.shape[0]).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
    )

    # ============================================================
    # Run-level log
    # ============================================================

    run_log = {
        "env_version": env_version,
        "seed": seed,
        "num_episodes": num_episodes,
        "episodes": [],
    }

    # ============================================================
    # Phase-C training loop
    # ============================================================

    for ep in range(num_episodes):

        symbol = np.random.choice(SYMBOLS) # episode-level symbol sampling (from curated list of 14 symbols)

        obs, info = env.reset(symbol=symbol)
        buffer = RolloutBuffer()

        done = False

        # --------------------------------------------------------
        # v4.1 drawdown diagnostics (per episode)
        # --------------------------------------------------------

        total_drawdown_penalty = 0.0
        max_drawdowns = []
        num_sells = 0

        # --------------------------------------------------------
        # Rollout collection
        # --------------------------------------------------------

        while not done:

            latent = torch.tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, value = policy.act(latent)

            action_int = int(action.item())
            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            # ----------------------------------------------------
            # v4.1 SELL diagnostics
            # ----------------------------------------------------

            if env_version == "v4.1" and action_int == 2 and info["position"] == 0:
                d_max = env.max_drawdown_in_trade
                penalty = env.lambda_drawdown * abs(d_max)

                total_drawdown_penalty += penalty
                max_drawdowns.append(d_max)
                num_sells += 1

            buffer.add(
                latent=latent,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
            )

            obs = next_obs

        # --------------------------------------------------------
        # GAE
        # --------------------------------------------------------

        rollout = buffer.get_tensors()

        advantages, returns = compute_gae(
            rewards=rollout["rewards"],
            values=rollout["values"],
            dones=rollout["dones"],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # --------------------------------------------------------
        # PPO update
        # --------------------------------------------------------

        stats = ppo_update(
            policy,
            optimizer,
            rollout["latents"],
            rollout["actions"],
            rollout["log_probs"],
            advantages,
            returns,
            config.clip_eps,
            config.value_coef,
            config.entropy_coef,
            config.num_epochs,
            config.batch_size,
            config.max_grad_norm,   # 👈 NEW
        )


        # --------------------------------------------------------
        # Episode diagnostics
        # --------------------------------------------------------

        actions = rollout["actions"].cpu().numpy()
        rewards = rollout["rewards"].cpu().numpy()

        action_freq = {
            "HOLD": float((actions == 0).mean()),
            "BUY": float((actions == 1).mean()),
            "SELL": float((actions == 2).mean()),
        }

        episode_log = {
            "episode": ep + 1,
            "symbol": symbol,
            "steps": len(actions),
            "total_reward": float(rewards.sum()),
            "entropy": float(stats["entropy"]),
            "action_freq": action_freq,
        }

        if env_version == "v4.1":
            episode_log.update({
                "num_sells": num_sells,
                "mean_drawdown_penalty": (
                    total_drawdown_penalty / num_sells
                    if num_sells > 0 else 0.0
                ),
                "max_drawdown_per_trade": max_drawdowns,
            })

        run_log["episodes"].append(episode_log)

        print(
            f"Ep {ep+1:03d} | {symbol:>10} | "
            f"reward={episode_log['total_reward']:+.4f} | "
            f"entropy={episode_log['entropy']:.4f} | "
            f"H={action_freq['HOLD']:.2f} "
            f"B={action_freq['BUY']:.2f} "
            f"S={action_freq['SELL']:.2f}"
        )

    # ============================================================
    # Write JSON log
    # ============================================================

    # This now creates 'backend/artifacts/phase_c/v4.1/' (example)
    os.makedirs(VERSIONED_OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(
        VERSIONED_OUTPUT_DIR,
        f"phase_c_multi_{env_version}_seed{seed}.json",
    )

    with open(out_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\nPhase-C run completed. Log written to:\n{out_path}\n")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_version", type=str, default="v4.1")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="backend/artifacts/phase_c_multi")

    args = parser.parse_args()

    train_phase_c(
        env_version=args.env_version,
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
