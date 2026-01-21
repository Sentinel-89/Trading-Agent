# backend/rl/train_ppo_phase_d.py - - forked from "train_ppo_phase_c_multi_symbol.py"

"""
Phase-D PPO training (deployment)

- Multi-symbol training
- Frozen GRU encoder
- Risk-aware reward (v4.1 only)
- Training until convergence (no fixed episode budget)
- PPO checkpointing every 100 updates
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
# Phase-D training
# ============================================================

def train_phase_d(
    seed: int,
    output_dir: str,
    max_episodes: int,
):
    """
    Phase-D PPO training loop.

    Notes:
    - max_episodes is a safety cap only
    - One PPO update per episode
    """

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        if fname.endswith(".csv"):
            symbol = fname.replace("_labeled.csv", "")
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
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
    # Environment (hard-locked v4.1)
    # ============================================================

    env = TradingEnv(
        data_by_symbol=data_by_symbol,
        encoder_ckpt_path=ENCODER_CKPT_PATH,
        feature_cols=feature_cols,
        env_version="v4.1",
        episode_mode="rolling_window",
        window_length=252,
        device=device,
    )

    # ============================================================
    # Policy + optimizer
    # ============================================================

    obs, _ = env.reset(symbol=SYMBOLS[0])
    policy = PPOActorCritic(latent_dim=obs.shape[0]).to(device)

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
        "episodes": [],
    }

    # ============================================================
    # Training loop
    # ============================================================

    for ep in range(max_episodes):

        symbol = np.random.choice(SYMBOLS)
        obs, _ = env.reset(symbol=symbol)
        buffer = RolloutBuffer()

        done = False
        total_drawdown_penalty = 0.0
        max_drawdowns = []
        num_sells = 0

        while not done:

            latent = torch.tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, value = policy.act(latent)

            action_int = int(action.item())
            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            if action_int == 2 and info["position"] == 0:
                d_max = env.max_drawdown_in_trade
                total_drawdown_penalty += env.lambda_drawdown * abs(d_max)
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

        rollout = buffer.get_tensors()

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            config.gamma,
            config.gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
            config.max_grad_norm,   
        )

        # --------------------------------------------------------
        # PPO checkpointing
        # --------------------------------------------------------

        ppo_update_step += 1

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
                },
                ckpt_path
            )

            print(f"[Checkpoint] Saved PPO checkpoint at step {ppo_update_step}")

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        run_log["episodes"].append({
            "episode": ep + 1,
            "symbol": symbol,
            "total_reward": float(rollout["rewards"].sum()),
            "entropy": float(stats["entropy"]),
        })

        print(
            f"Ep {ep+1:05d} | {symbol:>10} | "
            f"reward={run_log['episodes'][-1]['total_reward']:+.4f} | "
            f"entropy={stats['entropy']:.4f}"
        )

    # ============================================================
    # Save final model
    # ============================================================

    final_path = os.path.join(SEED_DIR, "final.pt")

    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "seed": seed,
            "ppo_update_step": ppo_update_step,
            "env_version": "v4.1",
        },
        final_path
    )

    # ============================================================
    # Write diagnostic log
    # ============================================================

    with open(os.path.join(SEED_DIR, "train_log.json"), "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\n[Final] Phase-D training finished.")
    print(f"[Final] Model saved to {final_path}")
    print(f"[Final] Logs written to {SEED_DIR}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=1_000_000)
    parser.add_argument("--output_dir", type=str, default="backend/artifacts/phase_d")

    args = parser.parse_args()

    train_phase_d(
        seed=args.seed,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
    )
