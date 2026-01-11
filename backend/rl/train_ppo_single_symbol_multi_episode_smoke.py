# backend/rl/train_ppo_single_symbol_multi_episode_smoke.py

"""
PPO single-symbol multi-episode smoke training.

Purpose:
- Validate PPO wiring over multiple episodes
- Inspect entropy decay and action frequency drift
- Confirm learning dynamics remain stable

NOT intended for:
- performance evaluation
- hyperparameter tuning
- multi-symbol training
"""

import os
import torch
import joblib
import pandas as pd

from backend.services.trading_env import TradingEnv
from backend.models.ppo_actor_critic import PPOActorCritic
from backend.rl.ppo_config import PPOConfig
from backend.rl.rollout_buffer import RolloutBuffer
from backend.rl.gae import compute_gae
from backend.rl.ppo_trainer import ppo_update


def train_single_symbol():
    """
    Multi-episode PPO smoke training (single symbol).

    Each episode:
    - collects one full rollout
    - performs one PPO update
    - logs diagnostics

    Goal:
    - Validate learning dynamics over time
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # PPO configuration
    # ============================================================

    config = PPOConfig()

    NUM_EPISODES = 5  # smoke-level only

    # ============================================================
    # Paths (identical to trading_env smoke test)
    # ============================================================

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

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
    # Environment (configure here!)
    # ============================================================

    env = TradingEnv(
    data_by_symbol=data_by_symbol,
    encoder_ckpt_path=ENCODER_CKPT_PATH,
    feature_cols=feature_cols,
    env_version="v4",                 # Axis A (explicit)
    episode_mode="rolling_window",    # Axis B (explicit)
    window_length=252,
    device=device,
    )

    obs, info = env.reset()

    # ============================================================
    # Policy + optimizer
    # ============================================================

    policy = PPOActorCritic(latent_dim=obs.shape[0]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    # ============================================================
    # Episode-level logs
    # ============================================================

    episode_rewards = []
    episode_entropies = []
    episode_action_freqs = []

    # ============================================================
    # Multi-episode loop
    # ============================================================

    for ep in range(NUM_EPISODES):

        obs, info = env.reset()
        buffer = RolloutBuffer()
        done = False

        # --------------------------------------------------------
        # Rollout collection
        # --------------------------------------------------------

        while not done:

            latent = torch.tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, value = policy.act(latent)

            action_int = int(action.item())
            next_obs, reward, terminated, truncated, _ = env.step(action_int)
            done = terminated or truncated

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
            policy=policy,
            optimizer=optimizer,
            observations=rollout["latents"],
            actions=rollout["actions"],
            old_log_probs=rollout["log_probs"],
            advantages=advantages,
            returns=returns,
            clip_eps=config.clip_eps,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
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

        episode_rewards.append(rewards.sum())
        episode_entropies.append(stats["entropy"])
        episode_action_freqs.append(action_freq)

        print(
            f"Episode {ep+1:02d} | "
            f"steps={len(actions):4d} | "
            f"reward={rewards.sum():+.4f} | "
            f"entropy={stats['entropy']:.4f} | "
            f"H={action_freq['HOLD']:.2f} "
            f"B={action_freq['BUY']:.2f} "
            f"S={action_freq['SELL']:.2f}"
        )

    # ============================================================
    # End-of-run summary
    # ============================================================

    print("\n================ MULTI-EPISODE SUMMARY ================\n")

    for i in range(NUM_EPISODES):
        print(
            f"Ep {i+1:02d} | "
            f"reward={episode_rewards[i]:+.4f} | "
            f"entropy={episode_entropies[i]:.4f} | "
            f"{episode_action_freqs[i]}"
        )

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    train_single_symbol()