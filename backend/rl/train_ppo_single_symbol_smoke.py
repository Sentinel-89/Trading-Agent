# backend/rl/train_ppo_single_symbol_smoke.py

#Logged output: log2_train_ppo_single_symbol_smoke.txt  (TradingEnv v1)

# Output is not from a single batch, but averaged across batches, whereas the rollout data stays the same:
    # Rollout data:
        # ◦ observations (latents)
        # ◦ actions
        # ◦ old_log_probs
        # ◦ advantages
        # ◦ returns

"""
PPO single-symbol smoke training.

This script is intended ONLY to:
- validate PPO wiring end-to-end
- verify numerical stability
- inspect action distribution and entropy

It is NOT intended for:
- performance evaluation
- hyperparameter tuning
- multi-symbol training
"""

# This PPO-Smoke-Test...
# • creates the env
# • collects rollouts
# • computes GAE
# • normalizes advantages
# • performs PPO updates
# • is NOT intended for performance yet.
# backend/rl/train_ppo_single_symbol.py
# Successfull run confirms:
    # • the run completes
    # • no runtime errors
    # • gradients flow
    # • PPO update executes

# INCLUDES DIAGNOSTICs Tool ("Entropy ≈ log(3) initially, Action distribution roughly uniform?, Advantages centered around zero?")

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
    Single-symbol PPO smoke training.

    Goal:
    - Validate end-to-end PPO wiring
    - Ensure gradients flow
    - Observe action distribution and entropy
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # PPO configuration
    # ============================================================

    config = PPOConfig()

    # ============================================================
    # Paths (identical to trading_env smoke test)
    # ============================================================

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

    ENCODER_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gru_encoder.pt")
    SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")

    # ============================================================
    # Feature columns (must match GRU training exactly)
    # ============================================================

    feature_cols = [
        "Open",
        "Close",
        "RSI",
        "MACD",
        "MACD_Signal",
        "ATR",
        "SMA_50",
        "SMA_Ratio",
        "OBV",
        "ROC_10",
        "RealizedVol_20",
    ]

    # ============================================================
    # Load scaler and processed data (same as smoke test)
    # ============================================================

    scaler = joblib.load(SCALER_PATH)

    data_by_symbol = {}

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue

        symbol = fname.replace("_labeled.csv", "")
        path = os.path.join(DATA_DIR, fname)

        df = pd.read_csv(path)
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
    # Policy network
    # ============================================================

    policy = PPOActorCritic(
        latent_dim=obs.shape[0]
    ).to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
    )

    # ============================================================
    # Rollout buffer
    # ============================================================

    buffer = RolloutBuffer()

    # ============================================================
    # Collect one full episode
    # ============================================================

    done = False

    while not done:

        latent = torch.tensor(
            obs, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            action, log_prob, value = policy.act(latent)

        # TradingEnv expects a Python int
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

    # ============================================================
    # Compute GAE and returns
    # ============================================================

    rollout = buffer.get_tensors()

    advantages, returns = compute_gae(
        rewards=rollout["rewards"],
        values=rollout["values"],
        dones=rollout["dones"],
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    # Advantage normalization (recommended)
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )

    # ============================================================
    # Rollout diagnostics (environment + policy sanity)
    # ============================================================

    actions = rollout["actions"].cpu().numpy()
    rewards = rollout["rewards"].cpu().numpy()

    print("\n================ PPO SMOKE TEST DIAGNOSTICS ================\n")

    print("[Rollout]")
    print(f"Total steps            : {len(actions)}")
    print(
        f"Action counts          : "
        f"HOLD={(actions == 0).sum()}, "
        f"BUY={(actions == 1).sum()}, "
        f"SELL={(actions == 2).sum()}"
    )
    print(f"Total episode reward   : {rewards.sum():+.6f}")

    # ============================================================
    # Advantage diagnostics
    # ============================================================

    adv_mean = advantages.mean().item()
    adv_std = advantages.std().item()

    print("\n[Advantages]")
    print(f"Mean                   : {adv_mean:+.6f}")
    print(f"Std                    : {adv_std:+.6f}")

    # ============================================================
    # PPO update + diagnostics
    # ============================================================

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

    print("\n[PPO Update]")
    print(f"Policy loss            : {stats['policy_loss']:+.6f}")
    print(f"Value loss             : {stats['value_loss']:+.6f}")
    print(f"Entropy                : {stats['entropy']:+.6f}")

    # ============================================================
    # Smoke test pass / fail criteria
    # ============================================================

    passed = True
    reasons = []

    if len(actions) <= 50:
        passed = False
        reasons.append("episode too short")

    if stats["entropy"] <= 0.5:
        passed = False
        reasons.append("entropy too low (policy collapsed)")

    if not (abs(adv_mean) < 1e-3):
        passed = False
        reasons.append("advantage mean not ~0")

    if not (0.5 < adv_std < 2.0):
        passed = False
        reasons.append("advantage std out of range")

    if not torch.isfinite(
        torch.tensor(
            [stats["policy_loss"], stats["value_loss"], stats["entropy"]]
        )
    ).all():
        passed = False
        reasons.append("non-finite PPO losses")

    # ============================================================
    # Final verdict
    # ============================================================

    print("\n==================== SMOKE TEST RESULT ====================")

    if passed:
        print("STATUS : PASS")
        print("Meaning: PPO wiring, environment, and learning signals are sane.")
    else:
        print("STATUS : FAIL")
        print("Reasons:")
        for r in reasons:
            print(f" - {r}")

    print("===========================================================\n")

if __name__ == "__main__":
    train_single_symbol()
