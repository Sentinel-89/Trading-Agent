# backend/tests/smoke_test_trading_env.py

import os
import joblib
import pandas as pd

from backend.rl.trading_env import TradingEnv

# ------------------------------------------------------------
# Paths (adjust only if you move files)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

ENCODER_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gru_encoder.pt")
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")

# ------------------------------------------------------------
# Feature columns (must match GRU training exactly)
# ------------------------------------------------------------
FEATURE_COLS = [
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

# ------------------------------------------------------------
# Load scaler (already fit on training data)
# ------------------------------------------------------------
scaler = joblib.load(SCALER_PATH)

# ------------------------------------------------------------
# Load data per symbol
# ------------------------------------------------------------
data_by_symbol = {}

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".csv"):
        continue

    symbol = fname.replace("_labeled.csv", "")
    path = os.path.join(DATA_DIR, fname)

    df = pd.read_csv(path)

    # Apply same scaling as training
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    data_by_symbol[symbol] = df

# ------------------------------------------------------------
# Instantiate environment
# ------------------------------------------------------------
env = TradingEnv(
    data_by_symbol=data_by_symbol,
    encoder_ckpt_path=ENCODER_CKPT_PATH,
    feature_cols=FEATURE_COLS,
    env_version="v3",              # v3 under test
    episode_mode="rolling_window", # explicit (Axis B)
    window_length=252,
)

# ------------------------------------------------------------
# Run one episode with random actions
# ------------------------------------------------------------
obs, info = env.reset()

print(f"Episode start | symbol={info['symbol']}")
print(f"Initial obs shape: {obs.shape}")
print("-" * 60)

# ------------------------------------------------------------
# v3-specific invariants (checked dynamically)
# ------------------------------------------------------------
assert obs.shape == (67,), "v3 observation must be 67-dimensional"

prev_time_in_trade = 0.0
was_in_position = False

done = False
step = 0

while not done:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    position = info["position"]

    # --------------------------------------------------------
    # v3 observation layout
    #
    # [0:64]   latent market state
    # [64]     position flag
    # [65]     normalized time-in-trade
    # [66]     unrealized return
    # --------------------------------------------------------
    position_flag = obs[64]
    time_in_trade = obs[65]
    unrealized_return = obs[66]

    # --------------------------------------------------------
    # v3 semantic assertions
    # --------------------------------------------------------
    assert position_flag in (0.0, 1.0)

    if position == 0:
        # Flat state
        assert time_in_trade == 0.0
        assert unrealized_return == 0.0
        prev_time_in_trade = 0.0
        was_in_position = False
    else:
        # In position
        assert 0.0 <= time_in_trade <= 1.0

        if not was_in_position:
            # First observation after BUY
            assert 0.0 <= time_in_trade <= (1.0 / env.max_holding_period)
            was_in_position = True

        else:
            # While holding
            assert time_in_trade >= prev_time_in_trade

        prev_time_in_trade = time_in_trade

    print(
        f"step={step:04d} | "
        f"action={action} | "
        f"reward={reward:+.5f} | "
        f"position={position} | "
        f"time_in_trade={time_in_trade:.3f} | "
        f"unrealized={unrealized_return:+.5f}"
    )

    step += 1

print("-" * 60)
print("Episode finished")
