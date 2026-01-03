# backend/tests/smoke_test_trading_env.py

import os
import joblib
import pandas as pd

from backend.services.trading_env import TradingEnv

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
    env_version="v1",
)

# ------------------------------------------------------------
# Run one episode with random actions
# ------------------------------------------------------------
obs, info = env.reset()

print(f"Episode start | symbol={info['symbol']}")
print(f"Initial obs shape: {obs.shape}")
print("-" * 60)

done = False
step = 0

while not done:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(
        f"step={step:04d} | "
        f"action={action} | "
        f"reward={reward:+.5f} | "
        f"position={info['position']}"
    )

    step += 1

print("-" * 60)
print("Episode finished")
