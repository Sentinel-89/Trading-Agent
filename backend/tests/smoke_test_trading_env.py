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
# Action constants (must match env action mapping)
# ------------------------------------------------------------
HOLD = 0
BUY = 1
SELL = 2

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
    env_version="v4",               # v4 under test
    episode_mode="rolling_window",  # explicit (Axis B)
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
# v4 observation invariants
# ------------------------------------------------------------
assert obs.shape == (69,), "v4 observation must be 69-dimensional"
assert env.max_drawdown_in_trade <= 0.0 # added for v4.1 (with reward shaping)

prev_time_in_trade = 0.0
was_in_position = False

prev_equity = None
prev_equity_peak = None

prev_position = env.position

done = False
step = 0

while not done:
    action = env.action_space.sample()

    # --------------------------------------------------------
    # Capture pre-step position (state-transition aware)
    # --------------------------------------------------------
    prev_position = env.position

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    position = info["position"]

    # --------------------------------------------------------
    # v4 observation layout
    #
    # [0:64]   latent market state
    # [64]     position flag
    # [65]     normalized time-in-trade
    # [66]     unrealized return
    # [67]     equity (normalized)
    # [68]     drawdown (normalized)
    # --------------------------------------------------------
    position_flag = obs[64]
    time_in_trade = obs[65]
    unrealized_return = obs[66]
    equity = obs[67]
    drawdown = obs[68]

    # --------------------------------------------------------
    # v3 semantic assertions (unchanged)
    # --------------------------------------------------------
    assert position_flag in (0.0, 1.0)

    if position == 0:
        assert time_in_trade == 0.0
        assert unrealized_return == 0.0
        prev_time_in_trade = 0.0
        was_in_position = False
    else:
        assert 0.0 <= time_in_trade <= 1.0

        if not was_in_position:
            # First step after BUY
            assert time_in_trade <= (1.0 / env.max_holding_period)
            was_in_position = True
        else:
            # While holding (non-terminal continuity)
            if not done:
                assert time_in_trade >= prev_time_in_trade

        prev_time_in_trade = time_in_trade

    # --------------------------------------------------------
    # v4 portfolio invariants (log-only, mark-to-market)
    # --------------------------------------------------------
    if not done:
        assert equity > 0.0, "Equity must remain positive"
        assert drawdown <= 0.0, "Drawdown must be <= 0"
        assert drawdown >= -1.0, "Drawdown must be >= -1"

    if not done and prev_equity_peak is not None:
        # Equity peak is monotonic non-decreasing
        assert env.equity_peak >= prev_equity_peak

    if not done and prev_equity is not None:
        if prev_position == 0 and action != BUY:
            # Flat → no BUY → no equity change
            assert abs(equity - prev_equity) < 1e-6

    prev_equity = equity
    prev_equity_peak = env.equity_peak

    print(
        f"step={step:04d} | "
        f"action={action} | "
        f"reward={reward:+.5f} | "
        f"position={position} | "
        f"time_in_trade={time_in_trade:.3f} | "
        f"unrealized={unrealized_return:+.5f} | "
        f"equity={equity:.5f} | "
        f"drawdown={drawdown:.5f}"
    )

    step += 1

print("-" * 60)
print("Episode finished")
