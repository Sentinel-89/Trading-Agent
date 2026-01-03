"""
End-to-End Encoder Sanity Check (Post-Training)
============================================================

This script performs *qualitative representation diagnostics* on a
PRETRAINED GRU encoder.

PURPOSE
----------------------------------------------------------------
This is NOT a training script.
This is NOT a performance benchmark.

Its sole purpose is to verify that the learned latent representation:

    • reacts sensibly to obvious market regimes
    • is stable under small temporal perturbations
    • encodes regime structure more than symbol identity

This is the FINAL verification step before freezing the encoder
and integrating it into the RL agent.

WHAT THIS SCRIPT CHECKS
----------------------------------------------------------------
1. Trend-head plausibility on obvious regimes
2. Latent stability across nearby windows
3. Cross-symbol regime similarity

All checks are *diagnostic*, not pass/fail.
Human interpretation is expected.
"""

import os
import torch
import numpy as np
import pandas as pd

# Local imports
from backend.models.gru_encoder import GRUSupervisedModel
from backend.utils.preprocessing import create_sequences, SEQUENCE_LENGTH, N_FEATURES

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "backend/data/processed"
MODEL_PATH = "backend/models/checkpoints/gru_encoder.pt"
SCALER_PATH = "backend/models/checkpoints/scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLUMNS = [
    'Open',
    'Close',
    'RSI',
    'MACD',
    'MACD_Signal',
    'ATR',
    'SMA_50',
    'OBV',
    'ROC_10',
    'SMA_Ratio',
    'RealizedVol_20'
]

# =============================================================================
# MANUAL REGIME CONFIGURATION (HUMAN-IN-THE-LOOP)
# =============================================================================
"""
IMPORTANT:
These windows MUST be selected manually by visual inspection
of the price chart for each symbol.

They are used ONLY for qualitative sanity checks.

Instructions:
1. Open the CSV in a charting tool (Excel, pandas plot, TradingView, etc.)
2. Identify obvious regimes by eye:
       • clear uptrend
       • clear downtrend
       • sideways / low-volatility
3. Provide the START INDEX of each window below.
"""
# INFO: You can use discover_strong_regimes.py in /utils folder to visually identify strong, bad, flat regimes!
REGIME_WINDOWS = {
    "RELIANCE_labeled.csv": {
        "uptrend_start":  913,  # RELIANCE 2023-11-13 = 913 = UPTREND for 30 days
        "downtrend_start": 675, # RELIANCE 2022-11-29 = 675 = DOWNTREND for 30 days
        "sideways_start": 191,  # RELIANCE 2020-12-15 = 191 = SIDEWAYS for 30 days
    },

    "TCS_labeled.csv": {
        "uptrend_start": 344, # TCS 2021-07-29 = 334 = UP
    },
}

    # Optional: add more symbols later
    # "TCS_labeled.csv": {
    #     "uptrend_start":  None,
    #     "downtrend_start": None,
    #     "sideways_start": None,
    # },

# =============================================================================
# HELPERS
# =============================================================================

# INFO: You can use discover_strong_regimes.py in /utils folder to visually identify strong, bad, flat regimes!

def load_encoder() -> GRUSupervisedModel:
    """
    Loads the pretrained GRU encoder with supervised heads.
    Heads are kept ONLY for diagnostic purposes.
    """
    model = GRUSupervisedModel(
        input_size=N_FEATURES,
        hidden_size=64,
        num_layers=1,
        dropout=0.0
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.encoder.load_state_dict(state)
    model.eval()

    return model


def load_scaler():
    import joblib
    return joblib.load(SCALER_PATH)


def load_symbol_csv(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, symbol)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def encode_window(model, scaler, window_df: pd.DataFrame):
    """
    Encodes a single window and returns:
        latent vector
        trend logits
    """
    features = scaler.transform(window_df[FEATURE_COLUMNS].values)
    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        latent = model.encoder(X)
        trend_logits, _ = model(X)

    return latent.squeeze(0), trend_logits.squeeze(0)


# =============================================================================
# CHECK 1 — TREND PLAUSIBILITY
# =============================================================================

def check_trend_plausibility(model, scaler):
    """
    Feeds obvious regime windows and prints trend logits.
    """
    print("\n" + "=" * 70)
    print("CHECK 1 — TREND HEAD PLAUSIBILITY")
    print("=" * 70)

    # Manually chosen rough regimes (by inspection) for... (set symbol accordingly!)
    symbol = "RELIANCE_labeled.csv"
    cfg = REGIME_WINDOWS.get(symbol)

    if cfg is None:
        raise ValueError(f"No regime configuration found for {symbol}")

    df = load_symbol_csv(symbol)

    regimes = {}

    if cfg["uptrend_start"] is not None:
        regimes["Uptrend Window"] = df.iloc[
            cfg["uptrend_start"] : cfg["uptrend_start"] + SEQUENCE_LENGTH
        ]

    if cfg["downtrend_start"] is not None:
        regimes["Downtrend Window"] = df.iloc[
            cfg["downtrend_start"] : cfg["downtrend_start"] + SEQUENCE_LENGTH
        ]

    if cfg["sideways_start"] is not None:
        regimes["Sideways Window"] = df.iloc[
            cfg["sideways_start"] : cfg["sideways_start"] + SEQUENCE_LENGTH
        ]

    if not regimes:
        raise RuntimeError(
            f"No valid regime windows configured for {symbol}. "
            "Please set indices in REGIME_WINDOWS."
        )

    for name, window in regimes.items():
        z, logits = encode_window(model, scaler, window)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        print(f"\n{name}")
        print(f"Trend probabilities [Down, Flat, Up]: {probs.round(3)}")


# =============================================================================
# CHECK 2 — LATENT STABILITY
# =============================================================================

def check_latent_stability(model, scaler):
    """
    Checks whether small temporal perturbations cause large latent jumps.
    """
    print("\n" + "=" * 70)
    print("CHECK 2 — LATENT STABILITY")
    print("=" * 70)

    df = load_symbol_csv("RELIANCE_labeled.csv")

    base_window = df.iloc[400:400 + SEQUENCE_LENGTH]
    shifted_window = df.iloc[401:401 + SEQUENCE_LENGTH]

    z1, _ = encode_window(model, scaler, base_window)
    z2, _ = encode_window(model, scaler, shifted_window)

    dist = torch.norm(z1 - z2).item()

    print(f"Latent L2 distance between adjacent windows: {dist:.6f}")


# =============================================================================
# CHECK 3 — CROSS-SYMBOL REGIME SIMILARITY
# =============================================================================

def check_cross_symbol_similarity(model, scaler):
    """
    Compares latent vectors of similar regimes across different symbols.

    Purpose:
        Verify that the encoder maps the SAME market regime
        (e.g. uptrend) from DIFFERENT stocks to nearby points
        in latent space.

    This confirms:
        - regime abstraction
        - symbol-invariant representation
    """

    print("\n" + "=" * 70)
    print("CHECK 3 — CROSS-SYMBOL REGIME SIMILARITY")
    print("=" * 70)

    # --------------------------------------------------------------
    # Configuration sanity
    # --------------------------------------------------------------
    required_symbols = ["RELIANCE_labeled.csv", "TCS_labeled.csv"]

    for sym in required_symbols:
        if sym not in REGIME_WINDOWS:
            raise RuntimeError(f"{sym} missing from REGIME_WINDOWS")

        if REGIME_WINDOWS[sym].get("uptrend_start") is None:
            raise RuntimeError(
                f"uptrend_start not defined for {sym} in REGIME_WINDOWS"
            )

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------
    df_rel = load_symbol_csv("RELIANCE_labeled.csv")
    df_tcs = load_symbol_csv("TCS_labeled.csv")

    # --------------------------------------------------------------
    # Extract regime windows (explicit, human-defined)
    # --------------------------------------------------------------
    rel_start = REGIME_WINDOWS["RELIANCE_labeled.csv"]["uptrend_start"]
    tcs_start = REGIME_WINDOWS["TCS_labeled.csv"]["uptrend_start"]

    w_rel = df_rel.iloc[rel_start : rel_start + SEQUENCE_LENGTH]
    w_tcs = df_tcs.iloc[tcs_start : tcs_start + SEQUENCE_LENGTH]

    # --------------------------------------------------------------
    # Encode windows
    # --------------------------------------------------------------
    z_rel, _ = encode_window(model, scaler, w_rel)
    z_tcs, _ = encode_window(model, scaler, w_tcs)

    # --------------------------------------------------------------
    # Compare latent vectors
    # --------------------------------------------------------------
    cosine_sim = torch.nn.functional.cosine_similarity(
        z_rel.unsqueeze(0),
        z_tcs.unsqueeze(0)
    ).item()

    print(
        f"Cosine similarity (RELIANCE uptrend vs TCS uptrend): "
        f"{cosine_sim:.4f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n====================================")
    print(" GRU ENCODER SANITY CHECK STARTED ")
    print("====================================")

    model = load_encoder()
    scaler = load_scaler()

    check_trend_plausibility(model, scaler)
    check_latent_stability(model, scaler)
    check_cross_symbol_similarity(model, scaler)

    print("\n====================================")
    print(" SANITY CHECK COMPLETE ")
    print("====================================\n")


if __name__ == "__main__":
    main()
