# backend/scripts/generate_dataset.py
"""
Dataset Generation Script
==============================================================
This script is responsible for creating the perfectly structured CSV *supervised training
dataset* used for GRU pretraining.

It performs the following steps for every chosen stock symbol:

    (1) Fetches raw historical OHLCV data from Kite Connect
    (2) Applies technical feature engineering (ATR, RSI, MACD, SMA50, etc.)
        using the code defined in backend/features/technical_features.py
    (3) Computes *forward-looking labels* (Trend + Expected Return)
        using backend/utils/labeling.py
    (4) Merges everything into a clean DataFrame
    (5) Saves each symbol as backend/data/processed/<symbol>_labeled.csv

These CSVs will later be consumed by:
    - the GRU pretraining script
    - sequence generation functions in preprocessing.py
    - validation scripts
    - etc.

    BEFORE RUNNING: adjust parameters like stocks, start- and end-date (see "CONFIGURATION")
"""

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------
import os
import pandas as pd
import numpy as np

from datetime import datetime

# The refactored modules used below:
from backend.services.kite_service import fetch_historical_data
from backend.features.technical_features import calculate_technical_features
from backend.utils.labeling import add_all_labels

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------

# Where the final labeled CSVs will be stored.
OUTPUT_DIR = "backend/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Symbols to include in the dataset.
DEFAULT_SYMBOLS = [
    # Banking & Finance
    "HDFCBANK", "ICICIBANK", "SBIN",
    # IT Services
    "TCS", "INFY", "HCLTECH",
    # Energy
    "RELIANCE", "NTPC",
    # FMCG
    "ITC", "HINDUNILVR",
    # Automobiles
    "MARUTI", "TATAMOTORS",
    # Metals
    "TATASTEEL", "JSWSTEEL",
]

# Dataset parameters — fully adjustable
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TREND_HORIZON = 1     # predict next-day direction
RETURN_HORIZON = 3    # predict 3-day returns


# --------------------------------------------------------------
# Core Helper: Process a single symbol
# --------------------------------------------------------------

def generate_single_symbol_dataset(symbol: str) -> pd.DataFrame:
    """
    Fetches raw OHLCV → computes features → adds labels.
    Returns a DataFrame that can be directly saved as CSV.

    The structure of the returned dataset is:

        Date
        <11 engineered features>  (Open, Close, RSI, MACD, ATR, …, RealizedVol_20)
        TrendLabel
        ExpectedReturn

    Notes:
    - Missing rows (due to rolling windows or label horizons) remain at the end
      as NaN and will be dropped before training.
    """

    print(f"\n==============================================")
    print(f"Processing symbol: {symbol}")
    print("==============================================")

    # 1. Retrieve raw OHLCV data
    df_raw = fetch_historical_data(symbol, start_date=START_DATE, end_date=END_DATE)
    if df_raw.empty:
        print(f"[ERROR] No data returned for {symbol}. Skipping.")
        return pd.DataFrame()
    print(f"[INFO] Retrieved {len(df_raw)} raw rows for {symbol}.")

    # 2. Apply technical feature engineering
    df_feat = calculate_technical_features(df_raw)
    if df_feat.empty:
        print(f"[ERROR] Failed to compute features for {symbol}.")
        return pd.DataFrame()

    print(f"[INFO] Feature matrix shape: {df_feat.shape}")

    # 3. Add trend + expected return labels
    df_labeled = add_all_labels(
        df_feat,
        trend_horizon=TREND_HORIZON,
        return_horizon=RETURN_HORIZON,
    )

    if df_labeled.empty:
        print(f"[ERROR] Label generation failed for {symbol}.")
        return pd.DataFrame()

    print(f"[INFO] Labeled dataset shape: {df_labeled.shape}")

    return df_labeled


# --------------------------------------------------------------
# Save helper
# --------------------------------------------------------------

def save_labeled_dataset(df: pd.DataFrame, symbol: str):
    """
    Writes the labeled DataFrame to disk.

    Output path:
        backend/data/processed/<SYMBOL>_labeled.csv

    Files produced here form the foundation of your supervised
    GRU encoder pretraining stage.
    """
    filename = f"{symbol}_labeled.csv"
    path = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(path)
    print(f"[OK] Saved dataset: {path}")


# --------------------------------------------------------------
# Main process
# --------------------------------------------------------------

def main(symbols=None):
    """
    Generates datasets for all requested symbols (default list).
    This function:

        - iterates over each stock
        - builds its dataset
        - saves it

    After execution, the processed/ directory will contain
    multiple <symbol>_labeled.csv files — your entire training corpus.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print("\n====================================================")
    print("          GENERATING SUPERVISED DATASETS")
    print("====================================================\n")

    for sym in symbols:
        df = generate_single_symbol_dataset(sym)
        if df.empty:
            continue
        save_labeled_dataset(df, sym)

    print("\n====================================================")
    print(" Dataset Generation COMPLETED SUCCESSFULLY")
    print(" Files saved to backend/data/processed/")
    print("====================================================\n")


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------

if __name__ == "__main__":
    main()


# Before 