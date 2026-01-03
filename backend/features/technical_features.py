"""
Technical Feature Engineering
===========================================================
This module contains the full feature-engineering pipeline
that transforms raw OHLCV price data into the standardized
feature matrix used by:

    • GRU pretraining
    • RL environment (state representation)
    • The /api/v1/features FastAPI endpoint
    • The dataset generation script

No API imports, no Kite logic — just feature engineering.

Design Goals
------------
1. All TA-Lib calls and statistical computations remain
   centralized in one place (no duplication).

2. Feature computation is deterministic and purely functional:
   input: raw OHLCV DataFrame
   output: cleaned, aligned DataFrame of numerical features.

3. The feature set is intentionally small (11 signals):
       - Momentum      (RSI, MACD, MACD Signal)
       - Volatility    (ATR, RealizedVol_20)
       - Trend         (SMA_50, SMA_Ratio)
       - Volume Flow   (OBV)
       - Price Dynamics (ROC_10)
       - Baseline data (Open, Close)

These features are chosen because they cover the key
dimensions of market structure (trend, momentum, volatility,
volume pressure) while remaining compact enough to fit into
sequence models without overfitting.
"""

import pandas as pd
import numpy as np
import talib as ta


# ======================================================================
# Main Technical Feature Function
# ======================================================================

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 11 essential technical indicators using TA-Lib and
    statistical operations.

    This function intentionally mirrors the core logic you had in
    main.py but is extracted for clearer separation of concerns.

    INPUT:
        df — raw OHLCV with columns:
            Open, High, Low, Close, Volume

    OUTPUT:
        Clean DataFrame with 11 final features:
            Open, Close,
            RSI, MACD, MACD_Signal, ATR,
            SMA_50, OBV, ROC_10, SMA_Ratio,
            RealizedVol_20

    Notes on Validation:
    --------------------
    - TA-Lib requires float64 (“double”) arrays; we coerce explicitly.
    - Rows containing NaN values after coercion are removed BEFORE
      computing indicators.
    - We require at least 55 rows so SMA_50 / ROC_10 / ATR / MACD
      have sufficient lookback periods.

    Notes on Consistency:
    ---------------------
    The output must have consistent column ordering because:
        • ML scalers depend on fixed feature positions
        • dataset generation pipeline concatenates symbols
        • GRU models expect feature_dim = N_FEATURES
    """

    # Defensive copy — avoid mutating caller
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    # ------------------------------------------------------------------
    # Validate required OHLCV fields
    # ------------------------------------------------------------------
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in required_cols:
        if c not in d.columns:
            print(f"Missing required column for feature engineering: {c}")
            return pd.DataFrame()

    # Convert everything to numeric (float64) to satisfy TA-Lib
    for col in required_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Remove rows where OHLCV became invalid after coercion
    d.dropna(subset=required_cols, inplace=True)

    # Ensure sufficient length for indicator lookbacks
    if len(d) < 55:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Prepare ndarray inputs for TA-Lib (float64 only)
    # ------------------------------------------------------------------
    close_array = np.asarray(d["Close"], dtype=np.float64)
    high_array = np.asarray(d["High"], dtype=np.float64)
    low_array = np.asarray(d["Low"], dtype=np.float64)
    volume_array = np.asarray(d["Volume"], dtype=np.float64)


    # ==================================================================
    # Momentum / Volatility Indicators (TA-Lib)
    # ==================================================================
    try:
        d["ATR"] = ta.ATR(high_array, low_array, close_array, timeperiod=14)
        d["RSI"] = ta.RSI(close_array, timeperiod=14)
        d["MACD"], d["MACD_Signal"], _ = ta.MACD(
            close_array,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )
    except Exception as e:
        print(f"TA-Lib error (ATR/RSI/MACD): {e}")
        return pd.DataFrame()

    # ==================================================================
    # Realized Volatility (Statistical Vol)
    # ------------------------------------------------------------------
    # This is NOT a supervised label — it is an input feature that helps
    # the model understand current volatility regime.
    #
    # Formula:
    #     returns[t] = (Close[t] / Close[t-1]) - 1
    #     RealizedVol_20[t] = std(returns over 20 days)
    #
    # Chosen because ATR captures range volatility, while realized vol
    # captures *distributional* volatility — complementary signals.
    # ==================================================================
    try:
        returns = pd.Series(close_array).pct_change().astype("float64")
        realized_vol = returns.rolling(window=20).std()
        d["RealizedVol_20"] = realized_vol.values
    except Exception as e:
        print(f"Error computing RealizedVol_20: {e}")
        return pd.DataFrame()

    # ==================================================================
    # Trend + Volume + Price Change Signals
    # ==================================================================
    try:
        d["SMA_50"] = ta.SMA(close_array, timeperiod=50)
        d["OBV"] = ta.OBV(close_array, volume_array)
        d["ROC_10"] = ta.ROC(close_array, timeperiod=10)
    except Exception as e:
        print(f"TA-Lib error (SMA_50 / OBV / ROC_10): {e}")
        return pd.DataFrame()

    # Context ratio — compares present close vs long-term average
    d["SMA_Ratio"] = (d["Close"] / d["SMA_50"]) - 1.0

    # ==================================================================
    # Final Feature Selection + Cleanup
    # ==================================================================

    FEATURE_COLUMNS = [
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

    feature_df = d[FEATURE_COLUMNS].copy()

    # Drop rows where indicators still contain NaN (initial lookbacks)
    feature_df.dropna(inplace=True)

    # Enforce strict float64 for all ML pipelines
    for col in FEATURE_COLUMNS:
        feature_df[col] = feature_df[col].astype("float64")

    return feature_df
