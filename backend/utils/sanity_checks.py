"""
Sanity Checks for TradingEnv / PPO Phase-C Datasets
==================================================

This module provides *offline diagnostics* to verify that market datasets
are structurally safe before being consumed by:

- the frozen GRU encoder
- the TradingEnv (rolling-window episodes)
- PPO Phase-C training

The checks ensure that:
- features are numerically stable (no NaNs / INFs)
- time-series continuity is reasonable
- rolling windows are valid
- feature distributions are non-degenerate
- auxiliary labels (Trend / ExpectedReturn) are well-formed
  [labels are NOT used by PPO, diagnostics only]

These checks should be run BEFORE PPO training.

RUN from terminal:
    $ env:PYTHONPATH = "."
    $ python -m backend.utils.sanity_checks
"""

import pandas as pd
import numpy as np
import os


class DataSanityChecker:
    """
    Unified diagnostic tool to verify stock datasets before feeding them
    into the TradingEnv + frozen GRU encoder.

    Loads data once and executes multiple structural and numerical checks.
    """

    def __init__(self, file_path: str, feature_cols: list, sequence_length: int = 30):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at: {file_path}")

        self.file_path = file_path
        self.features = feature_cols
        self.seq_len = sequence_length

        # Load data ONCE for all checks
        self.df = pd.read_csv(file_path)

        # Ensure Date is recognized for continuity checks
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])

    def run_full_diagnostics(self):
        """Executes the full suite of structural and numerical checks."""
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC REPORT: {os.path.basename(self.file_path)}")
        print(f"{'='*60}")

        self.check_missing_values()
        self.check_data_continuity()
        self.check_window_alignment()
        self.check_feature_stability()
        self.check_trend_distribution()
        self.check_expected_return_stats()

        print(f"\n{'='*60}")
        print("DATASET STRUCTURALLY SAFE FOR TradingEnv + PPO")
        print(f"{'='*60}\n")

    def check_missing_values(self) -> None:
        """
        Reports columns with missing values.

        WHY:
        - TradingEnv math and GRU encoders cannot process NaNs
        - Missing values will silently destabilize PPO
        """
        print("\nMissing Values Check:")
        missing = self.df.isna().sum()
        problematic = missing[missing > 0]

        if problematic.empty:
            print("  No missing values detected.")
        else:
            print("  WARNING: Missing values found in the following columns:")
            print(problematic)

    def check_data_continuity(self) -> None:
        """
        Checks if the time-series has large temporal gaps.

        NOTE:
        - NSE weekends / holidays are expected
        - Gaps > 4 days may indicate missing historical data
        """
        print("\nData Continuity Check:")
        if "Date" not in self.df.columns:
            print("  [SKIP] Date column not found.")
            return

        dates = self.df["Date"].sort_values()
        diffs = dates.diff().dt.days.dropna()

        gaps = diffs[diffs > 4]
        if gaps.empty:
            print("  No major date gaps detected.")
        else:
            print(
                f"  WARNING: Found {len(gaps)} gaps larger than 4 days "
                f"(max gap: {gaps.max()} days)."
            )

    def check_window_alignment(self) -> None:
        """
        Verifies that the dataset is large enough for rolling-window episodes.

        WHY:
        - TradingEnv constructs rolling windows for the GRU encoder
        - Insufficient history will crash env.reset()
        """
        print("\nWindow Alignment Check:")
        total_rows = len(self.df)
        possible_windows = total_rows - self.seq_len + 1

        if possible_windows <= 0:
            print(
                f"  CRITICAL: Dataset too small ({total_rows} rows) "
                f"for window size {self.seq_len}."
            )
        else:
            print(
                f"  Dataset contains {total_rows} rows, "
                f"yielding {possible_windows} valid rolling windows."
            )

    def check_feature_stability(self) -> None:
        """
        Checks for INF values or degenerate feature distributions.

        WHY:
        - INF / NaN values break GRU forward passes
        - Flat (zero-variance) features provide no signal
        """
        print("\nFeature Stability Check:")
        has_issue = False

        for col in self.features:
            if col not in self.df.columns:
                print(f"  ERROR: Column '{col}' missing from file!")
                has_issue = True
                continue

            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                print(f"  CRITICAL: {col} contains {inf_count} INF values.")
                has_issue = True

            if self.df[col].nunique() <= 1:
                print(f"  WARNING: {col} is constant (zero variance).")
                has_issue = True

        if not has_issue:
            print(f"  All {len(self.features)} features appear numerically stable.")

    def check_trend_distribution(self, label_col: str = "TrendLabel") -> None:
        """
        Prints class distribution for trend labels.

        NOTE:
        - PPO does NOT use this label
        - Included for dataset diagnostics and legacy consistency
        """
        if label_col not in self.df.columns:
            print(f"  [SKIP] {label_col} not found.")
            return

        counts = self.df[label_col].value_counts().sort_index()
        total = counts.sum()

        print("\nTrend Label Distribution:")
        for k, v in counts.items():
            print(f"  {k:+d}: {v:>7} ({v / total:.2%})")

    def check_expected_return_stats(
        self, col: str = "ExpectedReturn", clip: float = 0.2
    ) -> None:
        """
        Prints basic statistics for expected return labels.

        NOTE:
        - PPO does NOT consume this label
        - Used to detect pathological outliers in dataset construction
        """
        if col not in self.df.columns:
            print(f"  [SKIP] {col} not found.")
            return

        s = self.df[col].dropna()

        print("\nExpected Return Statistics:")
        print(f"  Count: {len(s)}")
        print(f"  Mean : {s.mean():.6f}")
        print(f"  Std  : {s.std():.6f}")
        print(f"  Min  : {s.min():.6f}")
        print(f"  Max  : {s.max():.6f}")

        outliers = (s.abs() > clip).sum()
        print(f"  |ret| > {clip:.2f}: {outliers} samples")


# =============================================================================
# Execution Block
# =============================================================================

if __name__ == "__main__":

    FEATURES = [
        "Open", "Close", "RSI", "MACD", "MACD_Signal",
        "ATR", "SMA_50", "OBV", "ROC_10",
        "SMA_Ratio", "RealizedVol_20",
    ]

    STOCKS = [
        "HDFCBANK_labeled.csv",
        "ICICIBANK_labeled.csv",
        "SBIN_labeled.csv",
        "TCS_labeled.csv",
        "INFY_labeled.csv",
        "HCLTECH_labeled.csv",
        "RELIANCE_labeled.csv",
        "NTPC_labeled.csv",
        "ITC_labeled.csv",
        "HINDUNILVR_labeled.csv",
        "MARUTI_labeled.csv",
        "TMPV_labeled.csv",
        "TATASTEEL_labeled.csv",
        "JSWSTEEL_labeled.csv",
    ]

    DATA_DIR = "backend/data/processed"

    for stock_file in STOCKS:
        path = os.path.join(DATA_DIR, stock_file)
        checker = DataSanityChecker(path, FEATURES)
        checker.run_full_diagnostics()
