"""
Sanity Checks for Supervised GRU Pretraining Labels
===================================================

This module provides *offline diagnostics* to verify that:
- labels are well-formed
- class imbalance is reasonable
- regression targets are numerically stable
- data continuity and feature stability are maintained

These checks should be run BEFORE model training.

RUN from terminal: $env:PYTHONPATH = "."; python -m backend.utils.sanity_checks
"""

import pandas as pd
import numpy as np
import os

class DataSanityChecker:
    """
    Unified diagnostic tool to verify stock datasets before feeding them
    into the GRU encoder. Loads data once to perform multiple tests.
    """
    
    def __init__(self, file_path: str, feature_cols: list, sequence_length: int = 30):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at: {file_path}")
        
        self.file_path = file_path
        self.features = feature_cols
        self.seq_len = sequence_length
        
        # Load data ONCE for all checks to improve efficiency
        self.df = pd.read_csv(file_path)
        
        # Ensure Date is recognized for continuity checks
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            # Don't set index globally to avoid breaking other logic, 
            # just use it locally where needed.

    def run_full_diagnostics(self):
        """Executes the full suite of structural, feature, and label checks."""
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC REPORT: {os.path.basename(self.file_path)}")
        print(f"{'='*60}")

        self.check_missing_values()
        self.check_data_continuity()
        self.check_window_alignment()
        self.check_feature_stability()
        self.check_trend_distribution()
        self.check_expected_return_stats()
        
        print(f"\n{'='*60}\nEND OF REPORT\n{'='*60}\n")

    def check_missing_values(self) -> None:
        """
        Reports columns with missing values.
        
        WHY: GRUs cannot process NaNs; missing values in features will crash 
        the training math.
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
        Checks if the time-series has gaps.
        
        In Indian markets (NSE), gaps are expected on weekends/holidays, 
        but a gap of > 4 days might indicate missing historical data.
        """
        print("\nData Continuity Check:")
        if 'Date' not in self.df.columns:
            print("  [SKIP] Date column not found.")
            return

        dates = pd.to_datetime(self.df['Date']).sort_values()
        diffs = dates.diff().dt.days.dropna()
        
        gaps = diffs[diffs > 4] # More than a long weekend
        if gaps.empty:
            print("  No major date gaps detected.")
        else:
            print(f"  WARNING: Found {len(gaps)} gaps larger than 4 days (max gap: {gaps.max()} days).")

    def check_window_alignment(self) -> None:
        """
        Verifies that the dataset is large enough for the GRU window.
        """
        print("\nWindow Alignment Check:")
        total_rows = len(self.df)
        possible_windows = total_rows - self.seq_len + 1
        
        if possible_windows <= 0:
            print(f"  CRITICAL: Dataset too small ({total_rows} rows) for window size {self.seq_len}.")
        else:
            print(f"  Dataset contains {total_rows} rows, yielding {possible_windows} valid GRU sequences.")

    def check_feature_stability(self) -> None:
        """
        Checks for Infinite values or extreme outliers in features.
        
        Neural networks (GRUs) fail if they hit 'inf' or 'nan' during backpropagation.
        """
        print("\nFeature Stability Check:")
        has_issue = False
        for col in self.features:
            if col not in self.df.columns:
                print(f"  • ERROR: Column '{col}' missing from file!")
                continue
            
            # Check for Infinity - common in ROC or Ratio indicators
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                print(f"  CRITICAL: {col} contains {inf_count} INF values.")
                has_issue = True
                
            # Check for "Flat" features (Zero Variance) - predictive dead-ends
            if self.df[col].nunique() <= 1:
                print(f"  WARNING: {col} is a constant value (zero variance).")
                has_issue = True

        if not has_issue:
            print(f"  All {len(self.features)} features appear numerically stable.")

    def check_trend_distribution(self, label_col: str = "TrendLabel") -> None:
        """
        Prints class distribution for trend labels.

        Expected:
            - no extreme dominance of a single class to avoid model bias
        """
        if label_col not in self.df.columns:
            print(f"  [SKIP] {label_col} not found.")
            return

        counts = self.df[label_col].value_counts().sort_index()
        total = counts.sum()

        print("\nTrend Label Distribution:")
        for k, v in counts.items():
            print(f"  {k:+d}: {v:>7} ({v / total:.2%})")

    def check_expected_return_stats(self, col: str = "ExpectedReturn", clip: float = 0.2) -> None:
        """
        Prints basic statistics for expected return labels.

        Checks:
            - mean near zero
            - reasonable standard deviation
            - extreme outliers that could skew loss
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
    # Standard feature set used in pretraining updated with your latest logic
    FEATURES = [
        "Open", "Close", "RSI", "MACD", "MACD_Signal", "ATR", 
        "SMA_50", "OBV", "ROC_10", "SMA_Ratio", "RealizedVol_20"
    ] 
    
    # Processed stocks to check
    STOCKS = ["RELIANCE_labeled.csv", "TCS_labeled.csv", "ICICIBANK_labeled.csv"]
    DATA_DIR = "backend/data/processed"

    for stock_file in STOCKS:
        path = os.path.join(DATA_DIR, stock_file)
        # Create checker instance (loads CSV once)
        checker = DataSanityChecker(path, FEATURES)
        # Run unified diagnostics
        checker.run_full_diagnostics()