# backend/models/datasets.py
"""
PyTorch Dataset Wrappers for GRU Pretraining
============================================================
Its responsibility is *purely data handling*:
    - load labeled CSVs
    - apply scaling
    - create sliding windows (sequences)
"""
# backend/models/datasets.py
"""
PyTorch Dataset Wrappers for GRU Pretraining
============================================================

This module bridges the gap between:

    • CSV datasets produced by `generate_dataset.py`
    • The GRU encoder defined in `gru_encoder.py`

Its responsibility is *purely data handling*:
    - load labeled CSVs
    - apply scaling
    - create sliding windows (sequences)
    - expose PyTorch-compatible datasets

IMPORTANT DESIGN PRINCIPLE
------------------------------------------------------------
This file does NOT:
    - define neural networks
    - perform training loops
    - contain business logic

It ONLY prepares data so that training code remains clean
and focused solely on optimization.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from sklearn.preprocessing import MinMaxScaler

# Local imports (core preprocessing logic)
from backend.utils.preprocessing import create_sequences


# =============================================================================
# Dataset for Supervised GRU Pretraining
# =============================================================================

class GRUPretrainingDataset(Dataset):
    """
    Dataset for supervised pretraining of the GRU encoder.

    Each sample corresponds to:
        X : (Sequence_Length, Num_Features)
        y_trend : int     (-1, 0, +1) mapped to {0,1,2}
        y_return : float (expected return over horizon)

    DATA FLOW
    ------------------------------------------------------------------
    CSV files (generate_dataset.py)
        ↓
    pandas DataFrame
        ↓
    scaling (MinMaxScaler)
        ↓
    sliding windows (create_sequences)
        ↓
    PyTorch Dataset
    """

    def __init__(
        self,
        csv_paths: List[str],
        feature_columns: List[str],
        trend_label_col: str = "TrendLabel",
        return_label_col: str = "ExpectedReturn",
        sequence_length: int = 30,

        # ---------------- NEW (temporal split control) ----------------
        split: str = "train",                 # "train" or "val"
        val_start_date: Optional[str] = None, # e.g. "2024-01-01"
        # --------------------------------------------------------------

        scaler: Optional[MinMaxScaler] = None,
        fit_scaler: bool = False,
    ):
        """
        Parameters
        ----------
        csv_paths : List[str]
            Paths to labeled CSV files (one or many symbols)

        feature_columns : List[str]
            Columns used as GRU input features (must match preprocessing)

        trend_label_col : str
            Name of trend classification label column

        return_label_col : str
            Name of expected return regression label column

        sequence_length : int
            Number of timesteps per GRU input window

        split : str
            Dataset split identifier.
            Must be either "train" or "val".

        val_start_date : Optional[str]
            Boundary date for temporal validation split.
            Required when split="val".
            All rows with Date >= val_start_date belong to validation.

        scaler : Optional[MinMaxScaler]
            If provided, used to transform features
            If None and fit_scaler=True → new scaler is fitted

        fit_scaler : bool
            Should ONLY be True during training
            NEVER during validation or inference
        """

        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'")

        if split == "val" and val_start_date is None:
            raise ValueError(
                "val_start_date must be provided when split='val'"
            )

        self.feature_columns = feature_columns
        self.sequence_length = sequence_length

        self.trend_label_col = trend_label_col
        self.return_label_col = return_label_col

        self.split = split
        self.val_start_date = val_start_date

        # --------------------------------------------------------------
        # 1. PRE-LOAD: Fit scaler globally across all symbols
        # --------------------------------------------------------------
        # We need to see the min/max of ALL stocks before processing windows.
        #
        # IMPORTANT:
        # The scaler MUST be fit ONLY on training-time data
        # to avoid look-ahead bias and leakage.
        if scaler is None:
            self.scaler = MinMaxScaler()
            if fit_scaler:
                all_dfs = []

                for p in csv_paths:
                    df = pd.read_csv(p)
                    df["Date"] = pd.to_datetime(df["Date"])

                    # Apply temporal split BEFORE fitting scaler
                    if split == "train":
                        df = df[df["Date"] < val_start_date]

                    all_dfs.append(df)

                full_temp_df = pd.concat(all_dfs, axis=0, ignore_index=True)
                self.scaler.fit(full_temp_df[feature_columns].values)
            else:
                raise ValueError(
                    "Scaler not provided and fit_scaler=False. "
                    "This would cause data leakage."
                )
        else:
            self.scaler = scaler


        # --------------------------------------------------------------
        # 2. PER-SYMBOL PROCESSING: Create sequences stock-by-stock
        # --------------------------------------------------------------
        X_list = []
        y_trend_list = []
        y_return_list = []

        for path in csv_paths:
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])

            # ----------------------------------------------------------
            # TEMPORAL SPLIT (per symbol)
            # ----------------------------------------------------------
            if split == "train":
                df = df[df["Date"] < val_start_date]
            else:  # split == "val"
                df = df[df["Date"] >= val_start_date]

            # Drop rows where labels are missing (tail rows after shifting)
            df.dropna(subset=[trend_label_col, return_label_col], inplace=True)

            # IMPORTANT:
            # After temporal splitting + dropna(), indices may be non-contiguous.
            # Reset index to guarantee feature/label alignment during windowing.
            df.reset_index(drop=True, inplace=True)


            # Skip symbols that do not have enough rows for windowing
            if len(df) < sequence_length:
                continue

            # Scale features for THIS SYMBOL ONLY
            features = df[feature_columns].values
            scaled = self.scaler.transform(features)

            # Create sequences for THIS SYMBOL ONLY
            # This avoids windows containing data from two different symbols.
            X_sym, _ = create_sequences(
                data=scaled,
                sequence_length=sequence_length,
                training_mode=True
            )

            # ----------------------------------------------------------
            # Labels must align with the END of each ACTUAL window
            #
            # window covers [t-(sequence_length-1) ... t]
            # → label corresponds to index t
            #
            # IMPORTANT:
            # The number of windows returned by create_sequences()
            # is the authoritative length. Labels must be trimmed
            # to match it exactly.
            # ----------------------------------------------------------
            offset = sequence_length - 1
            num_windows = X_sym.shape[0]

            y_trend_sym = (
                df[trend_label_col]
                .iloc[offset : offset + num_windows]
                .to_numpy(dtype=np.int64)
                + 1  # map {-1,0,1} -> {0,1,2}
            )

            y_return_sym = (
                df[return_label_col]
                .iloc[offset : offset + num_windows]
                .values
            )

            # Safety check: Ensure strict window/label alignment
            if len(y_trend_sym) != num_windows:
                raise RuntimeError(f"Window/label misalignment in {path}")


            X_list.append(X_sym)
            y_trend_list.append(y_trend_sym)
            y_return_list.append(y_return_sym)

        # --------------------------------------------------------------
        # 3. FINAL CONCATENATION: Create the "Universal" pool
        # --------------------------------------------------------------
        # Combine all symbol windows into one large PyTorch-compatible set
        self.X = torch.cat(X_list, dim=0) # (N, T, F)
        
        # Concatenate labels and convert to Tensors
        self.y_trend = torch.tensor(np.concatenate(y_trend_list), dtype=torch.long)
        self.y_return = torch.tensor(
            np.concatenate(y_return_list), dtype=torch.float32
        ).unsqueeze(-1) # (N, 1)

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns one training sample:

            X_window       → (SeqLen, NumFeatures)
            y_trend        → scalar class index
            y_expected_ret → scalar regression target
        """
        return (
            self.X[idx],
            self.y_trend[idx],
            self.y_return[idx]
        )


# =============================================================================
# Helper: Load dataset from directory
# =============================================================================

def load_gru_pretraining_dataset(
    data_dir: str,
    feature_columns: List[str],
    sequence_length: int,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = False,
) -> GRUPretrainingDataset:
    """
    Convenience loader that reads all CSVs in a directory.

    Example:
        dataset = load_gru_pretraining_dataset(
            "backend/data/processed",
            feature_columns=FEATURES,
            sequence_length=30,
            fit_scaler=True
        )
    """

    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    return GRUPretrainingDataset(
        csv_paths=csv_paths,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        scaler=scaler,
        fit_scaler=fit_scaler
    )