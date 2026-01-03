# Does NOT compute features (this is handled in main.py !) — it is only responsible for:

#   • scaling,
#   • shaping (i.e. conversion to) tensors (for GRU),
#   • producing (sliding) windows (for GRU).

# Same Preprocessing will be used for training and inference purpose!

# "Helper-Fucntion" called from other backend files (e.g. datasets.py, for RL environment, model training, etc.)

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional, Union

# =====================================================================================
# --- GLOBAL CONSTANTS ---
# =====================================================================================

# The number of historical time steps (days) the GRU will look back.
# This is the "Window Size" or "Sequence Length". For daily data, 20–30 days is common.
SEQUENCE_LENGTH = 30

# The number of features in each time step (Open, Close, RSI, MACD, etc.).
# After feature engineering, we currently have 10 final standardized features.
N_FEATURES = 11

# --- NORMALIZATION LOGIC (CRUCIAL FOR NEURAL NETWORKS) ---
# We define the scaler globally so it can be fitted once during *training*
# and then saved/loaded for *inference* to ensure consistent scaling.
SCALER = MinMaxScaler()


# =====================================================================================
# SCALER FIT / TRANSFORM
# =====================================================================================

def fit_and_save_scaler(df: pd.DataFrame, file_path: str = "scaler.pkl") -> None:
    """
    Fits the MinMaxScaler on the full training DataFrame and (optionally) saves it.
    
    IMPORTANT:
    Used ONLY ONCE during offline model training.
    During inference, we load the previously fitted scaler to ensure consistency.
    """
    SCALER.fit(df.values)

    # Real deployment: persist using joblib or pickle
    # import joblib
    # joblib.dump(SCALER, file_path)
    print("Scaler fitted and ready to be saved/used.")


def load_and_transform_data(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """
    Applies a *previously fitted* MinMaxScaler to new data (training or inference).

    Converts DataFrame → NumPy array → scaled array.
    """
    if df.empty:
        return np.array([])

    data_array = df.values
    scaled_data = scaler.transform(data_array)
    return scaled_data


# =====================================================================================
# HYBRID SEQUENCE GENERATOR (TRAINING + INFERENCE, NUMPY + DATAFRAME)
# =====================================================================================

def create_sequences(
    data: Union[np.ndarray, pd.DataFrame],
    sequence_length: int,
    training_mode: bool = False,
    feature_columns: Optional[List[str]] = None,
    symbol_col: Optional[str] = None,
    target: Optional[str] = None,
    prediction_horizon: int = 1,
    step: int = 1,
) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
    """
    Hybrid window/sequence generator for preparing GRU inputs.

    Supports:
      - NumPy arrays (fast path for inference & RL environments)
      - Pandas DataFrames (ideal for supervised GRU pretraining)
      - Training mode: generates ALL sliding windows
      - Inference mode: returns the *final window only*
      - Optional supervised labels ("target") for regression / classification tasks
      - Optional symbol grouping for multi-stock datasets

    OUTPUT SHAPES:
      X_tensor: (Batch, Sequence_Length, N_FEATURES)
      y_tensor: (Batch,) or (Batch, 1)     [only when target provided]
      start_index: int                     [inference-only, alignment marker]
    """

    # --- Validations --- #
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")

    # ==============================================================================
    # CASE A — NUMPY ARRAY INPUT (Fastest execution path, used by inference backend)
    # ==============================================================================
    if isinstance(data, np.ndarray):
        arr = data
        T = arr.shape[0]

        # Not enough observations to form a single window
        if T < sequence_length:
            return torch.empty(0), 0

        X_list = []
        y_list = []

        # ----------------------------------------------------------------------
        # TRAINING MODE (Build ALL Overlapping Sliding Windows)
        # ----------------------------------------------------------------------
        if training_mode:
            # Number of windows = T - L - horizon + 1
            num_sequences = T - sequence_length - prediction_horizon + 1
            if num_sequences <= 0:
                return torch.empty(0), 0

            for start in range(0, num_sequences, step):
                end = start + sequence_length
                X_list.append(arr[start:end])

                # Optional target labeling (requires feature_columns for ndarray)
                if target is not None:
                    if feature_columns is None:
                        raise ValueError("feature_columns must be provided for ndarray+target mode.")

                    label_idx = end + prediction_horizon - 1
                    y_list.append(arr[label_idx, feature_columns.index(target)])

            X = np.stack(X_list).astype("float32")
            X_tensor = torch.from_numpy(X)

            if target is not None:
                y = np.array(y_list, dtype="float32")
                return X_tensor, torch.from_numpy(y)

            return X_tensor, 0     # training → no start index needed

        # ----------------------------------------------------------------------
        # INFERENCE MODE (Only the Final, Most Recent Window)
        # ----------------------------------------------------------------------
        start_index = T - sequence_length
        final_seq = arr[start_index:T]
        X = np.expand_dims(final_seq, axis=0).astype("float32")
        X_tensor = torch.from_numpy(X)

        if target is not None:
            # Label corresponds to future horizon
            label_idx = T - 1 + prediction_horizon
            if label_idx >= T:
                return torch.empty(0), 0
            if feature_columns is None:
                raise ValueError("feature_columns must be provided")
            y_val = arr[label_idx, feature_columns.index(target)]
            return X_tensor, torch.tensor([float(y_val)], dtype=torch.float32)

        return X_tensor, start_index

    # ==============================================================================
    # CASE B — DATAFRAME INPUT (Used for supervised pretraining & multi-symbol inputs)
    # ==============================================================================
    elif isinstance(data, pd.DataFrame):
        df = data.copy()

        if feature_columns is None:
            raise ValueError("feature_columns must be provided when using a DataFrame.")

        for c in feature_columns:
            if c not in df.columns:
                raise ValueError(f"Feature column '{c}' missing from DataFrame.")

        # If multi-symbol dataset → group by symbol
        groups = [df]
        if symbol_col and symbol_col in df.columns:
            groups = [g.sort_index() for _, g in df.groupby(symbol_col)]

        X_windows, y_windows = [], []

        # Process each symbol independently
        for g in groups:
            g = g.sort_index()
            values = g[feature_columns].values
            T = values.shape[0]

            if T < sequence_length:
                continue

            # --------------------------------------------------------------
            # TRAINING MODE: Sliding Windows for All Sequences
            # --------------------------------------------------------------
            if training_mode:
                num_sequences = T - sequence_length - prediction_horizon + 1
                if num_sequences <= 0:
                    continue

                for start in range(0, num_sequences, step):
                    end = start + sequence_length
                    label_idx = end + prediction_horizon - 1
                    if label_idx >= T:
                        break

                    X_windows.append(values[start:end])
                    if target is not None:
                        y_windows.append(g.iloc[label_idx][target])

            else:
                # ----------------------------------------------------------
                # INFERENCE MODE: Only Final Window
                # ----------------------------------------------------------
                start_index = T - sequence_length
                final_seq = values[start_index:T]
                X_windows.append(final_seq)

                if target is not None:
                    label_idx = start_index + sequence_length - 1 + prediction_horizon
                    if label_idx < T:
                        y_windows.append(g.iloc[label_idx][target])
                    else:
                        y_windows.append(np.nan)

        # No windows extracted → return empty tensor
        if len(X_windows) == 0:
            return torch.empty(0), 0

        X = np.stack(X_windows).astype("float32")
        X_tensor = torch.from_numpy(X)

        # If labels requested → return (X, y)
        if target is not None:
            y = np.array(y_windows)
            valid_mask = ~np.isnan(y.astype("float64"))
            if valid_mask.sum() == 0:
                return torch.empty(0), 0
            return X_tensor[valid_mask], torch.from_numpy(y[valid_mask].astype("float32"))

        # Inference with multi-symbol dataset → ambiguous start index
        if not training_mode and symbol_col and symbol_col in df.columns:
            return X_tensor, -1

        # Single symbol inference → return final window alignment index
        if not training_mode:
            return X_tensor, df.shape[0] - sequence_length

        # Training mode without target
        return X_tensor, 0

    else:
        raise TypeError("data must be a numpy.ndarray or pandas.DataFrame")


# ----------------------------------------------------------------------------------
# NOTE: The actual GRU model definition will go into backend/models.py
# ----------------------------------------------------------------------------------
