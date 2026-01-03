# backend/models/data_split.py
"""
Temporal dataset splitting utilities
=====================================

This module provides helpers to split labeled financial datasets
into TRAIN and VALIDATION sets using *time-aware* splits.

IMPORTANT DESIGN CHOICES
-----------------------------------------------------
• Splits are deterministic (no shuffling)
• Splitting occurs at the SYMBOL / CSV level
• Each CSV is assumed to already be time-ordered
• Prevents look-ahead bias and data leakage
• Intended exclusively for supervised GRU pretraining
"""

from typing import List, Tuple


def temporal_train_val_split(
    csv_paths: List[str],
    val_ratio: float = 0.2
) -> Tuple[List[str], List[str]]:
    """
    Splits a list of CSV paths into train and validation sets.

    Parameters
    ----------
    csv_paths : List[str]
        Paths to labeled CSV files (one per symbol)

    val_ratio : float
        Fraction of CSVs assigned to validation

    Returns
    -------
    train_csvs : List[str]
    val_csvs   : List[str]
    """

    if not csv_paths:
        raise ValueError("csv_paths must be a non-empty list")

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    # Deterministic ordering
    csv_paths = sorted(csv_paths)

    split_idx = int(len(csv_paths) * (1 - val_ratio))

    train_csvs = csv_paths[:split_idx]
    val_csvs = csv_paths[split_idx:]

    return train_csvs, val_csvs
