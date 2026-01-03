# backend/utils/labeling.py
"""
Label generation utilities for supervised pretraining of the GRU encoder.

This module produces TWO forward-looking labels:

-----------------------------------------------------------------------------------
1. TREND LABEL (classification: up / down / flat)
-----------------------------------------------------------------------------------
Predicts the next-day direction of price movement.

    trend_label[t] = +1  (future price rises)
                       0  (flat within threshold)
                      -1  (future price falls)

The GRU sees the previous 30 days (or chosen window),
but the label represents what happens *after* that window.

-----------------------------------------------------------------------------------
2. EXPECTED RETURN LABEL (regression: magnitude)
-----------------------------------------------------------------------------------
Predicts the multi-day future return, default horizon: 3 days.

    expected_return[t] = (Close[t + 3] / Close[t]) - 1

This teaches the GRU how strong the move is, not just its direction.

-----------------------------------------------------------------------------------
IMPORTANT DESIGN DECISION
-----------------------------------------------------------------------------------
We do NOT include volatility regime as a label.
Realized volatility is backward-looking by definition and is better treated
as a FEATURE inside preprocessing.py, not as a supervised target.

This results in:
  • cleaner gradients,
  • faster convergence,
  • better generalization,
  • simpler multi-task learning.

"""

import pandas as pd
import numpy as np

# =====================================================================================
# Default configuration
# =====================================================================================

DEFAULT_TREND_HORIZON = 1          # predict next-day direction
DEFAULT_RETURN_HORIZON = 3         # predict 3-day future return
TREND_FLAT_THRESHOLD = 0.001       # ±0.1% is considered flat


# =====================================================================================
# TREND LABEL (classification)
# =====================================================================================

def compute_trend_label(
    close: pd.Series,
    horizon: int = DEFAULT_TREND_HORIZON,
    flat_threshold: float = TREND_FLAT_THRESHOLD
) -> pd.Series:
    """
    Computes the forward-looking trend direction.

    label[t] = +1 if price rises by more than +flat_threshold
               -1 if price falls by more than -flat_threshold
                0 otherwise

    This is a CLEAN and STABLE supervised signal for pretraining.
    """

    future = close.shift(-horizon)
    pct_change = (future - close) / close

    labels = np.where(
        pct_change > flat_threshold,  +1,
        np.where(pct_change < -flat_threshold, -1, 0)
    )

    return pd.Series(labels, index=close.index)


# =====================================================================================
# EXPECTED RETURN LABEL (regression)
# =====================================================================================

def compute_expected_return(
    close: pd.Series,
    horizon: int = DEFAULT_RETURN_HORIZON
) -> pd.Series:
    """
    Computes forward 3-day (or configurable) expected return:

        expected_return[t] = (Close[t+h] / Close[t]) - 1

    A smooth regression target that complements the trend classification.
    """

    future = close.shift(-horizon)
    expected_ret = (future / close) - 1
    return expected_ret


# =====================================================================================
# MAIN LABEL ASSEMBLY FUNCTION
# =====================================================================================

def add_all_labels(
    df: pd.DataFrame,
    trend_horizon: int = DEFAULT_TREND_HORIZON,
    return_horizon: int = DEFAULT_RETURN_HORIZON
) -> pd.DataFrame:
    """
    Adds TrendLabel and ExpectedReturn to the dataset.

    Inputs:
        df must contain a 'Close' column.

    Outputs:
        df with:
            - TrendLabel        (classification)
            - ExpectedReturn    (regression)

    NOTE:
        Volatility regime is intentionally NOT included as a label.
        It should be computed as a FEATURE inside preprocessing.py.
    """

    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    labeled = df.copy()

    labeled["TrendLabel"] = compute_trend_label(
        labeled["Close"],
        horizon=trend_horizon
    )

    labeled["ExpectedReturn"] = compute_expected_return(
        labeled["Close"],
        horizon=return_horizon
    )

    return labeled
