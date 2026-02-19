
"""backend/rl/buy_hold_bl.py

Baselines for CONTINUOUS portfolio evaluation JSONs.

Goal
----
Given an evaluation JSON produced by `evaluation_runner_continuous.py`, compute and print
fair baselines so you can compare PPO vs something simple.

Why baselines are tricky for "top-k"
----------------------------------
"Top-k" is a *constraint* (hold only k assets) + often a *rebalancing rule* (daily/weekly).
A plain buy-and-hold baseline can be defined in multiple reasonable ways.

This script provides two common baselines for top-k:

1) FIXED TOP-K BUY & HOLD (apples-to-apples with "topk_*" holdings count)
   - At t=start_t, take the PPO's initial weights, pick the top-k assets.
   - Allocate equal weight across those k assets (optionally include cash).
   - Then HOLD (no rebalancing) until end_t.

   Interpretation: "If I only held the top-k names chosen at the start and did nothing".

2) DYNAMIC TOP-K WEEKLY BASELINE (simple rules-based proxy)
   - Every N steps (weekly), compute last-period returns for each asset.
   - Pick top-k by momentum over the last N steps.
   - Equal-weight them (optionally include cash), then hold for next N steps.

   Interpretation: "A simple top-k momentum strategy".

You can choose which baseline to use depending on what you want to demonstrate.

Usage
-----
Edit EVAL_JSON below (or pass via CLI) and run:

  python backend/rl/buy_hold_bl.py --eval_json <path> --k 5 --weekly_n 5

"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def total_return(curve: List[float]) -> float:
    return float(curve[-1] / curve[0] - 1.0) if len(curve) >= 2 and curve[0] != 0 else 0.0


def max_drawdown(curve: List[float]) -> float:
    if len(curve) < 2:
        return 0.0
    arr = np.asarray(curve, dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    dd = arr / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def sharpe(curve: List[float], periods: int = 252) -> float:
    if len(curve) < 3:
        return 0.0
    arr = np.asarray(curve, dtype=np.float64)
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    return 0.0 if sd < 1e-12 else float((mu / sd) * np.sqrt(periods))


def summarize(name: str, curve: List[float]) -> str:
    return (
        f"{name:16s} final={curve[-1]:.6f} "
        f"ret={total_return(curve)*100:+.3f}% "
        f"mdd={max_drawdown(curve)*100:+.3f}% "
        f"sharpe={sharpe(curve):+.3f}"
    )


def load_aligned_close(
    data_dir: str,
    symbols: List[str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """Load Close prices for symbols and align by date intersection.

    This matches the evaluator's idea: intersection of dates and then reset to 0..N-1.
    """

    dfs = {}
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}_labeled.csv")
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise ValueError(f"{sym}: missing Date column")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df.loc[date_start:date_end]
        if "Close" not in df.columns:
            raise ValueError(f"{sym}: missing Close column")
        dfs[sym] = df[["Close"]].rename(columns={"Close": sym})

    common = None
    for df in dfs.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()

    close = pd.concat([dfs[s].reindex(common) for s in symbols], axis=1).dropna()
    close = close.reset_index(drop=True)
    return close


# ----------------------------
# Baselines
# ----------------------------

def equal_weight_buy_and_hold(
    close: pd.DataFrame,
    symbols: List[str],
    start_t: int,
    end_t: int,
    *,
    include_cash: bool,
) -> List[float]:
    """Equal-weight buy & hold over [start_t, end_t]."""
    if end_t <= start_t:
        return [1.0]

    n = len(symbols)
    if n == 0:
        return [1.0]

    # If include_cash=True, reserve 1/(n+1) in cash (0 return)
    w_assets = 1.0 / float(n + 1) if include_cash else 1.0 / float(n)

    eq = [1.0]
    for t in range(start_t + 1, end_t + 1):
        prev = close.loc[t - 1, symbols].to_numpy(dtype=float)
        cur = close.loc[t, symbols].to_numpy(dtype=float)
        r = (cur / prev) - 1.0
        port_ret = float(np.sum(w_assets * r))
        eq.append(eq[-1] * (1.0 + port_ret))
    return eq


def fixed_topk_buy_and_hold(
    close: pd.DataFrame,
    symbols: List[str],
    start_t: int,
    end_t: int,
    *,
    include_cash: bool,
    k: int,
    ppo_initial_weights: Optional[List[float]],
) -> Tuple[List[float], List[str]]:
    """Pick top-k at the start (using PPO's initial weights) and then hold.

    Returns: (equity_curve, picked_symbols)
    """

    if end_t <= start_t:
        return [1.0], []

    n = len(symbols)
    if n == 0:
        return [1.0], []

    k = int(max(1, min(k, n)))

    if not ppo_initial_weights or len(ppo_initial_weights) < n:
        # Fallback: equal-weight top-k is arbitrary; use first k symbols
        picked = symbols[:k]
    else:
        w = np.asarray(ppo_initial_weights[:n], dtype=float)
        picked_idx = np.argsort(w)[::-1][:k]
        picked = [symbols[i] for i in picked_idx]

    # Equal weight across picked, optional cash bucket
    w_assets = 1.0 / float(k + 1) if include_cash else 1.0 / float(k)

    eq = [1.0]
    for t in range(start_t + 1, end_t + 1):
        prev = close.loc[t - 1, picked].to_numpy(dtype=float)
        cur = close.loc[t, picked].to_numpy(dtype=float)
        r = (cur / prev) - 1.0
        port_ret = float(np.sum(w_assets * r))
        eq.append(eq[-1] * (1.0 + port_ret))

    return eq, picked


def momentum_topk_weekly(
    close: pd.DataFrame,
    symbols: List[str],
    start_t: int,
    end_t: int,
    *,
    include_cash: bool,
    k: int,
    weekly_n: int,
) -> Tuple[List[float], List[List[str]]]:
    """Simple top-k momentum baseline with weekly rebalancing.

    Every `weekly_n` steps, rank assets by return over the previous `weekly_n` steps
    (or from start_t on the first rebalance), select top-k, equal-weight, hold until
    the next rebalance.

    Returns: (equity_curve, holdings_per_rebalance)
    """

    n = len(symbols)
    if n == 0 or end_t <= start_t:
        return [1.0], []

    k = int(max(1, min(k, n)))
    weekly_n = int(max(1, weekly_n))

    eq = [1.0]
    holdings: List[List[str]] = []

    # Start by picking momentum based on last weekly_n steps ending at start_t.
    # If insufficient history, use a shorter lookback.
    t = start_t

    while t < end_t:
        lookback = min(weekly_n, t - start_t)  # 0 on first block
        if lookback <= 0:
            # no history yet; default to first k symbols
            picked = symbols[:k]
        else:
            prev = close.loc[t - lookback, symbols].to_numpy(dtype=float)
            cur = close.loc[t, symbols].to_numpy(dtype=float)
            mom = (cur / prev) - 1.0
            picked_idx = np.argsort(mom)[::-1][:k]
            picked = [symbols[i] for i in picked_idx]

        holdings.append(picked)

        # Hold picked for the next block
        block_end = min(end_t, t + weekly_n)

        # equal-weight across picked, optional cash
        w_assets = 1.0 / float(k + 1) if include_cash else 1.0 / float(k)

        for step in range(t + 1, block_end + 1):
            prev = close.loc[step - 1, picked].to_numpy(dtype=float)
            cur = close.loc[step, picked].to_numpy(dtype=float)
            r = (cur / prev) - 1.0
            port_ret = float(np.sum(w_assets * r))
            eq.append(eq[-1] * (1.0 + port_ret))

        t = block_end

    return eq, holdings


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", default="backend/artifacts/phase_d_continuous/seed_1/eval/ppo_step_00799.json")
    parser.add_argument("--data_dir", default="backend/data/processed")
    parser.add_argument("--date_start", default="2024-06-01")
    parser.add_argument("--date_end", default="2025-12-30")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--weekly_n", type=int, default=5)

    args = parser.parse_args()

    r = json.load(open(args.eval_json))

    # Use one scenario to get episode selection
    ep = r["scenarios"]["full_daily"]["episode"]
    symbols = list(ep["symbols"])
    start_t = int(ep["start_t"])
    end_t = int(ep["end_t"])

    include_cash = bool(r.get("env_cfg", {}).get("include_cash", True))

    close = load_aligned_close(args.data_dir, symbols, args.date_start, args.date_end)

    # PPO curves available in JSON
    ppo_full_weekly = r["scenarios"]["full_weekly"]["equity_curve"]
    ppo_topk_weekly = r["scenarios"]["topk_weekly"]["equity_curve"]
    ppo_topk_daily = r["scenarios"]["topk_daily"]["equity_curve"]
    ppo_full_daily = r["scenarios"].get("full_daily", {}).get("equity_curve")

    print(f"Eval JSON: {args.eval_json}")
    print(f"Basket (n={len(symbols)}): {symbols}")
    print(f"Episode index range: start_t={start_t} end_t={end_t} steps={end_t-start_t}")
    print(f"include_cash={include_cash}  k={args.k}  weekly_n={args.weekly_n}")
    print()

    # Baseline 1: equal-weight buy&hold across ALL assets
    bh_all = equal_weight_buy_and_hold(close, symbols, start_t, end_t, include_cash=include_cash)

    # Baseline 2: fixed top-k buy&hold (pick top-k from PPO initial weights at episode start)
    # Note: evaluator JSON currently does not store per-step weights; if you add trace later,
    # you can pass the true initial weights. For now we try to use info if present.
    ppo_init_w = None
    # Some env configs store weights in reset info; if your JSON ever includes it, use it.
    # Otherwise we fall back to a deterministic choice.
    # (You can extend evaluation_runner to store initial weights in JSON if you want.)

    bh_topk_fixed, picked = fixed_topk_buy_and_hold(
        close,
        symbols,
        start_t,
        end_t,
        include_cash=include_cash,
        k=args.k,
        ppo_initial_weights=ppo_init_w,
    )

    # Baseline 3: momentum top-k weekly (simple rules-based)
    bh_topk_mom_weekly, holdings = momentum_topk_weekly(
        close,
        symbols,
        start_t,
        end_t,
        include_cash=include_cash,
        k=args.k,
        weekly_n=args.weekly_n,
    )

    # Print comparisons
    if ppo_full_daily is not None:
        print(summarize("PPO full_daily", ppo_full_daily))
    print(summarize("PPO full_weekly", ppo_full_weekly))
    print(summarize("PPO topk_daily", ppo_topk_daily))
    print(summarize("PPO topk_weekly", ppo_topk_weekly))
    print()

    print(summarize("BH all_hold", bh_all))
    print(summarize("BH topk_fixed", bh_topk_fixed), "picked=", picked)
    print(summarize("BH topk_mom_wk", bh_topk_mom_weekly), "rebalances=", len(holdings))

    # Differences vs buy&hold baselines
    print("\nDiff vs BH all_hold:")
    print("  full_weekly:", f"{(total_return(ppo_full_weekly) - total_return(bh_all))*100:+.3f}%")
    print("  topk_weekly:", f"{(total_return(ppo_topk_weekly) - total_return(bh_all))*100:+.3f}%")

    print("\nDiff vs BH topk_mom_wk:")
    print("  topk_weekly:", f"{(total_return(ppo_topk_weekly) - total_return(bh_topk_mom_weekly))*100:+.3f}%")


if __name__ == "__main__":
    main()
