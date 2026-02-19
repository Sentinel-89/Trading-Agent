#!/usr/bin/env python3
"""
Plot entropy curves for:
  1) action masking run
  2) no-masking run (with invalid-action penalty)
and overlay invalid actions (no-masking) on a secondary axis.

Usage:
  python plot_entropy_invalid.py \
      --masked path/to/masked.json \
      --nomask path/to/no_mask.json \
      --out entropy_invalid.png \
      --window 25

Notes:
- Expects JSON format like:
  {"phase": "...", "seed": ..., "updates": [{"ppo_update": 0, "entropy": ..., "invalid_actions": ...}, ...]}
- If your JSON is a list of such dicts, the script will try to pick the first valid entry.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_log(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # Sometimes logs are stored as a list of runs
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "updates" in item and isinstance(item["updates"], list):
                return item
        raise ValueError(f"{path}: JSON is a list but no element looks like a run dict with 'updates'.")

    if not (isinstance(obj, dict) and "updates" in obj and isinstance(obj["updates"], list)):
        raise ValueError(f"{path}: JSON must be a dict with an 'updates' list (or a list of such dicts).")

    return obj


def extract_df(run: dict, *, require_invalid_actions: bool = False) -> pd.DataFrame:
    updates = run.get("updates", [])
    if not updates:
        raise ValueError("Run has empty 'updates'.")

    df = pd.DataFrame(updates)

    # Basic required columns
    for col in ("ppo_update", "entropy"):
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in updates.")

    if require_invalid_actions and "invalid_actions" not in df.columns:
        raise ValueError("Missing 'invalid_actions' in updates for no-masking run.")

    # Keep only what we need; coerce numeric
    keep_cols = ["ppo_update", "entropy"] + (["invalid_actions"] if "invalid_actions" in df.columns else [])
    df = df[keep_cols].copy()

    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ppo_update", "entropy"]).sort_values("ppo_update").reset_index(drop=True)
    df["ppo_update"] = df["ppo_update"].astype(int)

    # invalid_actions might be missing or NaN for masked run; keep as-is if present
    if "invalid_actions" in df.columns:
        df["invalid_actions"] = df["invalid_actions"].fillna(0)

    return df


def maybe_smooth(series: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked", type=Path, required=True, help="JSON log for action-masking run")
    ap.add_argument("--nomask", type=Path, required=True, help="JSON log for no-masking run (has invalid_actions)")
    ap.add_argument("--out", type=Path, default=Path("entropy_invalid.png"), help="Output image path")
    ap.add_argument("--window", type=int, default=1, help="Rolling-mean window for smoothing (1 = none)")
    ap.add_argument("--title", type=str, default="Entropy + Invalid Actions vs PPO Update")
    ap.add_argument(
        "--exclude_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Exclude PPO updates in [START, END] (inclusive) from plotting",
    )
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    args = ap.parse_args()

    run_masked = load_log(args.masked)
    run_nomask = load_log(args.nomask)

    df_m = extract_df(run_masked, require_invalid_actions=False)
    df_n = extract_df(run_nomask, require_invalid_actions=True)

    # Optionally exclude an update range (inclusive)
    if args.exclude_range is not None:
        start, end = args.exclude_range
        if start > end:
            start, end = end, start
        df_m = df_m[(df_m["ppo_update"] < start) | (df_m["ppo_update"] > end)].reset_index(drop=True)
        df_n = df_n[(df_n["ppo_update"] < start) | (df_n["ppo_update"] > end)].reset_index(drop=True)

    # Smooth if requested
    df_m["entropy_s"] = maybe_smooth(df_m["entropy"], args.window)
    df_n["entropy_s"] = maybe_smooth(df_n["entropy"], args.window)
    df_n["invalid_s"] = maybe_smooth(df_n["invalid_actions"], args.window)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Entropy lines
    l1 = ax1.plot(df_m["ppo_update"], df_m["entropy_s"], label="Entropy (masking)")
    l2 = ax1.plot(df_n["ppo_update"], df_n["entropy_s"], label="Entropy (no masking)")

    ax1.set_xlabel("PPO update")
    ax1.set_ylabel("Entropy")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    # Invalid actions on secondary axis (vertical bars)
    ax2 = ax1.twinx()

    # Choose a sensible bar width based on update spacing
    if len(df_n) >= 2:
        diffs = df_n["ppo_update"].diff().dropna()
        if len(diffs) and len(diffs.mode()):
            step = int(diffs.mode().iloc[0])
        else:
            step = int(diffs.median()) if len(diffs) else 1
        bar_width = max(1, int(0.8 * step))
    else:
        bar_width = 1

    bars = ax2.bar(
        df_n["ppo_update"],
        df_n["invalid_s"],
        width=bar_width,
        label="Invalid actions (no masking)",
        alpha=0.25,
        align="center",
    )
    ax2.set_ylabel("Invalid actions")

    # Combined legend (lines + bars)
    handles = l1 + l2 + [bars]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="best")

    ax1.set_title(args.title)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved: {args.out.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()