import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_updates(path: Path):
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # support either {"updates":[...]} or a raw list of update dicts
    if isinstance(obj, dict) and "updates" in obj and isinstance(obj["updates"], list):
        updates = obj["updates"]
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        updates = obj
    else:
        raise ValueError(f"{path} doesn't look like a list of updates or a dict with 'updates'")

    steps = []
    turnover = []
    entropy = []

    for u in updates:
        if not isinstance(u, dict):
            continue
        if "step" not in u:
            continue
        # Only keep rows where both metrics exist
        if "turnover" not in u or "entropy" not in u:
            continue

        steps.append(int(u["step"]))
        turnover.append(float(u["turnover"]))
        entropy.append(float(u["entropy"]))

    if not steps:
        # helpful message
        sample_keys = sorted(list(updates[0].keys())) if updates and isinstance(updates[0], dict) else []
        raise ValueError(
            f"No (step, turnover, entropy) rows found in {path}. "
            f"First update keys: {sample_keys}"
        )

    order = np.argsort(steps)
    steps = np.array(steps, dtype=int)[order]
    turnover = np.array(turnover, dtype=float)[order]
    entropy = np.array(entropy, dtype=float)[order]

    return steps, turnover, entropy


def rolling_mean(y: np.ndarray, window: int):
    if window <= 1:
        return y
    # simple centered moving average
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")


def apply_drop_range(steps, y, drop_start, drop_end):
    if drop_start is None and drop_end is None:
        return steps, y
    drop_start = -np.inf if drop_start is None else drop_start
    drop_end = np.inf if drop_end is None else drop_end
    keep = ~((steps >= drop_start) & (steps <= drop_end))
    return steps[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True, help="Path to train_log.json (e.g., mask_on)")
    ap.add_argument("--label_a", default="run_a")
    ap.add_argument("--run_b", default=None, help="Optional second train_log.json (e.g., mask_off)")
    ap.add_argument("--label_b", default="run_b")

    ap.add_argument("--window", type=int, default=25, help="Smoothing window (rolling mean). 1 disables.")
    ap.add_argument("--drop_start", type=int, default=None, help="Drop updates in [drop_start, drop_end]")
    ap.add_argument("--drop_end", type=int, default=None)

    ap.add_argument("--out", default="turnover.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # Load A
    a_steps, a_turn, a_ent = load_updates(Path(args.run_a))
    a_steps, a_turn = apply_drop_range(a_steps, a_turn, args.drop_start, args.drop_end)
    # apply same keep-mask to entropy by re-applying drop_range to entropy with the already-filtered steps
    a_steps2, a_ent = apply_drop_range(a_steps, a_ent, args.drop_start, args.drop_end)
    if len(a_steps2) != len(a_steps):
        # safety: if something odd happens, align to common length
        n = min(len(a_steps), len(a_steps2))
        a_steps = a_steps[:n]
        a_turn = a_turn[:n]
        a_ent = a_ent[:n]

    a_turn_s = rolling_mean(a_turn, args.window)
    a_ent_s = rolling_mean(a_ent, args.window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Turnover
    ax1.plot(a_steps, a_turn, alpha=0.25, linewidth=1, label=f"{args.label_a} (raw)")
    ax1.plot(a_steps, a_turn_s, linewidth=2, label=f"{args.label_a} (smooth w={args.window})")
    ax1.set_ylabel("Turnover")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # Entropy
    ax2.plot(a_steps, a_ent, alpha=0.25, linewidth=1, label=f"{args.label_a} (raw)")
    ax2.plot(a_steps, a_ent_s, linewidth=2, label=f"{args.label_a} (smooth w={args.window})")
    ax2.set_ylabel("Entropy")
    ax2.set_xlabel("Update (step)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    # Load B (optional)
    if args.run_b:
        b_steps, b_turn, b_ent = load_updates(Path(args.run_b))
        b_steps, b_turn = apply_drop_range(b_steps, b_turn, args.drop_start, args.drop_end)
        b_steps2, b_ent = apply_drop_range(b_steps, b_ent, args.drop_start, args.drop_end)
        if len(b_steps2) != len(b_steps):
            n = min(len(b_steps), len(b_steps2))
            b_steps = b_steps[:n]
            b_turn = b_turn[:n]
            b_ent = b_ent[:n]

        b_turn_s = rolling_mean(b_turn, args.window)
        b_ent_s = rolling_mean(b_ent, args.window)

        ax1.plot(b_steps, b_turn, alpha=0.25, linewidth=1, label=f"{args.label_b} (raw)")
        ax1.plot(b_steps, b_turn_s, linewidth=2, label=f"{args.label_b} (smooth w={args.window})")

        ax2.plot(b_steps, b_ent, alpha=0.25, linewidth=1, label=f"{args.label_b} (raw)")
        ax2.plot(b_steps, b_ent_s, linewidth=2, label=f"{args.label_b} (smooth w={args.window})")

    fig.suptitle("Turnover and Entropy vs Training Update")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    if args.show:
        plt.show()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()