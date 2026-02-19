#!/usr/bin/env python3
"""
Summarize evaluation JSONs produced by evaluation_runner_continuous.py.

Your eval JSONs look like:
{
  "checkpoint": "...",
  "summary": {"mean_total_return": ..., "mean_sharpe": ..., ...},
  "per_episode": [ {"episode": 0, "metrics": {"total_return": ...}}, ...]
}

This script:
- reads all matching JSON files in an eval directory
- extracts important summary metrics
- recomputes p10/median/p90 from per_episode (if present)
- prints a compact leaderboard and saves CSV
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _percentiles(values: List[float], ps: Tuple[int, int, int] = (10, 50, 90)) -> Dict[str, float]:
    if not values:
        return {}
    v = sorted(values)

    def pct(p: int) -> float:
        if len(v) == 1:
            return float(v[0])
        k = (len(v) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(v) - 1)
        if f == c:
            return float(v[f])
        return float(v[f] + (v[c] - v[f]) * (k - f))

    out: Dict[str, float] = {}
    for p in ps:
        label = "median" if p == 50 else f"p{p}"
        out[label] = pct(p)
    return out


@dataclass
class Row:
    filename: str
    checkpoint: str
    episodes: int
    window_length: Optional[int]
    episode_length: Optional[int]
    eval_deterministic: Optional[bool]

    mean_total_return: Optional[float]
    mean_sharpe: Optional[float]
    mean_max_drawdown: Optional[float]
    mean_baseline_total_return: Optional[float]
    mean_excess_total_return: Optional[float]
    mean_turnover: Optional[float]
    mean_rebalances: Optional[float]

    tr_p10: Optional[float] = None
    tr_median: Optional[float] = None
    tr_p90: Optional[float] = None
    sh_p10: Optional[float] = None
    sh_median: Optional[float] = None
    sh_p90: Optional[float] = None


def _extract_episode_lists(per_episode: Any) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"total_return": [], "sharpe": [], "max_drawdown": []}
    if not isinstance(per_episode, list):
        return out

    for ep in per_episode:
        if not isinstance(ep, dict):
            continue
        m = _safe_get(ep, "metrics", {})
        # some files may have "return" instead of "total_return"
        tr = _as_float(_safe_get(m, "total_return", _safe_get(m, "return")))
        sh = _as_float(_safe_get(m, "sharpe"))
        md = _as_float(_safe_get(m, "max_drawdown"))

        if tr is not None:
            out["total_return"].append(tr)
        if sh is not None:
            out["sharpe"].append(sh)
        if md is not None:
            out["max_drawdown"].append(md)

    return out


def _load_one(path: str, debug: bool = False) -> Optional[Row]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return None

    if not isinstance(data, dict):
        print(f"[WARN] JSON root not dict: {path}")
        return None

    summary = _safe_get(data, "summary", {})
    per_episode = _safe_get(data, "per_episode", None)

    if debug:
        print(f"[DEBUG] {os.path.basename(path)} top_keys={list(data.keys())}")
        print(f"[DEBUG] {os.path.basename(path)} summary_keys={list(summary.keys()) if isinstance(summary, dict) else []}")
        print(f"[DEBUG] {os.path.basename(path)} per_episode_len={len(per_episode) if isinstance(per_episode, list) else 0}")

    env_cfg = _safe_get(summary, "env_cfg", {})
    episodes = _safe_get(summary, "episodes")
    if episodes is None and isinstance(per_episode, list):
        episodes = len(per_episode)
    episodes = int(episodes) if episodes is not None else 0

    r = Row(
        filename=os.path.basename(path),
        checkpoint=str(_safe_get(data, "checkpoint", "")),
        episodes=episodes,
        window_length=_safe_get(env_cfg, "window_length"),
        episode_length=_safe_get(env_cfg, "episode_length"),
        eval_deterministic=_safe_get(env_cfg, "eval_deterministic"),

        mean_total_return=_as_float(_safe_get(summary, "mean_total_return")),
        mean_sharpe=_as_float(_safe_get(summary, "mean_sharpe")),
        mean_max_drawdown=_as_float(_safe_get(summary, "mean_max_drawdown")),
        mean_baseline_total_return=_as_float(_safe_get(summary, "mean_baseline_total_return")),
        mean_excess_total_return=_as_float(_safe_get(summary, "mean_excess_total_return")),
        mean_turnover=_as_float(_safe_get(summary, "mean_turnover")),
        mean_rebalances=_as_float(_safe_get(summary, "mean_rebalances")),
    )

    ep_lists = _extract_episode_lists(per_episode)
    tr_ps = _percentiles(ep_lists["total_return"], (10, 50, 90))
    sh_ps = _percentiles(ep_lists["sharpe"], (10, 50, 90))

    r.tr_p10, r.tr_median, r.tr_p90 = tr_ps.get("p10"), tr_ps.get("median"), tr_ps.get("p90")
    r.sh_p10, r.sh_median, r.sh_p90 = sh_ps.get("p10"), sh_ps.get("median"), sh_ps.get("p90")
    return r


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument(
        "--sort",
        default="mean_excess_total_return",
        help="mean_excess_total_return | mean_sharpe | mean_total_return | mean_max_drawdown",
    )
    ap.add_argument("--descending", action="store_true")
    ap.add_argument("--csv_out", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.eval_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(args.eval_dir, args.pattern)}")

    rows: List[Row] = []
    for p in paths:
        r = _load_one(p, debug=args.debug)
        if r is None:
            continue
        # Skip junk files that have neither per_episode nor means
        if r.episodes == 0 and r.mean_total_return is None and r.mean_sharpe is None:
            print(f"[WARN] no usable data in {p}")
            continue
        rows.append(r)

    if not rows:
        raise SystemExit("No usable eval JSONs found.")

    sort_key = args.sort

    def key_fn(r: Row) -> float:
        v = getattr(r, sort_key, None)
        if v is None:
            return float("-inf") if args.descending else float("inf")
        return float(v)

    rows.sort(key=key_fn, reverse=args.descending)

    header = [
        "file", "ckpt", "eps", "wl", "el", "det",
        "mean_ret", "mean_sh", "mean_mdd", "mean_excess",
        "ret_p10", "ret_med", "ret_p90",
        "sh_p10", "sh_med", "sh_p90",
    ]

    print("\t".join(header))
    for r in rows:
        print("\t".join([
            r.filename,
            r.checkpoint,
            str(r.episodes),
            "" if r.window_length is None else str(r.window_length),
            "" if r.episode_length is None else str(r.episode_length),
            "" if r.eval_deterministic is None else str(bool(r.eval_deterministic)),
            _fmt(r.mean_total_return),
            _fmt(r.mean_sharpe),
            _fmt(r.mean_max_drawdown),
            _fmt(r.mean_excess_total_return),
            _fmt(r.tr_p10),
            _fmt(r.tr_median),
            _fmt(r.tr_p90),
            _fmt(r.sh_p10),
            _fmt(r.sh_median),
            _fmt(r.sh_p90),
        ]))

    csv_path = args.csv_out or os.path.join(args.eval_dir, "leaderboard.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.filename, r.checkpoint, r.episodes, r.window_length, r.episode_length, r.eval_deterministic,
                r.mean_total_return, r.mean_sharpe, r.mean_max_drawdown, r.mean_excess_total_return,
                r.tr_p10, r.tr_median, r.tr_p90, r.sh_p10, r.sh_median, r.sh_p90,
            ])
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()