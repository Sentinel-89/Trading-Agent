"""Aggregate per-checkpoint evaluation JSONs to CSV tables.

Input: directory containing JSON files produced by `evaluation_runner_all.py`
(each includes {summary, per_symbol}).

Outputs:
- eval_per_checkpoint.csv
- eval_per_symbol.csv

Keep it separate from evaluation code so scripts stay small.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List


import pandas as pd


# --- Helper functions for robust payload normalization and summary computation ---

def _safe_mean(vals: List[Any]) -> float | None:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _compute_summary_from_per_symbol(per_symbol: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rets = [m.get("return") for m in per_symbol.values()]
    sharpes = [m.get("sharpe") for m in per_symbol.values()]
    mdds = [m.get("max_drawdown") for m in per_symbol.values()]
    excess = [m.get("excess_return_vs_bh") for m in per_symbol.values()]

    bh_rets: List[Any] = []
    bh_sharpes: List[Any] = []
    bh_mdds: List[Any] = []

    buys: List[int] = []
    sells: List[int] = []
    holds: List[int] = []

    for m in per_symbol.values():
        bh = m.get("buy_hold") or {}
        bh_rets.append(bh.get("return"))
        bh_sharpes.append(bh.get("sharpe"))
        bh_mdds.append(bh.get("max_drawdown"))

        tr = m.get("trades") or {}
        if isinstance(tr, dict):
            if tr.get("buys") is not None:
                buys.append(int(tr.get("buys")))
            if tr.get("sells") is not None:
                sells.append(int(tr.get("sells")))
            if tr.get("holds") is not None:
                holds.append(int(tr.get("holds")))

    return {
        "mean_return": _safe_mean(rets),
        "mean_sharpe": _safe_mean(sharpes),
        "mean_max_drawdown": _safe_mean(mdds),
        "mean_excess_return_vs_bh": _safe_mean(excess),
        "mean_buy_hold_return": _safe_mean(bh_rets),
        "mean_buy_hold_sharpe": _safe_mean(bh_sharpes),
        "mean_buy_hold_max_drawdown": _safe_mean(bh_mdds),
        "total_buys": int(sum(buys)) if buys else None,
        "total_sells": int(sum(sells)) if sells else None,
        "total_holds": int(sum(holds)) if holds else None,
        "n_symbols": int(len(per_symbol)),
    }


def _normalize_payload(payload: Dict[str, Any], pth: str) -> tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return (checkpoint_name, summary_dict, per_symbol_dict) for multiple json formats.

    Supported:
      A) {"checkpoint": ..., "summary": {...}, "per_symbol": {...}}
      B) {"summary": {...}, "per_symbol": {...}} (no checkpoint field)
      C) {"SYM": {...}, "SYM2": {...}}  (per-symbol-only json; discrete eval output)
    """
    ckpt = payload.get("checkpoint") or os.path.basename(pth)

    if isinstance(payload.get("per_symbol"), dict):
        per_symbol = payload.get("per_symbol") or {}
        summary = payload.get("summary") or {}
        if not summary:
            summary = _compute_summary_from_per_symbol(per_symbol)
        return ckpt, summary, per_symbol

    # Per-symbol-only format: top-level keys are symbols.
    if payload and all(isinstance(v, dict) for v in payload.values()):
        per_symbol = payload  # type: ignore[assignment]
        summary = _compute_summary_from_per_symbol(per_symbol)
        return ckpt, summary, per_symbol

    # Unknown format
    return ckpt, {}, {}


def load_eval_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(eval_dir: str) -> None:
    paths = sorted(glob.glob(os.path.join(eval_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {eval_dir}")

    per_ckpt_rows: List[Dict[str, Any]] = []
    per_symbol_rows: List[Dict[str, Any]] = []

    for pth in paths:
        payload = load_eval_json(pth)
        ckpt, summ, per_symbol = _normalize_payload(payload, pth)

        # Skip files that are not in an eval-json format we understand
        if not per_symbol:
            continue

        per_ckpt_rows.append({
            "checkpoint": ckpt,
            "eval_file": os.path.basename(pth),
            **(summ or {}),
        })

        for sym, m in (per_symbol or {}).items():
            bh = m.get("buy_hold") or {}
            tr = m.get("trades") or {}

            row = {
                "checkpoint": ckpt,
                "symbol": sym,
                "return": m.get("return"),
                "sharpe": m.get("sharpe"),
                "max_drawdown": m.get("max_drawdown"),
                "excess_return_vs_bh": m.get("excess_return_vs_bh"),

                # optional continuous-vector fields (if present)
                "action_abs_turnover": m.get("action_abs_turnover"),
                "action_abs_mean": m.get("action_abs_mean"),

                # optional buy&hold block (present in your discrete eval json)
                "buy_hold_return": bh.get("return"),
                "buy_hold_sharpe": bh.get("sharpe"),
                "buy_hold_max_drawdown": bh.get("max_drawdown"),

                # optional trades block (present in your discrete eval json)
                "buys": tr.get("buys") if isinstance(tr, dict) else None,
                "sells": tr.get("sells") if isinstance(tr, dict) else None,
                "holds": tr.get("holds") if isinstance(tr, dict) else None,
            }
            per_symbol_rows.append(row)

    df_ckpt = pd.DataFrame(per_ckpt_rows)
    sort_col = "mean_excess_return_vs_bh" if "mean_excess_return_vs_bh" in df_ckpt.columns else (
        "mean_return" if "mean_return" in df_ckpt.columns else None
    )
    if sort_col:
        df_ckpt = df_ckpt.sort_values(by=[sort_col], ascending=False)
    df_sym = pd.DataFrame(per_symbol_rows)

    if df_ckpt.empty:
        raise ValueError(
            f"No valid evaluation JSONs were parsed from {eval_dir}. "
            f"Expected either {{summary, per_symbol}} or a per-symbol-only JSON (top-level symbols)."
        )

    out_ckpt = os.path.join(eval_dir, "eval_per_checkpoint.csv")
    out_sym = os.path.join(eval_dir, "eval_per_symbol.csv")

    df_ckpt.to_csv(out_ckpt, index=False)
    df_sym.to_csv(out_sym, index=False)

    print(f"[Aggregate] wrote {out_ckpt}")
    print(f"[Aggregate] wrote {out_sym}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, required=True)
    args = ap.parse_args()
    aggregate(args.eval_dir)


if __name__ == "__main__":
    cli()
