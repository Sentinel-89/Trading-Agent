"""
backend/rl_inference.py

Deterministic inference and evaluation utilities for continuous PPO checkpoints.

- No validation/ranking.
- Dates come from CLI (front-end can pass start/end).
- Produces:
  - One JSON with 4 scenario curves + metrics
  - Up to 4 plots (BH vs PPO) aligned to the same episode start/end

Usage:
  python backend/rl_evaluation/eval_one_checkpoint_continuous.py \
    --ckpt path/to/checkpoint.pt \
    --start 2025-01-01 \
    --end 2025-12-30 \
    --out_dir backend/artifacts/phase_d_continuous/eval_single \
    --top_k 5 --weekly_n 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.rl_evaluation.evaluate_utils import compute_metrics, canonicalize_metrics
from backend.rl.multiasset.ppo_actor_critic_continuous import PPOActorCriticContinuous
from backend.rl.multiasset.trading_env_continuous import TradingEnvContinuous, TradingEnvContinuousConfig

# LIVE Kite data helpers
from pathlib import Path
from fastapi import HTTPException

from backend.services.kite_service import fetch_historical_data
from backend.features.technical_features import calculate_technical_features


MIN_COMMON_ROWS_DEFAULT = 32


# -------------------------
# JSON helpers
# -------------------------
# Serialize numpy/torch types and persist outputs atomically.
def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        if o.ndim == 0:
            return o.item()
        return o.detach().cpu().tolist()
    return str(o)


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


# -------------------------
# Data loading
# -------------------------
# Load local processed datasets, preserve raw prices, and align symbols on common dates.
def load_and_prepare_data(
    *,
    data_dir: str,
    scaler_path: str,
    feature_cols: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """Load *_labeled.csv files, slice date range, preserve raw prices, scale features.
    Returns dict(symbol -> df) with equal length across symbols (aligned intersection).
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    scaler = joblib.load(scaler_path)
    data: Dict[str, pd.DataFrame] = {}

    for fname in os.listdir(data_dir):
        if not fname.endswith("_labeled.csv"):
            continue

        sym = fname.replace("_labeled.csv", "")
        df = pd.read_csv(os.path.join(data_dir, fname))

        # Parse / slice dates
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].reset_index(drop=True)
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.loc[start_date:end_date]

        # Preserve raw prices for returns/baselines
        if "Close" not in df.columns:
            raise KeyError(f"Symbol {sym}: missing Close column")
        df["Close_raw"] = df["Close"].astype(float)
        if "Open" in df.columns:
            df["Open_raw"] = df["Open"].astype(float)

        # Check features exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Symbol {sym}: missing feature columns: {missing}")

        # Scale ONLY feature columns
        df.loc[:, feature_cols] = scaler.transform(df.loc[:, feature_cols].to_numpy())
        data[sym] = df

    if not data:
        raise ValueError(f"No *_labeled.csv files found in {data_dir}")

    # Align by common dates (intersection)
    common_idx = None
    for df in data.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) < MIN_COMMON_ROWS_DEFAULT:
        raise ValueError(
            f"Not enough common dates after alignment: n={0 if common_idx is None else len(common_idx)} "
            f"(min required {MIN_COMMON_ROWS_DEFAULT})."
        )

    common_idx = common_idx.sort_values()

    # Reindex to intersection; drop NAs; keep Date column; reset to 0..N-1
    for sym in list(data.keys()):
        tmp = data[sym].reindex(common_idx).dropna().reset_index()
        # After reset_index(), date values are expected in a 'Date' column.
        # Normalize fallback 'index' naming to 'Date' when needed.
        if "Date" not in tmp.columns and "index" in tmp.columns:
            tmp = tmp.rename(columns={"index": "Date"})
        data[sym] = tmp

    # Enforce equal-length across symbols
    lengths = {sym: len(df) for sym, df in data.items()}
    min_len = min(lengths.values()) if lengths else 0
    if min_len < MIN_COMMON_ROWS_DEFAULT:
        raise ValueError(
            f"Not enough common rows after alignment: min_len={min_len} (min required {MIN_COMMON_ROWS_DEFAULT})."
        )
    if len(set(lengths.values())) != 1:
        print(f"[DataAlign] Unequal lengths; truncating all to min_len={min_len}.")
        for sym in list(data.keys()):
            data[sym] = data[sym].iloc[:min_len].reset_index(drop=True)

    return data


# -------------------------
# Baseline: BH aligned to episode range
# -------------------------
# Build equal-weight buy-and-hold baseline over the same episode indices.

def equal_weight_buy_hold_curve_aligned(
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    *,
    start_t: int,
    end_t: int,
    start_equity: float = 1.0,
) -> List[float]:
    """Equal-weight buy&hold across symbols, using Close_raw, aligned to [start_t..end_t]."""
    if not symbols:
        return [start_equity]
    start_t = max(0, int(start_t))
    end_t = max(start_t, int(end_t))

    closes = [np.asarray(data_by_symbol[s]["Close_raw"].values, dtype=np.float64) for s in symbols]
    T = min(len(c) for c in closes)
    end_t = min(end_t, T - 1)
    if end_t - start_t < 1:
        return [start_equity]

    n = len(symbols)
    w = np.ones(n, dtype=np.float64) / float(n)

    equity = float(start_equity)
    curve = [equity]
    for t in range(start_t + 1, end_t + 1):
        rets = np.array([(c[t] / max(c[t - 1], 1e-12)) - 1.0 for c in closes], dtype=np.float64)
        equity *= (1.0 + float(np.dot(w, rets)))
        curve.append(float(equity))
    return curve


# -------------------------
# Baseline: Momentum Top-K fixed buy&hold aligned to episode range
# -------------------------
# Build fixed top-k momentum baseline selected at episode start.
def momentum_topk_fixed_buy_hold_curve_aligned(
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    *,
    start_t: int,
    end_t: int,
    lookback: int,
    top_k: int,
    start_equity: float = 1.0,
) -> List[float]:
    """Momentum Top-K fixed selection at start_t, then equal-weight hold to end_t.

    momentum(s) = Close_raw[start_t] / Close_raw[start_t - lookback] - 1

    Returns equity curve aligned to [start_t..end_t].
    """
    if not symbols:
        return [start_equity]
    if top_k <= 0:
        return [start_equity]

    start_t = max(0, int(start_t))
    end_t = max(start_t, int(end_t))

    # Need enough history for momentum
    if start_t - int(lookback) < 0:
        # not enough history; fall back to equal-weight across all symbols
        return equal_weight_buy_hold_curve_aligned(
            data_by_symbol, symbols, start_t=start_t, end_t=end_t, start_equity=start_equity
        )

    closes = {s: np.asarray(data_by_symbol[s]["Close_raw"].values, dtype=np.float64) for s in symbols}
    T = min(len(c) for c in closes.values())
    end_t = min(end_t, T - 1)

    # pick Top-K by momentum at start_t
    moms = []
    for s in symbols:
        c = closes[s]
        m = (c[start_t] / max(c[start_t - int(lookback)], 1e-12)) - 1.0
        moms.append((s, float(m)))
    moms.sort(key=lambda x: x[1], reverse=True)
    picked = [s for s, _ in moms[: min(int(top_k), len(moms))]]

    if not picked:
        return [start_equity]

    w = np.ones(len(picked), dtype=np.float64) / float(len(picked))

    equity = float(start_equity)
    curve = [equity]
    for t in range(start_t + 1, end_t + 1):
        rets = np.array([(closes[s][t] / max(closes[s][t - 1], 1e-12)) - 1.0 for s in picked], dtype=np.float64)
        equity *= (1.0 + float(np.dot(w, rets)))
        curve.append(float(equity))

    return curve


# -------------------------
# Env config filtering
# -------------------------
# Keep only config fields supported by current environment dataclass.
def _filter_cfg_dict(cfg_dict: dict) -> dict:
    fields = set(TradingEnvContinuousConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return {k: v for k, v in cfg_dict.items() if k in fields}


# -------------------------
# Deterministic evaluation
# -------------------------
# Run one deterministic environment rollout for a single scenario configuration.
@torch.no_grad()
def run_deterministic_episode(
    *,
    policy: PPOActorCriticContinuous,
    env: TradingEnvContinuous,
    device: str,
    options: dict,
) -> Tuple[List[float], Dict[str, Any]]:
    obs, info0 = env.reset(options=options)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    action_dim = int(env.action_space.shape[0])
    action_mask = torch.ones((1, action_dim), dtype=torch.float32, device=device)

    curve: List[float] = [float(info0.get("equity", 1.0))]
    turnover_last = float(info0.get("turnover", 0.0))
    rebalances_last = int(info0.get("rebalances", 0))

    done = False
    steps = 0

    while not done:
        actions, _, _ = policy.act_batch(obs_t, action_mask, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(actions[0].detach().cpu().numpy())

        curve.append(float(info.get("equity", curve[-1])))
        turnover_last = float(info.get("turnover", turnover_last))
        rebalances_last = int(info.get("rebalances", rebalances_last))

        done = bool(terminated) or bool(truncated)
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

        steps += 1
        if steps > 10_000_000:
            raise RuntimeError("Episode appears to run forever; check env termination logic.")

    ep_info = {
        "symbols": list(info0.get("symbols", [])),
        "start_t": int(info0.get("start_t", info0.get("step", 0))),
        "end_t": int(info0.get("end_t", 0)),
        "episode_length": int(info0.get("episode_length", len(curve))),
        "top_k": int(info.get("top_k", info0.get("top_k", 0))),
        "rebalance_every_n": int(info.get("rebalance_every_n", info0.get("rebalance_every_n", 1))),
        "turnover": float(turnover_last),
        "rebalances": int(rebalances_last),
    }
    return curve, ep_info


@torch.no_grad()
def evaluate_checkpoint_four_curves(
    ckpt_path: str,
    *,
    data_by_symbol: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    device: str,
    base_seed: int,
    encoder_path: str,
    top_k: int,
    weekly_n: int,
    fixed_basket: Optional[List[str]],
) -> Dict[str, Any]:
    # Evaluate one checkpoint across four rebalance/top-k scenarios.
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load env_cfg from checkpoint if present
    if "env_cfg" in ckpt and isinstance(ckpt["env_cfg"], dict):
        env_cfg = TradingEnvContinuousConfig(**_filter_cfg_dict(ckpt["env_cfg"]))
    else:
        env_cfg = TradingEnvContinuousConfig()

    # Create env once
    env = TradingEnvContinuous(
        data_by_symbol=data_by_symbol,
        feature_cols=feature_cols,
        config=env_cfg,
        seed=base_seed,
        encoder_ckpt_path=encoder_path,
        device="cpu",  # env runs encoder on CPU
    )

    obs0, _ = env.reset(options={"mode": "eval"})
    obs_dim = int(np.asarray(obs0).shape[-1])
    action_dim = int(env.action_space.shape[0])

    policy = PPOActorCriticContinuous(obs_dim=obs_dim, action_dim=action_dim).to(device)
    state = ckpt.get("policy_state_dict", ckpt)
    policy.load_state_dict(state)
    policy.eval()

    if fixed_basket is not None and len(fixed_basket) != int(env_cfg.num_assets):
        env.close()
        raise ValueError(
            f"--basket must have exactly num_assets={env_cfg.num_assets} symbols, got {len(fixed_basket)}"
        )

    base_options: Dict[str, Any] = {"mode": "eval"}
    if fixed_basket is not None:
        base_options["symbols"] = fixed_basket

    scenarios = {
        "full_daily": (0, 1),
        "full_weekly": (0, int(weekly_n)),
        "topk_daily": (int(top_k), 1),
        "topk_weekly": (int(top_k), int(weekly_n)),
    }

    results: Dict[str, Any] = {
        "checkpoint": os.path.basename(ckpt_path),
        "checkpoint_path": ckpt_path,
        "evaluated_at_unix": int(time.time()),
        "device": str(device),
        "top_k_param": int(top_k),
        "weekly_n": int(weekly_n),
        "env_cfg": asdict(env_cfg),
        "basket": fixed_basket,
        "scenarios": {},
    }

    for name, (k, n) in scenarios.items():
        # Skip top-k scenarios if disabled
        if int(top_k) <= 0 and name.startswith("topk_"):
            continue

        opts = dict(base_options)
        opts["top_k"] = int(k)
        opts["rebalance_every_n"] = int(max(1, n))

        curve, ep_info = run_deterministic_episode(policy=policy, env=env, device=device, options=opts)
        m = canonicalize_metrics(compute_metrics(curve), curve)

        results["scenarios"][name] = {
            "options": {"top_k": int(k), "rebalance_every_n": int(max(1, n))},
            "episode": ep_info,
            "metrics": m,
            "equity_curve": curve,
        }

    env.close()
    return results


# -------------------------
# FastAPI/Backend helpers (LIVE Kite data)
# -------------------------
# Prepare live market data, align symbols, pad to fixed basket size, and run deterministic inference.

def _to_naive_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(x, format="%Y-%m-%d", errors="raise")
    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            ts = ts.tz_localize(None)
    return ts


def _normalize_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"])
            out = out.set_index("Date")
        else:
            out.index = pd.to_datetime(out.index)

    if getattr(out.index, "tz", None) is not None:
        try:
            out.index = out.index.tz_convert(None)
        except Exception:
            out.index = out.index.tz_localize(None)

    dtype_str = str(out.index.dtype)
    if "tz" in dtype_str and getattr(out.index, "tz", None) is None:
        s = pd.to_datetime(out.index.astype(str))
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)
        out.index = pd.DatetimeIndex(s)

    return out.sort_index()


def _rebalance_every_n(freq: str) -> int:
    f = (freq or "weekly").lower().strip()
    if f == "daily":
        return 1
    if f == "weekly":
        return 5
    if f == "monthly":
        return 21
    raise HTTPException(status_code=400, detail="rebalance must be daily|weekly|monthly")


def _iso_date(d: pd.Timestamp) -> str:
    ts = pd.Timestamp(d)
    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            ts = ts.tz_localize(None)
    return ts.strftime("%Y-%m-%d")


def _make_dummy_df(index: pd.DatetimeIndex, feature_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    for c in feature_cols:
        df[c] = 0.0
    if "Open" in feature_cols:
        df["Open"] = 1.0
    if "Close" in feature_cols:
        df["Close"] = 1.0
    df["Close_raw"] = 1.0
    return df


def _align_on_common_dates(dfs: Dict[str, pd.DataFrame], min_rows: int) -> Dict[str, pd.DataFrame]:
    common: Optional[pd.DatetimeIndex] = None
    for df in dfs.values():
        common = df.index if common is None else common.intersection(df.index)

    if common is None:
        raise HTTPException(status_code=400, detail="No common date index across symbols")
    common = common.sort_values()

    if len(common) < min_rows:
        raise HTTPException(status_code=400, detail="Not enough common dates after alignment")

    out: Dict[str, pd.DataFrame] = {}
    for sym, df in dfs.items():
        d = df.loc[common].dropna(how="any")
        out[sym] = d

    min_len = min(len(df) for df in out.values())
    if min_len < min_rows:
        raise HTTPException(status_code=400, detail="Not enough rows after dropna")
    for sym in list(out.keys()):
        out[sym] = out[sym].iloc[:min_len]

    return out


def _fetch_features_from_kite(*, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Fetch buffered historical data and compute feature matrix for requested dates.
    start_ts = _to_naive_ts(start_date)
    end_ts = _to_naive_ts(end_date)

    # buffer for indicators + GRU warmup
    buffer_start = (start_ts - pd.Timedelta(days=260)).strftime("%Y-%m-%d")

    raw_df = fetch_historical_data(symbol, buffer_start)
    if raw_df is None or raw_df.empty:
        fetch_err = ""
        if isinstance(raw_df, pd.DataFrame):
            fetch_err = str(raw_df.attrs.get("fetch_error", "")).strip()
        if fetch_err:
            # Upstream data-source failure should not be presented as 404 route/data-missing.
            raise HTTPException(status_code=502, detail=f"Kite data fetch failed for {symbol}: {fetch_err}")
        raise HTTPException(status_code=404, detail=f"No Kite data for {symbol}")

    df = _normalize_dt_index(raw_df)

    feat = calculate_technical_features(df)
    if feat is None or feat.empty:
        raise HTTPException(status_code=400, detail=f"Feature calculation failed for {symbol}")

    feat = _normalize_dt_index(feat)

    feat = feat.loc[(feat.index >= start_ts) & (feat.index <= end_ts)]
    if feat.empty:
        raise HTTPException(status_code=400, detail=f"Not enough feature rows for {symbol} in selected range")

    return feat


def load_kite_scaled_data(
    *,
    symbols: List[str],
    start_date: str,
    end_date: str,
    feature_cols: List[str],
    scaler_path: str,
    min_rows: int,
) -> Dict[str, pd.DataFrame]:
    # Load and scale live feature frames, then align all symbols to common dates.
    if not Path(scaler_path).exists():
        raise HTTPException(status_code=500, detail=f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        feat = _fetch_features_from_kite(symbol=sym, start_date=start_date, end_date=end_date)

        if "Close" not in feat.columns:
            raise HTTPException(status_code=400, detail=f"Missing Close for {sym} after feature calc")
        feat = feat.copy()
        feat["Close_raw"] = feat["Close"].astype(float)

        missing = [c for c in feature_cols if c not in feat.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"{sym} missing required features: {missing}")

        feat.loc[:, feature_cols] = scaler.transform(feat.loc[:, feature_cols].to_numpy())
        feat = _normalize_dt_index(feat)
        feat = feat.dropna(how="any")

        out[sym] = feat

    out = _align_on_common_dates(out, min_rows=min_rows)
    return out


def buy_and_hold_equal_weight(
    close_raw: pd.DataFrame,
    initial_cash: float,
    max_positions: Optional[int] = None,
    symbol_order: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    # Compute equal-weight baseline curve over selected real symbols.
    if close_raw.empty:
        return []

    cols = list(close_raw.columns)
    if symbol_order:
        cols = [s for s in symbol_order if s in close_raw.columns]

    if max_positions is not None:
        k = int(max_positions)
        if k <= 0:
            k = len(cols)
        cols = cols[: min(k, len(cols))]

    if not cols:
        return []

    cr = close_raw.loc[:, cols]
    base = cr.iloc[0]
    rel = cr.divide(base)
    eq = rel.mean(axis=1) * float(initial_cash)

    return [{"date": _iso_date(d), "value": float(v)} for d, v in eq.items()]


@torch.no_grad()
def run_agent_from_kite_pad_mask(
    *,
    user_symbols: List[str],
    start_date: str,
    end_date: str,
    rebalance: str,
    max_positions: int,
    rotation_factor: float,
    initial_cash: float,
    scaler_path: str,
    encoder_path: str,
    ckpt_path: str,
    feature_cols: List[str],
    num_assets_fixed: int = 14,
    include_cash: bool = True,
    gru_window: int = 30,
) -> Dict[str, Any]:
    """LIVE Kite inference with padding + hard-mask.

    - User may supply 1..14 symbols; missing slots are padded with dummy assets.
    - Dummy assets are hard-masked so they receive ~0 weight.
    - `max_positions` enforces Top-K when 1..14; 0 means full.
    - Buy&Hold baseline uses the SAME real symbols, and if Top-K is enforced it uses only K symbols
      (deterministic: first K in the user's order).
    """

    syms = [s.strip() for s in user_symbols if s and s.strip()]
    syms = list(dict.fromkeys(syms))
    if not (1 <= len(syms) <= int(num_assets_fixed)):
        raise HTTPException(status_code=400, detail=f"symbols must be 1..{int(num_assets_fixed)}")

    # Enforce minimum row count for warmup and stable rollout.
    min_rows = max(int(gru_window) + 5, 32)

    # Stage 1: fetch live data, compute features, scale, and align.
    data_by_symbol = load_kite_scaled_data(
        symbols=syms,
        start_date=start_date,
        end_date=end_date,
        feature_cols=feature_cols,
        scaler_path=scaler_path,
        min_rows=min_rows,
    )

    common_idx = next(iter(data_by_symbol.values())).index

    # Stage 2: pad to fixed asset count with deterministic dummy symbols.
    padded = list(syms)
    dummy_asset_indices: List[int] = []
    while len(padded) < int(num_assets_fixed):
        dname = f"DUMMY_{len(padded)+1}"
        data_by_symbol[dname] = _make_dummy_df(common_idx, feature_cols)
        dummy_asset_indices.append(len(padded))
        padded.append(dname)

    # Stage 3: load policy checkpoint and environment metadata overrides.
    ckpt_p = Path(ckpt_path)
    if not ckpt_p.exists():
        raise HTTPException(status_code=500, detail=f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_p), map_location="cpu")

    # env config from checkpoint if present
    if isinstance(ckpt, dict) and isinstance(ckpt.get("env_cfg"), dict):
        env_cfg = TradingEnvContinuousConfig(**_filter_cfg_dict(ckpt["env_cfg"]))
    else:
        env_cfg = TradingEnvContinuousConfig()

    # enforce runtime shape
    env_cfg.num_assets = int(num_assets_fixed)
    if hasattr(env_cfg, "include_cash"):
        env_cfg.include_cash = bool(include_cash)
    if hasattr(env_cfg, "gru_window"):
        env_cfg.gru_window = int(gru_window)

    env = TradingEnvContinuous(
        data_by_symbol=data_by_symbol,
        feature_cols=feature_cols,
        config=env_cfg,
        seed=0,
        encoder_ckpt_path=str(encoder_path),
        device="cpu",
    )

    # Map requested max_positions to runtime top-k constraint.
    mp = int(max_positions)
    if mp <= 0 or mp >= int(num_assets_fixed):
        top_k = 0  # 0 means "no top-k constraint" in this evaluator
    else:
        top_k = mp

    # Map requested rebalance frequency to step interval.
    reb_n = int(_rebalance_every_n(rebalance))

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])  # assets (+ cash if enabled)

    policy = PPOActorCriticContinuous(obs_dim=obs_dim, action_dim=action_dim)
    state = ckpt.get("policy_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    policy.load_state_dict(state)
    policy.eval()

    action_mask = torch.ones((1, action_dim), dtype=torch.float32)

    # Use aligned index as canonical date axis for output curves.
    df0 = data_by_symbol[padded[0]]
    idx0 = df0.index

    # Build momentum overlay signal used for optional rotation bias.
    rotation_value = float(max(0.0, rotation_factor))
    mom_alpha = rotation_value
    mom_lookback = 20
    close_by_real_sym: Dict[str, np.ndarray] = {
        s: np.asarray(data_by_symbol[s]["Close_raw"].astype(float).values, dtype=np.float64) for s in syms
    }

    def _momentum_zscores(step_idx: int) -> np.ndarray:
        vals = np.zeros((len(syms),), dtype=np.float64)
        for i, s in enumerate(syms):
            arr = close_by_real_sym[s]
            t = int(max(0, min(step_idx, len(arr) - 1)))
            t0 = max(0, t - mom_lookback)
            base = float(arr[t0]) if len(arr) else 0.0
            curr = float(arr[t]) if len(arr) else 0.0
            if base > 0.0:
                vals[i] = (curr / base) - 1.0
            else:
                vals[i] = 0.0
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd <= 1e-12:
            return np.zeros_like(vals, dtype=np.float64)
        return (vals - mu) / sd

    def _rollout_for_n(n: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], pd.Timestamp]:
        # Run deterministic rollout for one rebalance interval and collect diagnostics.
        opts = {
            "mode": "eval",
            "symbols": padded,
            "top_k": int(top_k),
            "rebalance_every_n": int(max(1, n)),
        }
        obs_local, info0_local = env.reset(options=opts)
        obs_t_local = torch.tensor(obs_local, dtype=torch.float32).unsqueeze(0)

        curve_local: List[Dict[str, Any]] = []
        step0 = int(info0_local.get("step", 0))
        curve_local.append({"date": _iso_date(idx0[step0]), "value": float(initial_cash)})
        start_dt_local = pd.Timestamp(idx0[step0])

        last_info_local: Dict[str, Any] = info0_local if isinstance(info0_local, dict) else {}
        rebalance_events: List[Dict[str, Any]] = []
        prev_topk_set: Optional[set[str]] = None
        set_change_count = 0
        set_compares = 0
        prev_real_w: Optional[np.ndarray] = None
        weight_shifts: List[float] = []
        unique_selected_symbols: set[str] = set()
        done_local = False
        while not done_local:
            actions, _, _ = policy.act_batch(obs_t_local, action_mask, deterministic=True)
            logits = actions[0].detach().cpu().numpy()

            if mom_alpha > 0.0 and len(syms) > 0:
                step_sig = int(last_info_local.get("step", step0))
                mz = _momentum_zscores(step_sig)
                logits[: len(syms)] = logits[: len(syms)] + (mom_alpha * mz.astype(np.float32))

            if dummy_asset_indices:
                logits[np.array(dummy_asset_indices, dtype=int)] = -1e9

            next_obs, _, terminated, truncated, info = env.step(logits)
            last_info_local = info if isinstance(info, dict) else {}
            done_local = bool(terminated) or bool(truncated)

            step_i = int(last_info_local.get("step", len(idx0) - 1))
            step_i = max(0, min(step_i, len(idx0) - 1))
            step_date = _iso_date(idx0[step_i])
            curve_local.append(
                {"date": step_date, "value": float(last_info_local.get("equity", 1.0)) * float(initial_cash)}
            )

            if bool(last_info_local.get("did_rebalance", False)):
                w_all = np.asarray(last_info_local.get("weights", []), dtype=np.float64).reshape(-1)
                asset_w = w_all[: len(padded)] if w_all.size >= len(padded) else np.zeros((len(padded),), dtype=np.float64)
                cash_w = float(w_all[len(padded)]) if (w_all.size > len(padded)) else 0.0

                ranked = sorted(
                    [(sym, float(asset_w[i])) for i, sym in enumerate(padded)],
                    key=lambda x: x[1],
                    reverse=True,
                )
                selected_real = [
                    {"symbol": sym, "weight": wt}
                    for sym, wt in ranked
                    if (sym in syms) and (wt > 1e-8)
                ]
                if top_k > 0:
                    selected_real = selected_real[: int(top_k)]
                current_topk_set = set(x["symbol"] for x in selected_real)
                unique_selected_symbols.update(current_topk_set)
                if prev_topk_set is not None:
                    set_compares += 1
                    if current_topk_set != prev_topk_set:
                        set_change_count += 1
                prev_topk_set = current_topk_set

                real_w = asset_w[: len(syms)].astype(np.float64, copy=False)
                if prev_real_w is not None:
                    weight_shifts.append(float(np.sum(np.abs(real_w - prev_real_w))))
                prev_real_w = real_w.copy()

                rebalance_events.append(
                    {
                        "date": step_date,
                        "step": int(step_i),
                        "cash_weight": cash_w,
                        "selected_symbols": selected_real,
                    }
                )

            obs_t_local = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        ep_local = {
            "rebalances": int(last_info_local.get("rebalances", 0)),
            "turnover": float(last_info_local.get("turnover", 0.0)),
            "steps": int(last_info_local.get("step", len(curve_local) - 1)),
            "rebalance_every_n": int(last_info_local.get("rebalance_every_n", int(max(1, n)))),
            "final_equity": float(curve_local[-1]["value"]) if curve_local else float(initial_cash),
            "topk_set_change_rate": (float(set_change_count) / float(set_compares)) if set_compares > 0 else 0.0,
            "avg_weight_shift_per_rebalance": (float(np.mean(weight_shifts)) if weight_shifts else 0.0),
            "unique_selected_symbols_count": int(len(unique_selected_symbols)),
            "unique_selected_symbols": sorted(list(unique_selected_symbols)),
            "rotation_factor": float(rotation_value),
            "rebalance_events": rebalance_events,
        }
        return curve_local, ep_local, start_dt_local

    rebalance_map = {"daily": 1, "weekly": 5, "monthly": 21}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    curves_by_freq: Dict[str, List[Dict[str, Any]]] = {}
    starts_by_freq: Dict[str, pd.Timestamp] = {}

    for freq_name, n in rebalance_map.items():
        c, ep, st = _rollout_for_n(int(n))
        curves_by_freq[freq_name] = c
        diagnostics[freq_name] = ep
        starts_by_freq[freq_name] = st

    freq_key = (rebalance or "weekly").lower().strip()
    if freq_key not in curves_by_freq:
        raise HTTPException(status_code=400, detail="rebalance must be daily|weekly|monthly")
    agent_curve = curves_by_freq[freq_key]
    last_info = diagnostics[freq_key]
    agent_start_dt = starts_by_freq[freq_key]

    # Stage 4: compute buy-and-hold baseline on real symbols only.
    close_raw = pd.DataFrame(
        {s: data_by_symbol[s]["Close_raw"].astype(float).values for s in syms},
        index=common_idx,
    )

    # Start baseline on agent start date for direct curve comparability.
    close_raw_bh = close_raw.loc[close_raw.index >= agent_start_dt]

    bh_curve = buy_and_hold_equal_weight(
        close_raw_bh,
        initial_cash,
        max_positions=(top_k if top_k > 0 else None),
        symbol_order=syms,
    )

    # Align agent and baseline outputs to common dates before response serialization.
    agent_map = {p["date"]: p["value"] for p in agent_curve}
    bh_map = {p["date"]: p["value"] for p in bh_curve}
    common_dates = sorted(set(agent_map.keys()).intersection(bh_map.keys()))

    agent_out = [{"date": d, "value": float(agent_map[d])} for d in common_dates]
    bh_out = [{"date": d, "value": float(bh_map[d])} for d in common_dates]

    env.close()

    return {
        "checkpoint": str(ckpt_p),
        "basket": padded,
        "real_symbols": syms,
        "top_k": int(top_k),
        "rebalance": str(rebalance),
        "rebalance_every_n": int(reb_n),
        "rotation_factor": float(rotation_value),
        "rebalance_diagnostics": diagnostics,
        "episode": {
            "rebalances": int(last_info.get("rebalances", 0)),
            "turnover": float(last_info.get("turnover", 0.0)),
            "avg_turnover_per_rebalance": (
                float(last_info.get("turnover", 0.0)) / float(max(1, int(last_info.get("rebalances", 0))))
            ),
            "steps": int(last_info.get("steps", len(agent_out) - 1)),
            "final_equity": float(last_info.get("final_equity", agent_out[-1]["value"] if agent_out else initial_cash)),
        },
        "agent": agent_out,
        "buy_and_hold": bh_out,
    }


# -------------------------
# Plotting
# -------------------------
# Render comparable curve plots with shared horizon length.
def plot_compare_curves(
    *,
    out_path: str,
    title: str,
    curves: List[Tuple[str, List[float]]],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not curves:
        return
    L = min(len(c) for _, c in curves if c)
    if L <= 1:
        return

    plt.figure(figsize=(10, 4))
    for label, curve in curves:
        if curve:
            plt.plot(curve[:L], label=label)

    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------
# Main
# -------------------------
# Parse CLI inputs, evaluate checkpoint scenarios, and write JSON/plot artifacts.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")

    # dates from frontend
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")

    parser.add_argument(
        "--out_dir",
        default=os.path.join("backend", "artifacts", "phase_d_continuous", "eval_single"),
        help="Directory to write outputs",
    )

    parser.add_argument("--data_dir", default=os.path.join("backend", "data", "processed"))
    parser.add_argument("--scaler", default=os.path.join("backend", "models", "checkpoints", "scaler.pkl"))
    parser.add_argument(
        "--feature_cols",
        default="Open,Close,RSI,MACD,MACD_Signal,ATR,SMA_50,SMA_Ratio,OBV,ROC_10,RealizedVol_20",
    )
    parser.add_argument("--encoder", default=os.path.join("backend", "models", "checkpoints", "gru_encoder.pt"))

    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--weekly_n", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--basket",
        default="",
        help="Optional fixed basket of exactly num_assets symbols (comma-separated). If omitted, env selects deterministically from seed.",
    )

    args = parser.parse_args()

    fixed_basket = [s.strip() for s in args.basket.split(",") if s.strip()] or None
    feature_cols_default = [c.strip() for c in str(args.feature_cols).split(",") if c.strip()]

    # Read checkpoint metadata overrides if present
    ckpt_cpu = torch.load(args.ckpt, map_location="cpu")

    ckpt_feature_cols = ckpt_cpu.get("feature_cols")
    feature_cols = (
        [c.strip() for c in ckpt_feature_cols]
        if isinstance(ckpt_feature_cols, (list, tuple))
        else [c.strip() for c in str(ckpt_feature_cols).split(",") if c.strip()]
        if isinstance(ckpt_feature_cols, str)
        else feature_cols_default
    )
    scaler_path = ckpt_cpu.get("scaler_path") if isinstance(ckpt_cpu.get("scaler_path"), str) else args.scaler
    encoder_path = ckpt_cpu.get("gru_encoder_path") if isinstance(ckpt_cpu.get("gru_encoder_path"), str) else args.encoder

    # Load data for the requested window
    data = load_and_prepare_data(
        data_dir=args.data_dir,
        scaler_path=scaler_path,
        feature_cols=feature_cols,
        start_date=args.start,
        end_date=args.end,
    )

    # Evaluate
    result = evaluate_checkpoint_four_curves(
        args.ckpt,
        data_by_symbol=data,
        feature_cols=feature_cols,
        device=args.device,
        base_seed=int(args.seed),
        encoder_path=encoder_path,
        top_k=int(args.top_k),
        weekly_n=int(args.weekly_n),
        fixed_basket=fixed_basket,
    )

    ckpt_base = os.path.basename(args.ckpt).replace(".pt", "")
    tag = f"{ckpt_base}__{args.start}__{args.end}"
    out_json = os.path.join(args.out_dir, f"{tag}.json")
    _atomic_write_json(out_json, result)
    print(f"[OK] wrote JSON: {out_json}")

    # Make plots (BH vs PPO), aligned to the episode range of each scenario
    out_plots = os.path.join(args.out_dir, "plots")
    os.makedirs(out_plots, exist_ok=True)

    for scen_name, scen in result.get("scenarios", {}).items():
        if not isinstance(scen, dict):
            continue
        ppo_curve = scen.get("equity_curve", [])
        ep = scen.get("episode", {})
        if not ppo_curve or not isinstance(ep, dict):
            continue

        symbols = list(ep.get("symbols", []))
        if not symbols and fixed_basket is not None:
            symbols = fixed_basket
        if not symbols:
            print(f"[WARN] no symbols for {scen_name}, skipping plot")
            continue

        start_t = int(ep.get("start_t", 0))
        end_t = int(ep.get("end_t", 0))
        # If env doesn't provide end_t, align to PPO curve length
        if end_t <= start_t:
            end_t = start_t + (len(ppo_curve) - 1)

        bh_curve = equal_weight_buy_hold_curve_aligned(
            data,
            symbols,
            start_t=start_t,
            end_t=end_t,
            start_equity=1.0,
        )

        plot_path = os.path.join(out_plots, f"{tag}__{scen_name}.png")
        plot_compare_curves(
            out_path=plot_path,
            title=f"BH vs PPO ({scen_name}) | {args.start}..{args.end}",
            curves=[
                ("Buy & Hold (equal-weight)", bh_curve),
                (f"PPO ({scen_name})", ppo_curve),
            ],
        )
        print(f"[OK] wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
