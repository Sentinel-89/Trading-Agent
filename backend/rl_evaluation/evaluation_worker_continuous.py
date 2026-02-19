
"""backend/rl_evaluation/evaluation_runner_continuous.py

Deterministic evaluation for CONTINUOUS portfolio PPO checkpoints.

What this script does (updated)
------------------------------
1) VALIDATION pass (default: 2024-06-01 .. 2024-12-30)
   - Evaluate every checkpoint deterministically and save ONE JSON per checkpoint.
   - Rank checkpoints by a single scalar score (documented below).
   - Print and save the Top-5 checkpoints.

2) TEST pass (default: 2025-01-01 .. 2025-12-30)
   - Take the best 2 checkpoints from validation.
   - Evaluate again deterministically and save ONE JSON per checkpoint.
   - For each of the best checkpoints, also plot 4 comparisons vs a BUY-&-HOLD baseline:
       (a) BH vs full_daily   (no top-k, rebalance daily)
       (b) BH vs full_weekly  (no top-k, rebalance every N days)
       (c) BH vs topk_daily   (enforce top-k, rebalance daily)
       (d) BH vs topk_weekly  (enforce top-k, rebalance every N days)

Buy-&-Hold baseline here means: equal-weight portfolio across the basket (no cash),
held constant for the whole window. PnL is computed from RAW close prices.

Checkpoint ranking score (validation)
------------------------------------
Checkpoint ranking uses the validation FULL_DAILY scenario (closest to training settings).
Score is a weighted combination:

    score = (1.00 * sharpe)
          + (0.50 * total_return)
          - (1.50 * abs(max_drawdown))
          - (0.05 * turnover)

- Sharpe is the primary objective (risk-adjusted return).
- Total return helps separate similar Sharpe models.
- Drawdown penalty discourages very risky policies.
- Turnover penalty discourages excessive rebalancing.

Weights are configurable as needed.

Correctness rule
----------------
PnL is computed from RAW prices. Close_raw/Open_raw are preserved from CSVs.
Feature columns are scaled for the GRU encoder, but returns use Close_raw.

Outputs
-------
- Validation JSONs:   out_dir/val/<ckpt>.json
- Validation ranking: out_dir/val/top5.json
- Test JSONs:         out_dir/test/<ckpt>.json
- Test plots:         out_dir/test/plots/<ckpt>__<scenario>.png

Notes
-----
- This evaluator is deterministic given (seed, date range, basket selection).
- By default it runs a single deterministic episode per checkpoint (one basket).
  For multiple baskets/episodes per checkpoint, extend `run_episode_specs`.
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

# plotting (safe in CLI scripts)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.rl_evaluation.evaluate_utils import compute_metrics, canonicalize_metrics
from backend.rl.multiasset.ppo_actor_critic_continuous import PPOActorCriticContinuous
from backend.rl.multiasset.trading_env_continuous import TradingEnvContinuous, TradingEnvContinuousConfig


# ============================================================
# Defaults (override via CLI)
# ============================================================

DEFAULT_VAL_START = "2024-06-01"
DEFAULT_VAL_END = "2024-12-30"
DEFAULT_TEST_START = "2025-01-01"
DEFAULT_TEST_END = "2025-12-30"

DEFAULT_DATA_DIR = os.path.join("backend", "data", "processed")

# Keep small; actual requirements depend on checkpoint env_cfg (warmup/episode).
MIN_COMMON_ROWS_DEFAULT = 32


# ============================================================
# JSON helpers
# ============================================================

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


# ============================================================
# Data loading
# ============================================================

def load_and_prepare_data(
    *,
    data_dir: str,
    scaler_path: str,
    feature_cols: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """Load *_labeled.csv files, slice date range, preserve raw prices, scale features.

    Returns dict(symbol -> df) where every df has the SAME number of rows and can be indexed
    safely with iloc.
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

        # Preserve raw prices for returns
        if "Close" not in df.columns:
            raise KeyError(f"Symbol {sym}: missing Close column")
        df["Close_raw"] = df["Close"].astype(float)
        if "Open" in df.columns:
            df["Open_raw"] = df["Open"].astype(float)

        # Check features exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Symbol {sym}: missing feature columns: {missing}")

        # Scale ONLY feature columns (GRU inputs). Prices used for PnL come from *_raw.
        df.loc[:, feature_cols] = scaler.transform(df.loc[:, feature_cols].to_numpy())

        data[sym] = df

    if not data:
        raise ValueError(f"No *_labeled.csv files found in {data_dir}")

    # Align to common dates (intersection)
    common_idx = None
    for df in data.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) < MIN_COMMON_ROWS_DEFAULT:
        raise ValueError(
            f"Not enough common dates after alignment: n={0 if common_idx is None else len(common_idx)} "
            f"(min required {MIN_COMMON_ROWS_DEFAULT})."
        )

    common_idx = common_idx.sort_values()

    # Reindex to intersection; drop NAs; reset to 0..N-1.
    for sym in list(data.keys()):
        data[sym] = data[sym].reindex(common_idx).dropna().reset_index(drop=True)

    # Enforce equal-length across symbols (safety for iloc)
    lengths = {sym: len(df) for sym, df in data.items()}
    min_len = min(lengths.values()) if lengths else 0

    if min_len < MIN_COMMON_ROWS_DEFAULT:
        raise ValueError(
            f"Not enough common rows after alignment: min_len={min_len} (min required {MIN_COMMON_ROWS_DEFAULT})."
        )

    if len(set(lengths.values())) != 1:
        print(
            f"[DataAlign] Unequal lengths after dropna; truncating all to min_len={min_len}. "
            f"Example shortest: {sorted(lengths.items(), key=lambda x: x[1])[:5]}"
        )
        for sym in list(data.keys()):
            data[sym] = data[sym].iloc[:min_len].reset_index(drop=True)

    return data
def compute_validation_score(metrics: Dict[str, float], turnover: float) -> float:
    """Single scalar score for ranking checkpoints on validation.

    score = 1.00*sharpe + 0.50*total_return - 1.50*abs(max_drawdown) - 0.05*turnover
    """
    sr = float(metrics.get("sharpe", 0.0))
    tr = float(metrics.get("total_return", 0.0))
    mdd = float(metrics.get("max_drawdown", 0.0))
    return (1.00 * sr) + (0.50 * tr) - (1.50 * abs(mdd)) - (0.05 * float(turnover))


# ============================================================
# Baseline: equal-weight buy & hold curve (portfolio)
# ============================================================


def equal_weight_buy_hold_curve(
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    *,
    start_equity: float = 1.0,
) -> List[float]:
    """Equal-weight buy&hold across `symbols`, using Close_raw.

    - weights are fixed at 1/N for all assets, no cash.
    - equity evolves by portfolio daily return.
    """
    if not symbols:
        return [start_equity]

    # Align by row index (data_by_symbol already aligned and equal-length)
    n = len(symbols)
    w = np.ones(n, dtype=np.float64) / float(n)

    closes = [np.asarray(data_by_symbol[s]["Close_raw"].values, dtype=np.float64) for s in symbols]
    T = min(len(c) for c in closes)
    if T < 2:
        return [start_equity]

    equity = float(start_equity)
    curve = [equity]

    # daily returns per asset: (P_t/P_{t-1}-1)
    for t in range(1, T):
        rets = np.array([(c[t] / max(c[t - 1], 1e-12)) - 1.0 for c in closes], dtype=np.float64)
        port_ret = float(np.dot(w, rets))
        equity *= (1.0 + port_ret)
        curve.append(float(equity))

    return curve

# ============================================================
# Baselines: momentum Top-K (fixed and dynamic)
# ============================================================

def momentum_topk_fixed_buy_hold_curve(
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    *,
    start_t: int,
    end_t: int,
    lookback: int,
    top_k: int,
    start_equity: float = 1.0,
) -> List[float]:
    """Select Top-K by lookback momentum once at start_t, then equal-weight hold to end_t (inclusive).

    momentum(s) = Close_raw[start_t] / Close_raw[start_t-lookback] - 1

    Returns equity curve aligned to episode steps: length = (end_t-start_t+1).
    """
    if not symbols:
        return [start_equity]
    if top_k <= 0:
        raise ValueError("top_k must be > 0 for momentum baselines")
    if start_t - lookback < 0:
        raise ValueError(f"start_t={start_t} must be >= lookback={lookback}")

    # close arrays (already aligned by row)
    closes = {s: np.asarray(data_by_symbol[s]["Close_raw"].values, dtype=np.float64) for s in symbols}
    T = min(len(c) for c in closes.values())
    end_t = min(int(end_t), T - 1)

    moms = []
    for s in symbols:
        c = closes[s]
        m = (c[start_t] / max(c[start_t - lookback], 1e-12)) - 1.0
        moms.append((s, float(m)))
    moms.sort(key=lambda x: x[1], reverse=True)
    picked = [s for s, _ in moms[: min(top_k, len(moms))]]

    w = np.ones(len(picked), dtype=np.float64) / float(len(picked))

    equity = float(start_equity)
    curve = [equity]

    for t in range(start_t + 1, end_t + 1):
        rets = np.array([(closes[s][t] / max(closes[s][t - 1], 1e-12)) - 1.0 for s in picked], dtype=np.float64)
        equity *= (1.0 + float(np.dot(w, rets)))
        curve.append(float(equity))

    return curve


def momentum_topk_dynamic_curve(
    data_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    *,
    start_t: int,
    end_t: int,
    lookback: int,
    top_k: int,
    rebalance_every_n: int,
    start_equity: float = 1.0,
) -> List[float]:
    """Dynamic Top-K momentum: reselect Top-K every `rebalance_every_n` steps, equal-weight.

    No transaction costs (baseline).
    Returns equity curve aligned to episode steps: length = (end_t-start_t+1).
    """
    if not symbols:
        return [start_equity]
    if top_k <= 0:
        raise ValueError("top_k must be > 0 for momentum baselines")
    if rebalance_every_n <= 0:
        rebalance_every_n = 1
    if start_t - lookback < 0:
        raise ValueError(f"start_t={start_t} must be >= lookback={lookback}")

    closes = {s: np.asarray(data_by_symbol[s]["Close_raw"].values, dtype=np.float64) for s in symbols}
    T = min(len(c) for c in closes.values())
    end_t = min(int(end_t), T - 1)

    equity = float(start_equity)
    curve = [equity]

    picked: List[str] = []

    for t in range(start_t + 1, end_t + 1):
        # rebalance at the beginning of each block: use momentum computed at (t-1)
        if ((t - 1 - start_t) % rebalance_every_n) == 0 or not picked:
            mom_list = []
            for s in symbols:
                c = closes[s]
                m = (c[t - 1] / max(c[t - 1 - lookback], 1e-12)) - 1.0
                mom_list.append((s, float(m)))
            mom_list.sort(key=lambda x: x[1], reverse=True)
            picked = [s for s, _ in mom_list[: min(top_k, len(mom_list))]]

        w = np.ones(len(picked), dtype=np.float64) / float(len(picked))
        rets = np.array([(closes[s][t] / max(closes[s][t - 1], 1e-12)) - 1.0 for s in picked], dtype=np.float64)
        equity *= (1.0 + float(np.dot(w, rets)))
        curve.append(float(equity))

    return curve


# ============================================================
# Checkpoint helpers
# ============================================================

def list_checkpoints(ckpt_dir: str, include_latest: bool = True) -> List[str]:
    """Return sorted checkpoint paths in a directory."""
    paths: List[str] = []
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith(".pt"):
            continue
        if not include_latest and fn == "latest.pt":
            continue
        paths.append(os.path.join(ckpt_dir, fn))
    return sorted(paths)


def default_out_path(out_dir: str, ckpt_path: str) -> str:
    base = os.path.basename(ckpt_path)
    if base.endswith(".pt"):
        base = base[:-3]
    return os.path.join(out_dir, f"{base}.json")


def _filter_cfg_dict(cfg_dict: dict) -> dict:
    """Filter loaded config dict to dataclass fields (forward-compatible)."""
    fields = set(TradingEnvContinuousConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return {k: v for k, v in cfg_dict.items() if k in fields}


# ============================================================
# Deterministic evaluation
# ============================================================

@torch.no_grad()
def run_deterministic_episode(
    *,
    policy: PPOActorCriticContinuous,
    env: TradingEnvContinuous,
    device: str,
    options: dict,
) -> Tuple[List[float], Dict[str, Any]]:
    """Run one deterministic episode and return the equity curve + episode info."""

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
        # Deterministic action = mean (no sampling)
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
    start_t: Optional[int],
    end_t: Optional[int],
) -> Dict[str, Any]:
    """Load one checkpoint and produce 4 deterministic equity curves."""

    ckpt = torch.load(ckpt_path, map_location=device)

    # Load env_cfg from checkpoint if present
    if "env_cfg" in ckpt and isinstance(ckpt["env_cfg"], dict):
        env_cfg = TradingEnvContinuousConfig(**_filter_cfg_dict(ckpt["env_cfg"]))
    else:
        env_cfg = TradingEnvContinuousConfig()

    # Create env once (single-env evaluation) to keep results simple and deterministic.
    env = TradingEnvContinuous(
        data_by_symbol=data_by_symbol,
        feature_cols=feature_cols,
        config=env_cfg,
        seed=base_seed,
        encoder_ckpt_path=encoder_path,
        device="cpu",  # env runs encoder on CPU
    )

    # Infer dims
    obs0, info0 = env.reset(options={"mode": "eval"})
    obs_dim = int(np.asarray(obs0).shape[-1])
    action_dim = int(env.action_space.shape[0])

    # Build policy and load weights
    policy = PPOActorCriticContinuous(obs_dim=obs_dim, action_dim=action_dim).to(device)
    state = ckpt.get("policy_state_dict", ckpt)
    policy.load_state_dict(state)
    policy.eval()

    # Basket selection:
    # - Use fixed_basket when provided (must match num_assets).
    # - Otherwise use env internal selection with deterministic seeding.
    if fixed_basket is not None and len(fixed_basket) != int(env_cfg.num_assets):
        env.close()
        raise ValueError(
            f"--basket must have exactly num_assets={env_cfg.num_assets} symbols, got {len(fixed_basket)}"
        )

    # Common options
    base_options: Dict[str, Any] = {"mode": "eval"}
    if fixed_basket is not None:
        base_options["symbols"] = fixed_basket
    if start_t is not None:
        base_options["start_t"] = int(start_t)
    if end_t is not None:
        base_options["end_t"] = int(end_t)  # inclusive

    # Scenarios: (top_k, rebalance_every_n)
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
        "start_t": start_t,
        "end_t": end_t,
        "scenarios": {},
    }

    # Run each scenario
    for name, (k, n) in scenarios.items():
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


# ============================================================
# Plotting helpers
# ============================================================


def plot_compare_curves(
    *,
    out_path: str,
    title: str,
    curves: List[Tuple[str, List[float]]],
) -> None:
    """Plot multiple named curves on the same axes.

    `curves` is a list of (label, curve).
    Curves are aligned by min length.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not curves:
        return

    L = min(len(c) for _, c in curves if c)
    if L <= 1:
        return

    plt.figure(figsize=(10, 4))
    for label, curve in curves:
        if not curve:
            continue
        plt.plot(curve[:L], label=label)

    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", default="", help="Path to a single checkpoint .pt")
    parser.add_argument("--ckpt_dir", default="", help="Directory of checkpoints to evaluate")

    parser.add_argument(
        "--out_dir",
        default=os.path.join("backend", "artifacts", "phase_d_continuous", "eval"),
        help="Directory to write validation + test outputs",
    )

    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--scaler", default=os.path.join("backend", "models", "checkpoints", "scaler.pkl"))
    parser.add_argument(
        "--feature_cols",
        default="Open,Close,RSI,MACD,MACD_Signal,ATR,SMA_50,SMA_Ratio,OBV,ROC_10,RealizedVol_20",
        help="Comma-separated feature columns to scale",
    )
    parser.add_argument("--encoder", default=os.path.join("backend", "models", "checkpoints", "gru_encoder.pt"))

    # Validation window (requested)
    parser.add_argument("--val_start", default=DEFAULT_VAL_START)
    parser.add_argument("--val_end", default=DEFAULT_VAL_END)

    # Test window (requested)
    parser.add_argument("--test_start", default=DEFAULT_TEST_START)
    parser.add_argument("--test_end", default=DEFAULT_TEST_END)

    # Evaluation knobs
    parser.add_argument("--top_k", type=int, default=0, help="k for top-k scenarios (0 disables top-k)")
    parser.add_argument("--weekly_n", type=int, default=5, help="Rebalance interval for weekly scenarios")
    parser.add_argument("--mom_lookback", type=int, default=20, help="Lookback (bars) for momentum baselines")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--basket",
        default="",
        help=(
            "Optional fixed basket of exactly num_assets symbols (comma-separated). "
            "If omitted, env selects a basket deterministically from seed."
        ),
    )

    args = parser.parse_args()

    if not args.ckpt and not args.ckpt_dir:
        raise ValueError("Provide either --ckpt or --ckpt_dir")

    fixed_basket = [s.strip() for s in args.basket.split(",") if s.strip()] or None
    feature_cols = [c.strip() for c in str(args.feature_cols).split(",") if c.strip()]

    # Helpers
    def load_data_for_window(start: str, end: str, scaler_path: str, feature_cols_local: List[str]):
        return load_and_prepare_data(
            data_dir=args.data_dir,
            scaler_path=scaler_path,
            feature_cols=feature_cols_local,
            start_date=start,
            end_date=end,
        )

    def eval_with_ckpt_metadata(
        ckpt_path: str,
        *,
        data_window: Dict[str, pd.DataFrame],
        feature_cols_default: List[str],
        start: str,
        end: str,
    ) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], List[str], str, str]:
        """Evaluate ckpt using checkpoint-provided scaler/encoder/feature_cols when present.

        Returns: (result_json, data_used, feature_cols_used, scaler_path_used, encoder_path_used)
        """
        ckpt_cpu = torch.load(ckpt_path, map_location="cpu")

        ckpt_feature_cols = ckpt_cpu.get("feature_cols")
        feature_cols_local = (
            [c.strip() for c in ckpt_feature_cols]
            if isinstance(ckpt_feature_cols, (list, tuple))
            else [c.strip() for c in str(ckpt_feature_cols).split(",") if c.strip()]
            if isinstance(ckpt_feature_cols, str)
            else feature_cols_default
        )

        scaler_path = ckpt_cpu.get("scaler_path") if isinstance(ckpt_cpu.get("scaler_path"), str) else args.scaler
        encoder_path = (
            ckpt_cpu.get("gru_encoder_path")
            if isinstance(ckpt_cpu.get("gru_encoder_path"), str)
            else args.encoder
        )

        # Reload data if scaler/feature cols differ from the preloaded window
        data_local = data_window
        if scaler_path != args.scaler or feature_cols_local != feature_cols_default:
            data_local = load_data_for_window(start, end, scaler_path, feature_cols_local)

        result = evaluate_checkpoint_four_curves(
            ckpt_path,
            data_by_symbol=data_local,
            feature_cols=feature_cols_local,
            device=args.device,
            base_seed=int(args.seed),
            encoder_path=encoder_path,
            top_k=int(args.top_k),
            weekly_n=int(args.weekly_n),
            fixed_basket=fixed_basket,
            start_t=None,
            end_t=None,
        )

        return result, data_local, feature_cols_local, scaler_path, encoder_path

    # --------------------------------------------------------
    # SINGLE CKPT MODE (no ranking)
    # --------------------------------------------------------
    if args.ckpt:
        # evaluate on validation window (single)
        val_data = load_data_for_window(args.val_start, args.val_end, args.scaler, feature_cols)
        result, data_used, feature_cols_used, scaler_used, encoder_used = eval_with_ckpt_metadata(
            args.ckpt, data_window=val_data, feature_cols_default=feature_cols, start=args.val_start, end=args.val_end
        )

        out_dir_val = os.path.join(args.out_dir, "val")
        out_json = default_out_path(out_dir_val, args.ckpt)
        _atomic_write_json(out_json, result)
        print(f"[OK] wrote validation JSON: {out_json}")
        return

    # --------------------------------------------------------
    # DIRECTORY MODE: (1) validation + ranking, (2) test best-2 + plots
    # --------------------------------------------------------

    ckpt_paths = list_checkpoints(args.ckpt_dir, include_latest=True)
    if not ckpt_paths:
        raise ValueError(f"No checkpoints found in {args.ckpt_dir}")

    out_dir_val = os.path.join(args.out_dir, "val")
    out_dir_test = os.path.join(args.out_dir, "test")
    out_dir_plots = os.path.join(out_dir_test, "plots")
    os.makedirs(out_dir_val, exist_ok=True)
    os.makedirs(out_dir_test, exist_ok=True)
    os.makedirs(out_dir_plots, exist_ok=True)

    print(f"[VAL] window={args.val_start}..{args.val_end}")

    # Preload validation data once (may be reloaded per-ckpt if scaler differs)
    val_data_default = load_data_for_window(args.val_start, args.val_end, args.scaler, feature_cols)

    ranking_rows: List[Dict[str, Any]] = []

    for ckpt_path in ckpt_paths:
        base = os.path.basename(ckpt_path)
        out_json = default_out_path(out_dir_val, ckpt_path)
        if os.path.exists(out_json):
            # If already exists, still include it in ranking by reading it.
            with open(out_json, "r") as f:
                result = json.load(f)
        else:
            result, _, _, _, _ = eval_with_ckpt_metadata(
                ckpt_path,
                data_window=val_data_default,
                feature_cols_default=feature_cols,
                start=args.val_start,
                end=args.val_end,
            )
            _atomic_write_json(out_json, result)

        # Score from FULL_DAILY
        scen = result.get("scenarios", {}).get("full_daily", {})
        m = scen.get("metrics", {}) if isinstance(scen, dict) else {}
        ep = scen.get("episode", {}) if isinstance(scen, dict) else {}
        metrics = {k: float(v) for k, v in m.items()} if isinstance(m, dict) else {}
        turnover = float(ep.get("turnover", 0.0)) if isinstance(ep, dict) else 0.0
        score = compute_validation_score(metrics, turnover)

        ranking_rows.append(
            {
                "checkpoint": base,
                "checkpoint_path": ckpt_path,
                "val_json": out_json,
                "score": float(score),
                "sharpe": float(metrics.get("sharpe", 0.0)),
                "total_return": float(metrics.get("total_return", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "turnover": float(turnover),
            }
        )

        print(
            f"[VAL] {base} score={score:.4f} sharpe={metrics.get('sharpe', 0.0):.4f} "
            f"ret={metrics.get('total_return', 0.0):.4f} mdd={metrics.get('max_drawdown', 0.0):.4f} "
            f"turnover={turnover:.4f}",
            flush=True,
        )

    # Rank and keep top-5
    ranking_rows = sorted(ranking_rows, key=lambda x: float(x["score"]), reverse=True)
    top5 = ranking_rows[:5]

    top5_payload = {
        "val_window": {"start": args.val_start, "end": args.val_end},
        "score_formula": "score = 1.00*sharpe + 0.50*total_return - 1.50*abs(max_drawdown) - 0.05*turnover",
        "top5": top5,
    }

    top5_path = os.path.join(out_dir_val, "top5.json")
    _atomic_write_json(top5_path, top5_payload)

    print("\n[TOP-5 checkpoints on validation]", flush=True)
    for i, row in enumerate(top5, 1):
        print(
            f"  {i}) {row['checkpoint']} score={row['score']:.4f} sharpe={row['sharpe']:.4f} "
            f"ret={row['total_return']:.4f} mdd={row['max_drawdown']:.4f} turnover={row['turnover']:.4f}",
            flush=True,
        )

    # Best-2 for test
    best2 = ranking_rows[:2]
    if len(best2) < 1:
        raise RuntimeError("No checkpoints available for test evaluation.")

    print(f"\n[TEST] window={args.test_start}..{args.test_end} (evaluating best {min(2, len(best2))})")

    # Preload test data once (may be reloaded per-ckpt if scaler differs)
    test_data_default = load_data_for_window(args.test_start, args.test_end, args.scaler, feature_cols)

    for row in best2:
        ckpt_path = row["checkpoint_path"]
        ckpt_base = os.path.basename(ckpt_path).replace(".pt", "")

        result, data_used, _, _, _ = eval_with_ckpt_metadata(
            ckpt_path,
            data_window=test_data_default,
            feature_cols_default=feature_cols,
            start=args.test_start,
            end=args.test_end,
        )

        out_json_test = default_out_path(out_dir_test, ckpt_path)
        _atomic_write_json(out_json_test, result)
        print(f"[OK] wrote test JSON: {out_json_test}")

        # Determine basket used by env reset (from the first scenario episode info)
        scen0 = result.get("scenarios", {}).get("full_daily", {})
        ep0 = scen0.get("episode", {}) if isinstance(scen0, dict) else {}
        symbols = list(ep0.get("symbols", [])) if isinstance(ep0, dict) else []
        if not symbols and fixed_basket is not None:
            symbols = fixed_basket

        if not symbols:
            print(f"[WARN] could not infer basket symbols for {ckpt_base}; skipping plots")
            continue


        # Determine episode index range from env episode info (use full_daily episode as reference)
        start_t_ep = int(ep0.get("start_t", 0))
        end_t_ep = int(ep0.get("end_t", 0))

        # Baseline 1: equal-weight buy & hold (BH) on the SAME basket
        bh_curve = equal_weight_buy_hold_curve(data_used, symbols, start_equity=1.0)

        # Momentum baselines require start/end indices; fall back to BH length-1 if end_t is missing.
        if end_t_ep <= 0:
            end_t_ep = len(bh_curve) - 1

        # Momentum baselines (Top-K, computed from same aligned data_used)
        if int(args.top_k) > 0:
            mom_fixed = momentum_topk_fixed_buy_hold_curve(
                data_used,
                symbols,
                start_t=start_t_ep,
                end_t=end_t_ep,
                lookback=int(args.mom_lookback),
                top_k=int(args.top_k),
                start_equity=1.0,
            )
            mom_dynamic_weekly = momentum_topk_dynamic_curve(
                data_used,
                symbols,
                start_t=start_t_ep,
                end_t=end_t_ep,
                lookback=int(args.mom_lookback),
                top_k=int(args.top_k),
                rebalance_every_n=int(args.weekly_n),
                start_equity=1.0,
            )
        else:
            mom_fixed = []
            mom_dynamic_weekly = []

        # --- (1) FULL scenarios vs BH equal-weight ---
        for scen_name, title in [
            ("full_daily", "BH vs PPO (full daily | 14 stocks)"),
            ("full_weekly", f"BH vs PPO (full weekly | 14 stocks, N={int(args.weekly_n)})"),
        ]:
            scen = result.get("scenarios", {}).get(scen_name, {})
            curve = scen.get("equity_curve", []) if isinstance(scen, dict) else []
            if not curve:
                continue

            plot_path = os.path.join(out_dir_plots, f"{ckpt_base}__{scen_name}.png")
            plot_compare_curves(
                out_path=plot_path,
                title=title,
                curves=[
                    ("Buy & Hold (equal-weight)", bh_curve),
                    (f"PPO ({scen_name})", curve),
                ],
            )

        # --- (2) TOP-K scenarios vs FIXED momentum baseline ---
        if mom_fixed:
            for scen_name, title in [
                ("topk_daily", f"Momentum-fixed vs PPO (topk daily, k={int(args.top_k)}, lookback={int(args.mom_lookback)})"),
                ("topk_weekly", f"Momentum-fixed vs PPO (topk weekly, k={int(args.top_k)}, N={int(args.weekly_n)}, lookback={int(args.mom_lookback)})"),
            ]:
                scen = result.get("scenarios", {}).get(scen_name, {})
                curve = scen.get("equity_curve", []) if isinstance(scen, dict) else []
                if not curve:
                    continue

                plot_path = os.path.join(out_dir_plots, f"{ckpt_base}__{scen_name}__mom_fixed.png")
                plot_compare_curves(
                    out_path=plot_path,
                    title=title,
                    curves=[
                        ("Momentum Top-K (fixed buy&hold)", mom_fixed),
                        (f"PPO ({scen_name})", curve),
                    ],
                )

        # --- (3) Extra: dynamic momentum vs PPO topk_weekly ---
        if mom_dynamic_weekly:
            scen = result.get("scenarios", {}).get("topk_weekly", {})
            curve = scen.get("equity_curve", []) if isinstance(scen, dict) else []
            if curve:
                plot_path = os.path.join(out_dir_plots, f"{ckpt_base}__topk_weekly__mom_dynamic.png")
                plot_compare_curves(
                    out_path=plot_path,
                    title=f"Dynamic momentum vs PPO (topk weekly, k={int(args.top_k)}, N={int(args.weekly_n)}, lookback={int(args.mom_lookback)})",
                    curves=[
                        ("Momentum Top-K (dynamic weekly)", mom_dynamic_weekly),
                        (f"PPO (topk_weekly)", curve),
                    ],
                )

        print(f"[OK] wrote plots for {ckpt_base} -> {out_dir_plots}")


if __name__ == "__main__":
    main()
