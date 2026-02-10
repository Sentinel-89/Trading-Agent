
"""backend/rl/evaluation_runner_continuous.py

Deterministic evaluation for CONTINUOUS portfolio PPO checkpoints.

What it does
------------
For each checkpoint, this script runs a *deterministic* backtest (policy mean action)
over a chosen test date range and writes ONE JSON result per checkpoint.

For each checkpoint we produce **four** equity curves:
  1) full_daily   : no top-k restriction, rebalance every step
  2) full_weekly  : no top-k restriction, rebalance every N steps (default N=5)
  3) topk_daily   : enforce top-k holdings, rebalance every step
  4) topk_weekly  : enforce top-k holdings, rebalance every N steps

Why four curves?
---------------
Training is usually done with daily rebalancing and no sparsity constraints.
In production/front-end you might want: "hold max k positions" and/or "rebalance weekly".
These are *evaluation-time knobs* (no retraining required) supported by TradingEnvContinuous.

Correctness rule
----------------
PnL must be computed from RAW prices. We preserve Close_raw/Open_raw from CSVs.
Feature columns are scaled for the GRU encoder, but returns use Close_raw.

Outputs
-------
One JSON per checkpoint with:
- checkpoint metadata
- environment config used for evaluation
- basket/time slice used
- for each scenario: equity curve + metrics + turnover/rebalances

Notes
-----
- This evaluator is deterministic given (seed, date range, basket selection).
- By default it runs a single deterministic episode per checkpoint (one basket).
  If you want multiple baskets/episodes per checkpoint, extend `run_episode_specs`.
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

from backend.rl_evaluation.evaluate_utils import compute_metrics
from backend.rl.multiasset.ppo_actor_critic_continuous import PPOActorCriticContinuous
from backend.rl.multiasset.trading_env_continuous import TradingEnvContinuous, TradingEnvContinuousConfig


# ============================================================
# Defaults (override via CLI)
# ============================================================

DEFAULT_TEST_START = "2024-06-01"
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
    test_start: str,
    test_end: str,
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
            df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].reset_index(drop=True)
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.loc[test_start:test_end]

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
# Metrics helpers
# ============================================================

def total_return(curve: List[float]) -> float:
    if not curve or len(curve) < 2:
        return 0.0
    a = float(curve[0])
    b = float(curve[-1])
    return 0.0 if a == 0.0 else (b / a) - 1.0


def max_drawdown(curve: List[float]) -> float:
    if not curve or len(curve) < 2:
        return 0.0
    arr = np.asarray(curve, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    dd = (arr / np.maximum(running_max, 1e-12)) - 1.0
    return float(np.min(dd))


def sharpe(curve: List[float], periods: int = 252) -> float:
    if not curve or len(curve) < 3:
        return 0.0
    arr = np.asarray(curve, dtype=np.float64)
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(float(periods)))


def canonicalize_metrics(m: Any, curve: List[float]) -> Dict[str, float]:
    """Normalize metric names and ensure key metrics exist."""
    out: Dict[str, float] = {}

    if isinstance(m, dict):
        for k, v in m.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue

    # aliases
    if "total_return" not in out:
        for alt in ["cumulative_return", "cum_return", "return"]:
            if alt in out:
                out["total_return"] = float(out[alt])
                break

    if "max_drawdown" not in out:
        for alt in ["mdd", "max_dd", "drawdown"]:
            if alt in out:
                out["max_drawdown"] = float(out[alt])
                break

    if "sharpe" not in out:
        for alt in ["sharpe_ratio", "sr"]:
            if alt in out:
                out["sharpe"] = float(out[alt])
                break

    # ensure values exist
    out.setdefault("total_return", total_return(curve))
    out.setdefault("max_drawdown", max_drawdown(curve))
    out.setdefault("sharpe", sharpe(curve))

    return out


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

    # We keep window_length/episode_length from checkpoint unless user explicitly slices start/end.
    # NOTE: In TradingEnvContinuous, `window_length` means warmup/burn-in (min start index).

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
    # - If fixed_basket provided, we use it (must match num_assets)
    # - Else, rely on env's internal random choice, but seeded => deterministic.
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
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", default="", help="Path to a single checkpoint .pt")
    parser.add_argument("--ckpt_dir", default="", help="Directory of checkpoints to evaluate")
    parser.add_argument("--watch", action="store_true", help="Watch ckpt_dir for new checkpoints")
    parser.add_argument("--poll_secs", type=int, default=10)

    parser.add_argument(
        "--out_dir",
        default=os.path.join("backend", "artifacts", "phase_d_continuous", "eval"),
        help="Directory to write per-checkpoint JSON results",
    )
    parser.add_argument("--out", default="", help="Output JSON path (only used with --ckpt)")

    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--scaler", default=os.path.join("backend", "models", "checkpoints", "scaler.pkl"))
    parser.add_argument("--test_start", default=DEFAULT_TEST_START)
    parser.add_argument("--test_end", default=DEFAULT_TEST_END)
    parser.add_argument(
        "--feature_cols",
        default="Open,Close,RSI,MACD,MACD_Signal,ATR,SMA_50,SMA_Ratio,OBV,ROC_10,RealizedVol_20",
        help="Comma-separated feature columns to scale",
    )
    parser.add_argument("--encoder", default=os.path.join("backend", "models", "checkpoints", "gru_encoder.pt"))

    # Evaluation knobs
    parser.add_argument("--top_k", type=int, default=0, help="k for top-k scenarios (0 disables top-k)")
    parser.add_argument("--weekly_n", type=int, default=5, help="Rebalance interval for weekly scenarios")
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
    parser.add_argument(
        "--start_t",
        type=int,
        default=-1,
        help="Optional eval start index (over aligned test data). -1 means default earliest valid.",
    )
    parser.add_argument(
        "--end_t",
        type=int,
        default=-1,
        help="Optional eval end index INCLUSIVE (over aligned test data). -1 means run to end.",
    )

    args = parser.parse_args()

    if not args.ckpt and not args.ckpt_dir:
        raise ValueError("Provide either --ckpt or --ckpt_dir")

    fixed_basket = [s.strip() for s in args.basket.split(",") if s.strip()] or None

    feature_cols = [c.strip() for c in str(args.feature_cols).split(",") if c.strip()]

    # Load data once (shared across checkpoints)
    print(
        f"[data] dir={args.data_dir} test={args.test_start}..{args.test_end} "
        f"scaler={args.scaler} feature_cols={feature_cols}",
        flush=True,
    )
    data_by_symbol = load_and_prepare_data(
        data_dir=args.data_dir,
        scaler_path=args.scaler,
        feature_cols=feature_cols,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    start_t = None if int(args.start_t) < 0 else int(args.start_t)
    end_t = None if int(args.end_t) < 0 else int(args.end_t)

    def eval_one(ckpt_path: str, out_path: str) -> None:
        # Prefer checkpoint-provided metadata when available (feature/scaler/encoder)
        ckpt_cpu = torch.load(ckpt_path, map_location="cpu")

        ckpt_feature_cols = ckpt_cpu.get("feature_cols")
        feature_cols_local = (
            [c.strip() for c in ckpt_feature_cols]
            if isinstance(ckpt_feature_cols, (list, tuple))
            else [c.strip() for c in str(ckpt_feature_cols).split(",") if c.strip()]
            if isinstance(ckpt_feature_cols, str)
            else feature_cols
        )

        scaler_path = ckpt_cpu.get("scaler_path") if isinstance(ckpt_cpu.get("scaler_path"), str) else args.scaler
        encoder_path = ckpt_cpu.get("gru_encoder_path") if isinstance(ckpt_cpu.get("gru_encoder_path"), str) else args.encoder

        # If checkpoint feature_cols/scaler differ, we must reload data
        data_local = data_by_symbol
        if feature_cols_local != feature_cols or scaler_path != args.scaler:
            print(
                f"[data] reload for ckpt metadata: scaler={scaler_path} feature_cols={feature_cols_local}",
                flush=True,
            )
            data_local = load_and_prepare_data(
                data_dir=args.data_dir,
                scaler_path=scaler_path,
                feature_cols=feature_cols_local,
                test_start=args.test_start,
                test_end=args.test_end,
            )

        print(f"[Eval] ckpt={ckpt_path} -> out={out_path}", flush=True)

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
            start_t=start_t,
            end_t=end_t,
        )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _atomic_write_json(out_path, result)
        print(f"[OK] wrote {out_path}", flush=True)

    # Single checkpoint
    if args.ckpt:
        out_path = args.out or default_out_path(args.out_dir, args.ckpt)
        eval_one(args.ckpt, out_path)
        return

    # Directory mode
    os.makedirs(args.out_dir, exist_ok=True)

    def run_dir_once() -> None:
        for ckpt_path in list_checkpoints(args.ckpt_dir, include_latest=True):
            out_path = default_out_path(args.out_dir, ckpt_path)
            if os.path.exists(out_path):
                continue
            eval_one(ckpt_path, out_path)

    if not args.watch:
        run_dir_once()
        return

    # Watch mode
    print(f"[watch] ckpt_dir={args.ckpt_dir} -> out_dir={args.out_dir} poll_secs={args.poll_secs}")
    while True:
        run_dir_once()
        time.sleep(max(1, int(args.poll_secs)))


if __name__ == "__main__":
    main()
