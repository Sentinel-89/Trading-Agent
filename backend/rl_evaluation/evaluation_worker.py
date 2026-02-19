import os
import time
import torch
import json
import joblib
import numpy as np
import pandas as pd
import random
import argparse
from typing import Dict, Any, Tuple, List

from gymnasium.vector import SyncVectorEnv
from backend.rl.trading_env import TradingEnv
from backend.rl.ppo_actor_critic import PPOActorCritic
from backend.rl_evaluation.evaluate_utils import compute_metrics, plot_equity_curves


def normalize_infos(infos, n_envs: int):
    """
    Gymnasium VectorEnv may return:
      - list[dict] (one dict per env), OR
      - dict[str, array-like] (one key with per-env values)
    Normalize to list[dict] of length n_envs.
    """
    if isinstance(infos, list):
        return infos

    if isinstance(infos, dict):
        out = [dict() for _ in range(n_envs)]
        for k, v in infos.items():
            try:
                for i in range(n_envs):
                    out[i][k] = v[i]
            except Exception:
                # scalar or non-indexable: copy to all
                for i in range(n_envs):
                    out[i][k] = v
        return out

    raise TypeError(f"Unsupported infos type: {type(infos)}")


CHECK_INTERVAL = 10  # seconds
EVAL_ENVS = 14
VAL_START: str   = "2024-06-01"
VAL_END: str     = "2024-12-30"
TEST_START: str   = "2025-01-01"
TEST_END: str     = "2025-12-30"


def load_data(data_dir, scaler_path, feature_cols, config, start_date: str, end_date: str):
    scaler = joblib.load(scaler_path)
    data = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            sym = fname.replace("_labeled.csv", "")
            df = pd.read_csv(os.path.join(data_dir, fname), parse_dates=["Date"])
            df.rename(columns={"Date": "date"}, inplace=True)
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)

            # Preserve raw prices for evaluation baselines
            if "Close_raw" not in df.columns and "Close" in df.columns:
                df["Close_raw"] = df["Close"].astype(float)

            # Avoid sklearn warning about feature names (scaler fitted without names)
            df[feature_cols] = scaler.transform(df[feature_cols].to_numpy())
            data[sym] = df
    return data


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, data_by_symbol, encoder_ckpt, feature_cols, device, use_action_mask: bool = False):
    SEED = 12345
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("[Eval] Deterministic evaluation enabled (fixed seeds, deterministic policy)")

    symbols = sorted(data_by_symbol.keys())[:EVAL_ENVS]

    def make_env(symbol):
        def _init():
            env = TradingEnv(
                data_by_symbol={symbol: data_by_symbol[symbol]},  # Single-symbol environment
                encoder_ckpt_path=encoder_ckpt,
                feature_cols=feature_cols,
                env_version="v4",
                episode_mode="full_history",
                random_start=False,
                device="cpu",
            )
            return env
        return _init
    # Some gymnasium versions reject autoreset_mode="disabled".
    # Fall back to default and ignore post-done data to avoid autoreset contamination.
    try:
        env = SyncVectorEnv([make_env(s) for s in symbols], autoreset_mode="disabled")
    except (TypeError, ValueError):
        env = SyncVectorEnv([make_env(s) for s in symbols])

    obs, infos = env.reset(seed=SEED)
    infos = normalize_infos(infos, len(symbols))

    if obs.ndim != 2:
        raise RuntimeError(f"[Eval] Expected obs shape (n_envs, obs_dim), got {obs.shape}")

    # --- Sanity check: explicit symbol ↔ env binding ---
    for i, s in enumerate(symbols):
        env_sym = infos[i].get("symbol") if isinstance(infos[i], dict) else None
        if env_sym != s:
            raise RuntimeError(
                f"[Eval] Symbol mismatch in env {i}: expected={s}, got={env_sym}"
            )

    def extract_action_masks(infos, device):
        """Prefer `true_action_mask` (always valid), fall back to `action_mask` if present."""
        masks = []
        for info in infos:
            if not isinstance(info, dict):
                raise TypeError(f"[Eval] info must be dict, got {type(info)}")
            if "true_action_mask" in info:
                masks.append(info["true_action_mask"])
            elif "action_mask" in info:
                masks.append(info["action_mask"])
            else:
                raise KeyError("[Eval] Missing `true_action_mask`/`action_mask` in env info during evaluation")
        return torch.tensor(np.stack(masks), dtype=torch.float32, device=device)

    action_masks = extract_action_masks(infos, device) if use_action_mask else None

    policy = PPOActorCritic(obs_dim=obs.shape[1]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    assert not policy.training, "Policy must be in eval() mode for evaluation"

    done = np.zeros(len(symbols), dtype=bool)
    equity_curves = [[] for _ in symbols]
    peak_equity = [None for _ in symbols]
    max_drawdowns = [0.0 for _ in symbols]

    trade_stats = {
        s: {
            "chosen_buys": 0,
            "chosen_sells": 0,
            "chosen_holds": 0,
            "executed_buys": 0,
            "executed_sells": 0,
            "invalid_actions": 0,
        }
        for s in symbols
    }

    while not done.all():
        latent = torch.tensor(obs, dtype=torch.float32, device=device)
        actions, _, _ = policy.act(
            latent,
            action_masks if use_action_mask else None,
            deterministic=True,
        )

        obs, _, terminated, truncated, infos = env.step(actions.cpu().numpy())
        infos = normalize_infos(infos, len(symbols))
        action_masks = extract_action_masks(infos, device) if use_action_mask else None

        # Keep track of which envs were already finished before this step
        done_prev = done.copy()
        done |= (terminated | truncated)

        for i in range(len(symbols)):
            # If this env already finished earlier, ignore any post-done data (autoreset can restart it)
            if done_prev[i]:
                continue

            if "equity" not in infos[i]:
                raise KeyError(f"[Eval] Missing `equity` in info for env {i}")
            eq = infos[i]["equity"]
            equity_curves[i].append(eq)

            if peak_equity[i] is None:
                peak_equity[i] = eq
            else:
                peak_equity[i] = max(peak_equity[i], eq)

            dd = (eq - peak_equity[i]) / peak_equity[i]
            max_drawdowns[i] = min(max_drawdowns[i], dd)

            # chosen actions
            act = int(actions[i].item())
            if act == 0:
                trade_stats[symbols[i]]["chosen_holds"] += 1
            elif act == 1:
                trade_stats[symbols[i]]["chosen_buys"] += 1
            elif act == 2:
                trade_stats[symbols[i]]["chosen_sells"] += 1

            # executed actions / validity (from env)
            info = infos[i]
            trade_stats[symbols[i]]["executed_buys"] = int(info.get("executed_buys", trade_stats[symbols[i]]["executed_buys"]))
            trade_stats[symbols[i]]["executed_sells"] = int(info.get("executed_sells", trade_stats[symbols[i]]["executed_sells"]))
            trade_stats[symbols[i]]["invalid_actions"] = int(info.get("invalid_actions", trade_stats[symbols[i]]["invalid_actions"]))

    return symbols, equity_curves, trade_stats, max_drawdowns


def score_checkpoint(metrics_by_symbol: Dict[str, Any], metric: str = "excess_return", dd_penalty: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """Return (aggregate_score, per_symbol_scores).

    metric:
      - "excess_return": uses `excess_return_vs_bh` if present else `return`
      - "sharpe": uses `sharpe` if present else 0
      - "return": uses `return` if present else 0

    Aggregate is the mean over symbols of: base_metric - dd_penalty * abs(max_drawdown)
    """
    per_sym: Dict[str, float] = {}
    scores: List[float] = []

    for sym, m in metrics_by_symbol.items():
        try:
            dd = float(m.get("max_drawdown", 0.0))
        except Exception:
            dd = 0.0
        dd = abs(dd)

        base = 0.0
        if metric == "excess_return":
            base = float(m.get("excess_return_vs_bh", m.get("return", 0.0)))
        elif metric == "sharpe":
            base = float(m.get("sharpe", 0.0))
        elif metric == "return":
            base = float(m.get("return", 0.0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        s = base - dd_penalty * dd
        per_sym[sym] = s
        scores.append(s)

    agg = float(np.mean(scores)) if len(scores) else float("-inf")
    return agg, per_sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="backend/artifacts/phase_d/mask_off/seed_0")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Select evaluation window")
    ap.add_argument("--topk", type=int, default=5, help="How many best checkpoints to print")
    ap.add_argument("--score_metric", choices=["excess_return", "sharpe", "return"], default="excess_return")
    ap.add_argument("--dd_penalty", type=float, default=1.0, help="Penalty multiplier on |max_drawdown|")
    ap.add_argument("--start_date", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    ap.add_argument("--end_date", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    args = ap.parse_args()

    base_dir = args.base_dir
    ckpt_dir = os.path.join(base_dir, "checkpoints")

    start_date = VAL_START if args.split == "val" else TEST_START
    end_date = VAL_END if args.split == "val" else TEST_END

    if args.start_date is not None:
        start_date = args.start_date
    if args.end_date is not None:
        end_date = args.end_date

    log_dir = os.path.join(base_dir, f"eval_{args.split}_{start_date}_to_{end_date}")
    os.makedirs(log_dir, exist_ok=True)

    feature_cols = [
        "Open",
        "Close",
        "RSI",
        "MACD",
        "MACD_Signal",
        "ATR",
        "SMA_50",
        "OBV",
        "ROC_10",
        "SMA_Ratio",
        "RealizedVol_20",
    ]
        
    data = load_data(
        "backend/data/processed",
        "backend/models/checkpoints/scaler.pkl",
        feature_cols,
        config=None,
        start_date=start_date,
        end_date=end_date,
    )

    for sym, df in data.items():
        print(f"[Data] {sym}: min date = {df['date'].min()}, max date = {df['date'].max()}")

    # Align baseline with env start (TradingEnv uses a 30-bar window => starts at step 29)
    START_STEP = 29

    buy_hold_curves = {}
    for sym, df in data.items():
        if "Close_raw" not in df.columns:
            raise KeyError(f"[Eval] Missing Close_raw for {sym}; ensure load_data preserves raw prices")
        prices = df["Close_raw"].values.astype(np.float64)
        if len(prices) <= START_STEP:
            continue
        equity = prices[START_STEP:] / prices[START_STEP]
        buy_hold_curves[sym] = equity.tolist()

    # Plot buy-and-hold baseline ONCE (it does not depend on checkpoint)
    plot_equity_curves(
        [buy_hold_curves[s] for s in sorted(buy_hold_curves.keys())[:EVAL_ENVS]],
        sorted(buy_hold_curves.keys())[:EVAL_ENVS],
        os.path.join(log_dir, "buy_hold.png"),
        percent=True,
    )

    seen = set()
    leaderboard: List[Tuple[float, str]] = []  # (score, ckpt_filename)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))

        for ckpt in ckpts:
            if ckpt in seen:
                continue

            path = os.path.join(ckpt_dir, ckpt)
            print(f"[Eval] Evaluating {ckpt}")

            ckpt_data = torch.load(path, map_location=device)
            use_action_mask = bool(ckpt_data.get("use_action_mask", False))

            print(
                f"[Eval] Action masking enabled (from checkpoint metadata): {use_action_mask}"
            )

            symbols, curves, trade_stats, max_drawdowns = evaluate_checkpoint(
                path,
                data,
                "backend/models/checkpoints/gru_encoder.pt",
                feature_cols,
                device,
                use_action_mask=use_action_mask,
            )

            metrics = {}
            for s, c in zip(symbols, curves):
                m = compute_metrics(c)
                bh = compute_metrics(buy_hold_curves[s])
                m["buy_hold"] = bh
                m["excess_return_vs_bh"] = m["return"] - bh["return"]
                m["max_drawdown"] = max_drawdowns[symbols.index(s)]
                m["trades"] = {
                    "chosen_buys": trade_stats[s]["chosen_buys"],
                    "chosen_sells": trade_stats[s]["chosen_sells"],
                    "chosen_holds": trade_stats[s]["chosen_holds"],
                    "executed_buys": trade_stats[s]["executed_buys"],
                    "executed_sells": trade_stats[s]["executed_sells"],
                    "invalid_actions": trade_stats[s]["invalid_actions"],
                }
                metrics[s] = m

            # Save per-symbol metrics
            with open(os.path.join(log_dir, ckpt.replace(".pt", ".json")), "w") as f:
                json.dump(metrics, f, indent=2)

            # Plot equity curves
            plot_equity_curves(
                curves,
                symbols,
                os.path.join(log_dir, ckpt.replace(".pt", ".png")),
                percent=True,
            )

            # Score checkpoint and update leaderboard
            ckpt_score, _ = score_checkpoint(metrics, metric=args.score_metric, dd_penalty=args.dd_penalty)
            leaderboard.append((ckpt_score, ckpt))
            leaderboard.sort(key=lambda x: x[0], reverse=True)

            topk = leaderboard[: max(1, args.topk)]
            print("\n[Eval] ===== Running Top Checkpoints =====")
            for rank, (sc, name) in enumerate(topk, start=1):
                print(f"[Eval] #{rank:02d}  score={sc:+.6f}  ckpt={name}")
            print("[Eval] ===================================\n")

            # Persist leaderboard
            with open(os.path.join(log_dir, "leaderboard.json"), "w") as f:
                json.dump(
                    {
                        "score_metric": args.score_metric,
                        "dd_penalty": args.dd_penalty,
                        "split": args.split,
                        "start_date": start_date,
                        "end_date": end_date,
                        "topk": [{"rank": i + 1, "checkpoint": name, "score": sc} for i, (sc, name) in enumerate(topk)],
                    },
                    f,
                    indent=2,
                )

            seen.add(ckpt)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
