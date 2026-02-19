import os
import json
import glob
import math

EVAL_DIR = "backend/artifacts/phase_d/mask_on/seed_0/eval"  # <-- change if needed
TOP_K = 10

# Prefer executed trades from the env (more meaningful than chosen actions)
USE_EXECUTED_TRADES = True

# If very few symbols trade, don't hard-filter everything; we already penalize via diversification
MIN_ACTIVE_SYMBOL_FRACTION = 0.25  # 25% of symbols must trade to avoid being discarded

def safe_mean(xs):
    xs2 = []
    for x in xs:
        if x is None:
            continue
        try:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                continue
        except Exception:
            pass
        xs2.append(x)
    return sum(xs2) / len(xs2) if xs2 else None

results = []
json_paths = glob.glob(os.path.join(EVAL_DIR, "*.json"))
print(f"[Rank] Found {len(json_paths)} json files in {EVAL_DIR}")

for path in json_paths:
    with open(path, "r") as f:
        data = json.load(f)

    total_buys = 0
    sharpes = []
    returns = []
    trades = []

    total_symbols = len(data)
    active_symbols = 0

    for symbol, metrics in data.items():
        t = metrics.get("trades", {})
        if USE_EXECUTED_TRADES:
            buys = int(t.get("executed_buys", 0))
            sells = int(t.get("executed_sells", 0))
        else:
            buys = int(t.get("chosen_buys", 0))
            sells = int(t.get("chosen_sells", 0))

        # Ignore symbols that never executed a trade
        if (buys + sells) == 0:
            continue

        active_symbols += 1

        total_buys += buys

        sharpes.append(metrics.get("sharpe"))
        returns.append(metrics.get("return"))
        trades.append(buys + sells)

    # --------------------------------------------------
    # FILTER 1: must trade on multiple symbols
    # --------------------------------------------------
    MIN_ACTIVE_SYMBOLS = max(1, int(math.ceil(total_symbols * MIN_ACTIVE_SYMBOL_FRACTION)))

    if active_symbols < MIN_ACTIVE_SYMBOLS:
        # Too inactive across symbols; skip this checkpoint entirely
        continue

    mean_sharpe = safe_mean(sharpes)
    mean_return = safe_mean(returns)
    mean_trades = safe_mean(trades)


    # --------------------------------------------------
    # Composite score (simple & robust)
    # --------------------------------------------------
    # --------------------------------------------------
    # Composite score (base)
    # --------------------------------------------------
    base_score = (
        (mean_sharpe if mean_sharpe is not None else -1e9)
        + 0.3 * (mean_return if mean_return is not None else 0.0)
    )

    # Diversification factor (0–1)
    diversification = active_symbols / total_symbols

    score = base_score * diversification

    # --------------------------------------------------
    # Inactivity penalty
    # --------------------------------------------------
    MIN_TRADES = 30

    if mean_trades < MIN_TRADES:
        score *= (mean_trades / MIN_TRADES) ** 2

    results.append({
        "file": os.path.basename(path),
        "score": score,
        "mean_sharpe": mean_sharpe,
        "mean_return": mean_return,
        "mean_trades": mean_trades,
        "total_buys": total_buys,
        "active_symbols": active_symbols,
        "total_symbols": total_symbols,
    })

# ------------------------------------------------------
# Sort & display
# ------------------------------------------------------
results.sort(key=lambda x: x["score"], reverse=True)

print(f"\nTop {TOP_K} trading checkpoints:\n")

for r in results[:TOP_K]:
    print(
        f"{r['file']}\n"
        f"  score        : {r['score']:.4f}\n"
        f"  mean sharpe  : {r['mean_sharpe']:.4f}\n"
        f"  mean return  : {r['mean_return']:.4f}\n"
        f"  mean trades  : {r['mean_trades']:.1f}\n"
        f"  total buys   : {r['total_buys']}\n"
        f"  active symbols: {r['active_symbols']}/{r['total_symbols']}\n"
    )

if not results:
    if len(json_paths) == 0:
        print("❌ No JSON files found. Check EVAL_DIR path.")
    else:
        print("❌ No checkpoints passed filters (likely no executed trades or MIN_ACTIVE_SYMBOL_FRACTION too high).")