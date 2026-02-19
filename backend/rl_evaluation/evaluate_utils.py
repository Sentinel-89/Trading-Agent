import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List


def compute_metrics(equity):
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size < 2:
        return {"return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    # Total return
    total_return = equity[-1] - 1.0

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # Log returns
    log_returns = np.diff(np.log(equity))

    mean = log_returns.mean()
    std = log_returns.std()

    # Variance guard (prevents fake infinite Sharpe)
    if std < 1e-6 or len(log_returns) < 2:
        sharpe = 0.0
    else:
        sharpe = (mean / std) * np.sqrt(252)

        # Mild autocorrelation awareness (not full Lo penalty)
        rho1 = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        if not np.isnan(rho1):
            sharpe = sharpe / (1.0 + 2.0 * abs(rho1))

    return {
        "return": float(total_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
    }


def _total_return(curve: List[float]) -> float:
    if not curve or len(curve) < 2:
        return 0.0
    a = float(curve[0])
    b = float(curve[-1])
    return 0.0 if a == 0.0 else (b / a) - 1.0


def _max_drawdown(curve: List[float]) -> float:
    if not curve or len(curve) < 2:
        return 0.0
    arr = np.asarray(curve, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    dd = (arr / np.maximum(running_max, 1e-12)) - 1.0
    return float(np.min(dd))


def _sharpe(curve: List[float], periods: int = 252) -> float:
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
    out: Dict[str, float] = {}

    if isinstance(m, dict):
        for k, v in m.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue

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

    out.setdefault("total_return", _total_return(curve))
    out.setdefault("max_drawdown", _max_drawdown(curve))
    out.setdefault("sharpe", _sharpe(curve))
    return out


def plot_equity_curves(curves, symbols, out_path, percent: bool = False):
    if percent:
        curves = [[(v - 1.0) * 100.0 for v in c] for c in curves]
        ylabel = "Return (%)"
    else:
        ylabel = "Equity"
    plt.figure(figsize=(10, 6))
    for c, s in zip(curves, symbols):
        plt.plot(c, label=s)

    plt.legend()
    plt.title("Equity Curves (Deterministic Eval)")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
