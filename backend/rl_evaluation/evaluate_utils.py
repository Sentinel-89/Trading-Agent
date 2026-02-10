import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(equity):
    equity = np.asarray(equity, dtype=np.float64)

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