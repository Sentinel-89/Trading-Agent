# backend/analysis/plot_phase_c_actions.py
#
# Plot Phase-C action distributions (diagnostic)
#
# Purpose:
# - Visualize BUY / SELL / HOLD behavior per run
# - Compare policy behavior across envs, regimes, and seeds
#
# NOTE:
# - Diagnostic-only (not learning curves)
# - One plot per run
#

import os
import json
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACT_DIR_SINGLE = os.path.join(BASE_DIR, "artifacts", "phase_c")
ARTIFACT_DIR_MULTI  = os.path.join(BASE_DIR, "artifacts", "phase_c_multi")

PLOT_DIR = os.path.join(BASE_DIR, "artifacts", "plots", "phase_c_actions")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# Runs
# ============================================================

RUNS = [
    # Single-symbol
    ("Single v3 (Seed 0)",   ARTIFACT_DIR_SINGLE, "v3/phase_c_v3_seed0.json"),
    ("Single v3 (Seed 1)",   ARTIFACT_DIR_SINGLE, "v3/phase_c_v3_seed1.json"),
    ("Single v4.1 (Seed 0)", ARTIFACT_DIR_SINGLE, "v4.1/phase_c_v4.1_seed0.json"),
    ("Single v4.1 (Seed 1)", ARTIFACT_DIR_SINGLE, "v4.1/phase_c_v4.1_seed1.json"),

    # Multi-symbol
    ("Multi v3 (Seed 0)",    ARTIFACT_DIR_MULTI,  "v3/phase_c_multi_v3_seed0.json"),
    ("Multi v3 (Seed 1)",    ARTIFACT_DIR_MULTI,  "v3/phase_c_multi_v3_seed1.json"),
    ("Multi v4.1 (Seed 0)",  ARTIFACT_DIR_MULTI,  "v4.1/phase_c_multi_v4.1_seed0.json"),
    ("Multi v4.1 (Seed 1)",  ARTIFACT_DIR_MULTI,  "v4.1/phase_c_multi_v4.1_seed1.json"),
]

# ============================================================
# Action colors (fixed, semantic)
# ============================================================

ACTION_COLORS = {
    "BUY":  "#2ca02c",  # green
    "SELL": "#d62728",  # red
    "HOLD": "#7f7f7f",  # gray
}

# ============================================================
# Loader
# ============================================================

def load_actions(path):
    with open(path, "r") as f:
        data = json.load(f)

    eps = data["episodes"]
    return {
        "episode": [e["episode"] for e in eps],
        "BUY":  [e["action_freq"]["BUY"]  for e in eps],
        "SELL": [e["action_freq"]["SELL"] for e in eps],
        "HOLD": [e["action_freq"]["HOLD"] for e in eps],
    }

# ============================================================
# Plot per run
# ============================================================

for label, base, rel in RUNS:
    path = os.path.join(base, rel)
    if not os.path.exists(path):
        continue

    print(f"Plotting actions: {label}")
    d = load_actions(path)

    plt.figure(figsize=(10, 6))

    for action in ["BUY", "SELL", "HOLD"]:
        plt.plot(
            d["episode"],
            d[action],
            label=action,
            color=ACTION_COLORS[action],
            lw=2 if action != "HOLD" else 1.5,
            alpha=0.9,
        )

    plt.title(f"Phase-C — Action Distribution ({label})")
    plt.xlabel("Episode")
    plt.ylabel("Action Frequency")
    plt.legend()
    plt.grid(alpha=0.3)

    safe_label = (
        label.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    plt.savefig(
        os.path.join(PLOT_DIR, f"phase_c_actions_{safe_label}.png"),
        dpi=300,
        bbox_inches="tight",
    )

plt.show()
