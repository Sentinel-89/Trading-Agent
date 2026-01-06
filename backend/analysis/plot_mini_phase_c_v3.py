# backend/analysis/plot_mini_phase_c_v3.py
#
# Plot Mini-Phase-C v3 results (short vs extended runs)
#
# Purpose:
# - Compare early learning stability (5 episodes)
# - Visualize behavioral trends (25 episodes)
#
# NOTE:
# - This script is read-only
# - It consumes experiment artifacts only
# - No training code depends on this file
#

import os
import json
import matplotlib.pyplot as plt

# ============================================================
# Resolve paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts",
    "mini_phase_c",
)

RUNS = [
    {
        "label": "Mini-Phase-C (5 episodes)",
        "filename": "mini_phase_c_v3_20260106_120213.json",
    },
    {
        "label": "Mini-Phase-C (25 episodes)",
        "filename": "mini_phase_c_v3_20260106_135337.json",
    },
]

# ============================================================
# Helper: load run
# ============================================================

def load_run(path):
    with open(path, "r") as f:
        data = json.load(f)

    episodes = data["episodes"]

    return {
        "episode": [e["episode"] for e in episodes],
        "reward": [e["reward"] for e in episodes],
        "entropy": [e["entropy"] for e in episodes],
        "buy": [e["action_freq"]["BUY"] for e in episodes],
        "sell": [e["action_freq"]["SELL"] for e in episodes],
        "hold": [e["action_freq"]["HOLD"] for e in episodes],
    }

# ============================================================
# Load all runs
# ============================================================

runs_data = []

for run in RUNS:
    path = os.path.join(ARTIFACT_DIR, run["filename"])
    runs_data.append(
        {
            "label": run["label"],
            "data": load_run(path),
        }
    )

# ============================================================
# Plot 1: Reward per episode
# ============================================================

plt.figure()

for run in runs_data:
    d = run["data"]
    plt.plot(
        d["episode"],
        d["reward"],
        marker="o",
        label=run["label"],
    )

plt.xlabel("Episode")
plt.ylabel("Total Episode Reward")
plt.title("Mini-Phase-C v3 — Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Plot 2: Entropy per episode
# ============================================================

plt.figure()

for run in runs_data:
    d = run["data"]
    plt.plot(
        d["episode"],
        d["entropy"],
        marker="o",
        label=run["label"],
    )

plt.xlabel("Episode")
plt.ylabel("Policy Entropy")
plt.title("Mini-Phase-C v3 — Entropy Evolution")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Plot 3: Action frequencies (separate figure per run)
# ============================================================

for run in runs_data:
    d = run["data"]

    plt.figure()
    plt.plot(d["episode"], d["buy"], label="BUY")
    plt.plot(d["episode"], d["sell"], label="SELL")
    plt.plot(d["episode"], d["hold"], label="HOLD")

    plt.xlabel("Episode")
    plt.ylabel("Action Frequency")
    plt.title(f"Mini-Phase-C v3 — Action Distribution ({run['label']})")
    plt.legend()
    plt.grid(True)
    plt.show()
