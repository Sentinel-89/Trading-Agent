# backend/analysis/plot_phase_c_multi_symbol_v3_vs_v4_1.py
#
# Plot Phase-C results: Single vs Multi, split by environment and seed
#
# Core question:
# What happens to already-validated Phase-C learning dynamics
# when the episode-level data distribution is broadened?
#
# NOTE:
# - Read-only analysis script
# - Consumes Phase-C JSON artifacts only
#

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACT_DIR_SINGLE = os.path.join(BASE_DIR, "artifacts", "phase_c")
ARTIFACT_DIR_MULTI  = os.path.join(BASE_DIR, "artifacts", "phase_c_multi")

PLOT_DIR = os.path.join(BASE_DIR, "artifacts", "plots", "phase_c_multi")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# Runs
# ============================================================

RUNS = [
    ("Single v3 (Seed 0)",   ARTIFACT_DIR_SINGLE, "v3/phase_c_v3_seed0.json"),
    ("Single v3 (Seed 1)",   ARTIFACT_DIR_SINGLE, "v3/phase_c_v3_seed1.json"),
    ("Single v4.1 (Seed 0)", ARTIFACT_DIR_SINGLE, "v4.1/phase_c_v4.1_seed0.json"),
    ("Single v4.1 (Seed 1)", ARTIFACT_DIR_SINGLE, "v4.1/phase_c_v4.1_seed1.json"),

    ("Multi v3 (Seed 0)",    ARTIFACT_DIR_MULTI,  "v3/phase_c_multi_v3_seed0.json"),
    ("Multi v3 (Seed 1)",    ARTIFACT_DIR_MULTI,  "v3/phase_c_multi_v3_seed1.json"),
    ("Multi v4.1 (Seed 0)",  ARTIFACT_DIR_MULTI,  "v4.1/phase_c_multi_v4.1_seed0.json"),
    ("Multi v4.1 (Seed 1)",  ARTIFACT_DIR_MULTI,  "v4.1/phase_c_multi_v4.1_seed1.json"),
]

# ============================================================
# Color palette (ENV × REGIME)
# ============================================================

ENV_COLORS = {
    "v3": {
        "Single": "#a1c9f4",   # light blue
        "Multi":  "#1f77b4",   # deep blue
    },
    "v4.1": {
        "Single": "#ffbe7d",   # light orange
        "Multi":  "#ff7f0e",   # deep orange
    },
}

# ============================================================
# Utilities
# ============================================================

def parse_label(label: str):
    regime = "Single" if label.startswith("Single") else "Multi"
    env    = "v4.1" if "v4.1" in label else "v3"
    seed   = "Seed 1" if "Seed 1" in label else "Seed 0"
    return regime, env, seed


def load_run(path):
    with open(path, "r") as f:
        data = json.load(f)

    eps = data["episodes"]
    out = {
        "episode": [e["episode"] for e in eps],
        "reward":  [e["total_reward"] for e in eps],
        "entropy": [e["entropy"] for e in eps],
    }

    if eps and "mean_drawdown_penalty" in eps[0]:
        out["mean_dd_penalty"] = [e["mean_drawdown_penalty"] for e in eps]

    return out


runs = []
for label, base, rel in RUNS:
    path = os.path.join(base, rel)
    if os.path.exists(path):
        print(f"Loading {label}")
        runs.append((label, load_run(path)))

# ============================================================
# Learning curves: reward & entropy
# ============================================================

for env in ["v3", "v4.1"]:
    for seed in ["Seed 0", "Seed 1"]:

        # -------------------------
        # Reward
        # -------------------------
        plt.figure(figsize=(10, 6))

        for label, d in runs:
            regime, e, s = parse_label(label)
            if e != env or s != seed:
                continue

            plt.plot(
                d["episode"],
                d["reward"],
                lw=2.5,
                color=ENV_COLORS[env][regime],
                label=f"{regime} {env} ({seed})",
            )

        plt.title(f"Phase-C — Episode Reward ({env}, {seed}, Single vs Multi)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(
            os.path.join(PLOT_DIR, f"phase_c_reward_{env}_{seed.replace(' ', '').lower()}_single_vs_multi.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # -------------------------
        # Entropy
        # -------------------------
        plt.figure(figsize=(10, 6))

        for label, d in runs:
            regime, e, s = parse_label(label)
            if e != env or s != seed:
                continue

            plt.plot(
                d["episode"],
                d["entropy"],
                lw=2.5,
                color=ENV_COLORS[env][regime],
                label=f"{regime} {env} ({seed})",
            )

        plt.title(f"Phase-C — Policy Entropy ({env}, {seed}, Single vs Multi)")
        plt.xlabel("Episode")
        plt.ylabel("Policy Entropy")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(
            os.path.join(PLOT_DIR, f"phase_c_entropy_{env}_{seed.replace(' ', '').lower()}_single_vs_multi.png"),
            dpi=300,
            bbox_inches="tight",
        )

# ============================================================
# Drawdown penalty (v4.1 only, seed-wise)
# ============================================================

for seed in ["Seed 0", "Seed 1"]:
    plt.figure(figsize=(10, 6))

    for label, d in runs:
        regime, env, s = parse_label(label)
        if env != "v4.1" or s != seed:
            continue
        if "mean_dd_penalty" not in d:
            continue

        plt.plot(
            d["episode"],
            d["mean_dd_penalty"],
            lw=2.5,
            color=ENV_COLORS["v4.1"][regime],
            label=f"{regime} v4.1 ({seed})",
        )

    plt.title(f"Phase-C v4.1 — Mean Drawdown Penalty ({seed}, Single vs Multi)")
    plt.xlabel("Episode")
    plt.ylabel("Mean Drawdown Penalty")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        os.path.join(PLOT_DIR, f"phase_c_drawdown_v4_1_{seed.replace(' ', '').lower()}_single_vs_multi.png"),
        dpi=300,
        bbox_inches="tight",
    )

plt.show()
