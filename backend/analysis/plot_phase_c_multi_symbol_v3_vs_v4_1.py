# backend/analysis/plot_phase_c_multi_symbol_v3_vs_v4_1.py
#
# Plot Phase-C results (Multi-Symbol): v3 vs v4.1
#
# Purpose:
# - Compare learning dynamics under baseline (v3) and risk-aware (v4.1) environments
# - Report pooled learning curves under multi-symbol episode sampling
# - Provide additive symbol-wise diagnostics for generalization analysis
#
# NOTE:
# - Read-only analysis script
# - Consumes Phase-C JSON artifacts only
#

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Resolve paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------
# Phase-C artifacts
# ------------------------------------------------------------

# Single-symbol Phase-C (baseline runs)
ARTIFACT_DIR_SINGLE = os.path.join(BASE_DIR, "artifacts", "phase_c")

# Multi-symbol Phase-C (Full Phase-C runs)
ARTIFACT_DIR_MULTI = os.path.join(BASE_DIR, "artifacts", "phase_c_multi")

# ------------------------------------------------------------
# Plot output directory (multi-symbol analysis)
# ------------------------------------------------------------

PLOT_DIR = os.path.join(BASE_DIR, "artifacts", "plots", "phase_c_multi")
os.makedirs(PLOT_DIR, exist_ok=True)

# Updated to match new folder structure, separating multi-symbol- from complementary single-symbol runs
RUNS = [
    # =========================================================
    # Single-symbol Phase-C (baseline)
    # =========================================================
    {
        "label": "Single v3 (Seed 0)",
        "base_dir": ARTIFACT_DIR_SINGLE,
        "filename": os.path.join("v3", "phase_c_v3_seed0.json"),
    },
    {
        "label": "Single v3 (Seed 1)",
        "base_dir": ARTIFACT_DIR_SINGLE,
        "filename": os.path.join("v3", "phase_c_v3_seed1.json"),
    },
    {
        "label": "Single v4.1 (Seed 0)",
        "base_dir": ARTIFACT_DIR_SINGLE,
        "filename": os.path.join("v4.1", "phase_c_v4.1_seed0.json"),
    },
    {
        "label": "Single v4.1 (Seed 1)",
        "base_dir": ARTIFACT_DIR_SINGLE,
        "filename": os.path.join("v4.1", "phase_c_v4.1_seed1.json"),
    },

    # =========================================================
    # Multi-symbol Phase-C (Full Phase-C)
    # =========================================================
    {
        "label": "Multi v3 (Seed 0)",
        "base_dir": ARTIFACT_DIR_MULTI,
        "filename": os.path.join("v3", "phase_c_multi_v3_seed0.json"),
    },
    {
        "label": "Multi v3 (Seed 1)",
        "base_dir": ARTIFACT_DIR_MULTI,
        "filename": os.path.join("v3", "phase_c_multi_v3_seed1.json"),
    },
    {
        "label": "Multi v4.1 (Seed 0)",
        "base_dir": ARTIFACT_DIR_MULTI,
        "filename": os.path.join("v4.1", "phase_c_multi_v4.1_seed0.json"),
    },
    {
        "label": "Multi v4.1 (Seed 1)",
        "base_dir": ARTIFACT_DIR_MULTI,
        "filename": os.path.join("v4.1", "phase_c_multi_v4.1_seed1.json"),
    },
]

# ============================================================
# Helper: load run 
# ============================================================

def load_run(path):
    with open(path, "r") as f:
        data = json.load(f)

    episodes = data["episodes"]

    out = {
        "episode": [e["episode"] for e in episodes],
        "reward": [e["total_reward"] for e in episodes],
        "entropy": [e["entropy"] for e in episodes],
        "buy": [e["action_freq"]["BUY"] for e in episodes],
        "sell": [e["action_freq"]["SELL"] for e in episodes],
        "hold": [e["action_freq"]["HOLD"] for e in episodes],
        "symbol": [e.get("symbol", "SINGLE") for e in episodes], # backward compatibility with single-symbol Phase-C logs
    }

    if len(episodes) > 0 and "mean_drawdown_penalty" in episodes[0]:
        out["mean_dd_penalty"] = [e["mean_drawdown_penalty"] for e in episodes]

    return out

# ============================================================
# Load available runs
# ============================================================

runs_data = []

for run in RUNS:
    path = os.path.join(run["base_dir"], run["filename"])
    if os.path.exists(path):
        print(f"Loading {run['label']}...")
        runs_data.append({"label": run["label"], "data": load_run(path)})
    else:
        print(f"Skipping {run['label']} (file not found: {path})")

if not runs_data:
    print("No data found. Ensure training is complete before running analysis.")
    exit()

# ============================================================
# Define the granular color map (Version + Seed)
# ============================================================
# Deep colors for Seed 0, Lighter tints for Seed 1
color_map = {
    "v3 (Seed 0)": "#1f77b4",   # Deep Blue
    "v3 (Seed 1)": "#a1c9f4",   # Sky Blue
    "v4.1 (Seed 0)": "#ff7f0e", # Deep Orange
    "v4.1 (Seed 1)": "#ffbe7d", # Light Orange
}

# ============================================================
# Plot 1: Episode reward (Version Comparison)
# ============================================================
plt.figure(figsize=(10, 6))
for run in runs_data:
    label = run["label"]
    # Fallback to a default if label doesn't match exactly
    color = color_map.get(label, "#333333") 
    
    plt.plot(
        run["data"]["episode"], 
        run["data"]["reward"], 
        label=label, 
        color=color, 
        lw=2, 
        alpha=0.9
    )

plt.title("Phase-C Multi-Symbol — Episode Reward (v3 vs v4.1)")
plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "phase_c_1_reward.png"), dpi=300, bbox_inches='tight')

# ============================================================
# Plot 2: Policy entropy (Version Comparison)
# ============================================================
plt.figure(figsize=(10, 6))
for run in runs_data:
    label = run["label"]
    color = color_map.get(label, "#333333")
    
    plt.plot(
        run["data"]["episode"], 
        run["data"]["entropy"], 
        label=label, 
        color=color, 
        lw=2, 
        alpha=0.9
    )

plt.title("Phase-C Multi-Symbol — Entropy Evolution")
plt.xlabel("Episode"); plt.ylabel("Policy Entropy"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "phase_c_2_entropy.png"), dpi=300, bbox_inches='tight')

# ============================================================
# Plot 3: Action frequencies (Action Type Comparison)
# ============================================================
# Standardized financial colors for BUY/SELL/HOLD
for run in runs_data:
    d = run["data"]
    plt.figure(figsize=(10, 6))
    plt.plot(d["episode"], d["buy"],  label="BUY",  color='#2ca02c', lw=2) 
    plt.plot(d["episode"], d["sell"], label="SELL", color='#d62728', lw=2)
    plt.plot(d["episode"], d["hold"], label="HOLD", color='#7f7f7f', lw=1.5, linestyle='--')

    plt.title(f"Phase-C Multi-Symbol — Action Distribution ({run['label']})")
    plt.xlabel("Episode"); plt.ylabel("Action Frequency"); plt.legend(); plt.grid(True, alpha=0.2)
    
    safe_label = run["label"].lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(PLOT_DIR, f"phase_c_3_actions_{safe_label}.png"), dpi=300, bbox_inches='tight')

# ============================================================
# Plot 4: Drawdown penalty (v4.1 only)
# ============================================================
plt.figure(figsize=(10, 6))
# Define shades for drawdown specifically
dd_colors = {"v4.1 (Seed 0)": "#9467bd", "v4.1 (Seed 1)": "#c5b0d5"}

for run in runs_data:
    label = run["label"]
    d = run["data"]
    if "mean_dd_penalty" in d:
        color = dd_colors.get(label, "#9467bd")
        plt.plot(d["episode"], d["mean_dd_penalty"], label=label, color=color, lw=2)

plt.title("Phase-C Multi-Symbol v4.1 — Drawdown Penalty per Episode")
plt.xlabel("Episode"); plt.ylabel("Mean Penalty"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "phase_c_4_drawdown_penalty.png"), dpi=300, bbox_inches='tight')

# ============================================================
# Plot 5: Reward distribution by symbol (Multi-symbol only)
# ============================================================

from collections import defaultdict

symbol_rewards = defaultdict(list)

for run in runs_data:
    d = run["data"]
    for r, s in zip(d["reward"], d["symbol"]):
        symbol_rewards[s].append(r)

# Remove SINGLE if present
symbol_rewards.pop("SINGLE", None)

if symbol_rewards:
    plt.figure(figsize=(12, 6))
    symbols = sorted(symbol_rewards.keys())
    means = [np.mean(symbol_rewards[s]) for s in symbols]
    stds = [np.std(symbol_rewards[s]) for s in symbols]

    plt.bar(symbols, means, yerr=stds, capsize=4, alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.title("Phase-C Multi-Symbol — Reward by Symbol")
    plt.ylabel("Episode Reward (mean ± std)")
    plt.grid(True, axis="y", alpha=0.3)

    plt.savefig(
        os.path.join(PLOT_DIR, "phase_c_5_reward_by_symbol.png"),
        dpi=300,
        bbox_inches="tight",
    )

# ============================================================
# Plot 6: Mean entropy by symbol (Multi-symbol only)
# ============================================================

symbol_entropy = defaultdict(list)

for run in runs_data:
    d = run["data"]
    for e, s in zip(d["entropy"], d["symbol"]):
        symbol_entropy[s].append(e)

symbol_entropy.pop("SINGLE", None)

if symbol_entropy:
    plt.figure(figsize=(12, 6))
    symbols = sorted(symbol_entropy.keys())
    means = [np.mean(symbol_entropy[s]) for s in symbols]

    plt.bar(symbols, means, alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.title("Phase-C Multi-Symbol — Mean Policy Entropy by Symbol")
    plt.ylabel("Mean Entropy")
    plt.grid(True, axis="y", alpha=0.3)

    plt.savefig(
        os.path.join(PLOT_DIR, "phase_c_6_entropy_by_symbol.png"),
        dpi=300,
        bbox_inches="tight",
    )

plt.show()