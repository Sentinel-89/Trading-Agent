# backend/analysis/plot_phase_c_v3_vs_v4_1.py
#
# Plot Phase-C results: v3 vs v4.1
#
# Purpose:
# - Compare learning dynamics under baseline and risk-aware environments
# - Visualize reward, entropy, action frequencies, and drawdown metrics
#
# NOTE:
# - Read-only analysis script
# - Consumes Phase-C JSON artifacts only
#

import os
import json
import matplotlib.pyplot as plt

# ============================================================
# Resolve paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts", "phase_c")
PLOT_DIR = os.path.join(BASE_DIR, "artifacts", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Updated to match your exact filenames in the explorer
RUNS = [
    {
        "label": "v3 (Seed 0)", 
        "filename": os.path.join("v3", "phase_c_v3_seed0.json")
    },
    {
        "label": "v3 (Seed 1)", 
        "filename": os.path.join("v3", "phase_c_v3_seed1.json")
    },
    {
        "label": "v4.1 (Seed 0)", 
        "filename": os.path.join("v4.1", "phase_c_v4.1_seed0.json")
    },
    {
        "label": "v4.1 (Seed 1)", 
        "filename": os.path.join("v4.1", "phase_c_v4.1_seed1.json")
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
    }

    if len(episodes) > 0 and "mean_drawdown_penalty" in episodes[0]:
        out["mean_dd_penalty"] = [e["mean_drawdown_penalty"] for e in episodes]

    return out

# ============================================================
# Load available runs
# ============================================================

runs_data = []

for run in RUNS:
    path = os.path.join(ARTIFACT_DIR, run["filename"])
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

plt.title("Phase-C — Episode Reward (v3 vs v4.1)")
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

plt.title("Phase-C — Entropy Evolution")
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

    plt.title(f"Phase-C — Action Distribution ({run['label']})")
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

plt.title("Phase-C v4.1 — Drawdown Penalty per Episode")
plt.xlabel("Episode"); plt.ylabel("Mean Penalty"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "phase_c_4_drawdown_penalty.png"), dpi=300, bbox_inches='tight')

plt.show()