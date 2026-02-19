import os, json
import matplotlib.pyplot as plt

EVAL_DIR = "backend/artifacts/phase_d_continuous/seed_1/eval"  # folder with .json files
SCENARIO = "full_daily"  # choose: full_daily/full_weekly/topk_daily/topk_weekly

# Load all jsons
paths = sorted(
    os.path.join(EVAL_DIR, fn)
    for fn in os.listdir(EVAL_DIR)
    if fn.endswith(".json")
)

# Pick a few (or all)
paths = paths  # or paths[-10:] for last 10 checkpoints

plt.figure()

for p in paths:
    with open(p, "r") as f:
        r = json.load(f)
    curve = r["scenarios"][SCENARIO]["equity_curve"]
    label = r["checkpoint"].replace(".pt", "")
    plt.plot(curve, label=label)

plt.title(f"Scenario: {SCENARIO}")
plt.xlabel("Step")
plt.ylabel("Equity")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()