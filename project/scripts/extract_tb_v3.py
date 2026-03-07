"""
Compare V1 (PPO), V2 (PPO+Greedy), and V3 (MaskablePPO) training curves.

Reads TensorBoard event files and eval/evaluations.npz files, then
saves side-by-side figures to project/experiments/reports/v3_figures/.

Usage:
    python project/scripts/extract_tb_v3.py
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE = Path("project/logs")
OUT  = Path("project/experiments/reports/v3_figures")
OUT.mkdir(parents=True, exist_ok=True)

V1_EVAL = BASE / "ppo_score_based_v1_20260224_113524" / "eval" / "evaluations.npz"
V1_TB   = BASE / "ppo_score_based_v1_20260224_113524" / "logs" / "tensorboard"

V3_EVAL = BASE / "maskable_ppo_score_v3_20260303_183435" / "eval" / "evaluations.npz"
V3_TB   = BASE / "maskable_ppo_score_v3_20260303_183435" / "logs" / "tensorboard"


def read_eval_npz(path):
    """Return (timesteps, mean_rewards, std_rewards) from evaluations.npz."""
    if not Path(path).exists():
        return None, None, None
    d = np.load(path)
    ts   = d["timesteps"]
    res  = d["results"]
    return ts, res.mean(axis=1), res.std(axis=1)


def read_tb_scalar(tb_log_dir, tag):
    """Read a scalar tag from all .tfevents files under tb_log_dir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not importable for parsing — skipping TB scalars")
        return [], []

    steps, values = [], []
    for ef in sorted(Path(tb_log_dir).rglob("events.out.tfevents.*")):
        ea = EventAccumulator(str(ef))
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                steps.append(e.step)
                values.append(e.value)
    if steps:
        order = np.argsort(steps)
        steps  = np.array(steps)[order]
        values = np.array(values)[order]
    return steps, values


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

v1_ts, v1_mean, v1_std = read_eval_npz(V1_EVAL)
v3_ts, v3_mean, v3_std = read_eval_npz(V3_EVAL)

print(f"V1 eval points: {len(v1_ts) if v1_ts is not None else 'missing'}")
print(f"V3 eval points: {len(v3_ts) if v3_ts is not None else 'missing'}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Eval reward curve comparison
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

COLORS = {"V1 PPO\n(random opp)": ("#1f77b4", v1_ts, v1_mean, v1_std),
          "V3 MaskablePPO\n(random opp)": ("#2ca02c", v3_ts, v3_mean, v3_std)}

for label, (color, ts, mean, std) in COLORS.items():
    if ts is None:
        continue
    ax.plot(ts / 1e6, mean, label=label, color=color, linewidth=2)
    ax.fill_between(ts / 1e6, mean - std, mean + std, alpha=0.15, color=color)

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("Training Timesteps (M)", fontsize=12)
ax.set_ylabel("Mean Eval Reward", fontsize=12)
ax.set_title("V1 (PPO) vs V3 (MaskablePPO) — Eval Reward over Training", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

out = OUT / "v1_vs_v3_eval_reward.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: V3-only eval reward with confidence band
# ──────────────────────────────────────────────────────────────────────────────

if v3_ts is not None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(v3_ts / 1e6, v3_mean, color="#2ca02c", linewidth=2, label="V3 MaskablePPO")
    ax.fill_between(v3_ts / 1e6, v3_mean - v3_std, v3_mean + v3_std,
                    alpha=0.2, color="#2ca02c")
    ax.axhline(v3_mean.max(), color="red", linestyle=":", linewidth=1,
               label=f"Peak: {v3_mean.max():.1f} @ {v3_ts[v3_mean.argmax()]/1e6:.2f}M")
    ax.set_xlabel("Training Timesteps (M)", fontsize=12)
    ax.set_ylabel("Mean Eval Reward", fontsize=12)
    ax.set_title("V3 MaskablePPO — Evaluation Reward Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    out = OUT / "v3_eval_reward.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Bar chart — Win rate comparison (filled after eval)
# ──────────────────────────────────────────────────────────────────────────────
# This will be updated once evaluate_maskable_ppo.py results are in.
# For now we show V1 results as placeholder.

import glob, json

v1_evals = sorted(glob.glob("project/experiments/evaluation/ppo_score_based_eval_v3/*.json"))
v3_evals = sorted(glob.glob("project/experiments/evaluation/maskable_ppo_v3_eval/*.json"))

if v1_evals and v3_evals:
    v1_data = json.load(open(v1_evals[-1]))
    v3_data = json.load(open(v3_evals[-1]))

    opponents = ["vs_random_wrapper", "vs_random_agent", "vs_greedy"]
    labels    = ["Random (wrapper)", "RandomAgent", "GreedyAgent"]

    v1_wr = [v1_data.get(k, {}).get("win_rate", 0) for k in opponents]
    v3_wr = [v3_data.get(k, {}).get("win_rate", 0) for k in opponents]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, v1_wr, w, label="V1 PPO",          color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, v3_wr, w, label="V3 MaskablePPO",  color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="50% baseline")
    ax.set_title("V1 (PPO) vs V3 (MaskablePPO) — Win Rate by Opponent", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for rect, val in zip(ax.patches, v1_wr + v3_wr):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.8,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    out = OUT / "v1_vs_v3_win_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
else:
    print("Win rate bar chart skipped (eval JSON not yet available — re-run after evaluation)")

print("\nDone.")
