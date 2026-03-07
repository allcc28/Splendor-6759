"""
One-command evaluation pipeline:  eval → plots → summary

Runs evaluate_maskable_ppo.py, regenerates comparison plots (extract_tb_v3.py),
and prints a consolidated summary — so the full "eval → report" cycle is a
single reproducible command.

Usage:
    python project/scripts/run_eval_pipeline.py
    python project/scripts/run_eval_pipeline.py --model <path> --games 100
    python project/scripts/run_eval_pipeline.py --skip-plots

Pipeline steps:
    1. eval     — run evaluate_maskable_ppo.py, save JSON
    2. plots    — run extract_tb_v3.py (picks up the new JSON automatically)
    3. summary  — print consolidated table with V1 baseline comparison
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# V1 (score-based PPO) baseline for comparison
# ---------------------------------------------------------------------------
V1_BASELINE = {
    "Random (wrapper)": 51.0,
    "RandomAgent":      43.0,
    "GreedyAgent":      53.0,
}


def _run_step(step_name: str, cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Print step header, run subprocess, check for errors."""
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  Step: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(bar)
    result = subprocess.run(cmd, check=check)
    return result


def _find_latest_eval_json(eval_dir: str) -> Path | None:
    """Return the most recently created eval JSON in eval_dir."""
    pattern = Path(eval_dir) / "eval_v3_maskable_*.json"
    candidates = sorted(Path(eval_dir).glob("eval_v3_maskable_*.json"))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Run full eval → plots → summary pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="project/logs/maskable_ppo_score_v3_20260303_183435/final_model",
        help="Path to trained MaskablePPO model (without .zip)",
    )
    parser.add_argument("--games",       type=int,  default=100)
    parser.add_argument("--max-turns",   type=int,  default=200)
    parser.add_argument(
        "--eval-output",
        type=str,
        default="project/experiments/evaluation/maskable_ppo_v3_eval",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip extract_tb_v3.py regeneration (useful in WSL if matplotlib is slow)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  Splendor MaskablePPO — Full Evaluation Pipeline")
    print("=" * 80)
    print(f"  Model:    {args.model}")
    print(f"  Games:    {args.games} per opponent")
    print(f"  Eval dir: {args.eval_output}")
    ts_start = datetime.now()
    print(f"  Started:  {ts_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # -----------------------------------------------------------------------
    # Step 1: Evaluation
    # -----------------------------------------------------------------------
    eval_cmd = [
        sys.executable, "project/scripts/evaluate_maskable_ppo.py",
        "--model", args.model,
        "--games", str(args.games),
        "--max-turns", str(args.max_turns),
        "--output", args.eval_output,
    ]
    _run_step("Evaluate MaskablePPO agent", eval_cmd)

    # -----------------------------------------------------------------------
    # Step 2: Regenerate comparison plots
    # -----------------------------------------------------------------------
    if not args.skip_plots:
        plot_cmd = [sys.executable, "project/scripts/extract_tb_v3.py"]
        try:
            _run_step("Regenerate comparison plots", plot_cmd)
        except subprocess.CalledProcessError as exc:
            print(f"\n  ⚠️  Plot generation failed (exit code {exc.returncode})")
            print("     Training logs may not be available — continuing to summary.")
    else:
        print("\n  ── Skipping plot generation (--skip-plots) ──")

    # -----------------------------------------------------------------------
    # Step 3: Print consolidated summary
    # -----------------------------------------------------------------------
    latest_json = _find_latest_eval_json(args.eval_output)
    if latest_json is None:
        print(f"\n  ⚠️  No eval JSON found in {args.eval_output}  — cannot print summary.")
        return

    with open(latest_json) as f:
        results = json.load(f)

    key_map = {
        "Random (wrapper)": "vs_random_wrapper",
        "RandomAgent":      "vs_random_agent",
        "GreedyAgent":      "vs_greedy",
    }

    print(f"\n{'=' * 80}")
    print("  Pipeline Summary")
    print(f"{'=' * 80}")
    print(f"  Eval file: {latest_json.name}")
    print()
    print(f"  {'Opponent':<22}  {'V1 Win%':>8}  {'V3 Win%':>8}  {'Δ (pp)':>8}  {'Agent':>6}  {'Opp':>6}")
    print("  " + "─" * 66)

    for label, key in key_map.items():
        if key not in results:
            print(f"  {label:<22}  (not found in JSON)")
            continue
        r = results[key]
        v3_pct  = r["win_rate"]
        v1_pct  = V1_BASELINE.get(label, 0.0)
        delta   = v3_pct - v1_pct
        sign    = "+" if delta >= 0 else ""
        print(
            f"  {label:<22}  {v1_pct:>7.1f}%  {v3_pct:>7.1f}%"
            f"  {sign}{delta:>6.1f}  "
            f"  {r['agent_scores']['mean']:>5.1f}"
            f"  {r['opponent_scores']['mean']:>5.1f}"
        )

    ts_end = datetime.now()
    elapsed = (ts_end - ts_start).total_seconds()
    print(f"\n  Total elapsed: {elapsed:.0f}s")
    print(f"  Figures:       project/experiments/reports/v3_figures/")
    print(f"  Eval JSON:     {latest_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
