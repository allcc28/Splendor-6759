# Session Log: 2026-02-26 — The Masking Pivot

## Overview
Concluding Experiment 1 (PPO vs. Greedy Opponent) and making the architectural decision to switch to Action Masking for all future training.

## Experiment 1 (v2) Post-Mortem
- **Config**: PPO vs `GreedyAgentBoost`.
- **Primary Finding**: The agent developed a "Passive Avoidance" strategy. 
  - To minimize the -10 reward from illegal actions (which the Greedy opponent forces by taking resources), the agent learned to make safe but useless moves (e.g., reserving cards it can never buy).
  - Average game length increased to ~90 steps, but average score remained near 0.
- **Metric Failure**: Explained variance dropped to **-0.167**, indicating the value network was completely unable to predict the outcome of states.
- **Comparison**: While v1 (Random opponent) achieved 50% win rates, v2 achieved < 5% vs Greedy.

## The Technical Pivot: MaskablePPO
We have identified that the 200-dim discrete action space is the primary bottleneck. Even with 1M timesteps, the agent spends too much entropy "finding" the legal moves rather than optimizing strategy.

**Decision**: 
1. Install `sb3-contrib`.
2. Update `SplendorGymWrapper` to provide an action mask.
3. Transition to `MaskablePPO`.
4. This will "physically" prevent the agent from sampling illegal moves, allowing the neural network to focus entirely on which *legal* move is best.

## Next Steps
1. Environment update: `pip install sb3-contrib`.
2. Wrapper update: Implement `action_masks()` method.
3. Initiate Phase 2 (Event-based Reward Shaping) using the new Maskable architecture.

## Artifacts Created
- `project/experiments/reports/ppo_v2_greedy_opponent_report.md`
- `project/experiments/reports/v2_figures/v1_vs_v2_training_comparison.png`
- `project/scripts/extract_tb_v2.py` (Comparative plotting tool)
