# PPO+Lookahead Evaluation — ablation_heavy_event

- depth=1, top_k=15
- α=0.1, β=0.7, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 98.2% | [97.2%,98.9%] | 15.44 | 0.51 | 2.49 |
| Greedy | 92.5% | [90.7%,94.0%] | 15.09 | 6.54 | 6.39 |
