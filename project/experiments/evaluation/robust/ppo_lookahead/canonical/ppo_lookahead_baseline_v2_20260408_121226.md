# PPO+Lookahead Evaluation — baseline_v2

- depth=0, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.2% | [94.8%,97.2%] | 15.71 | 0.88 | 0.41 |
| Greedy | 2.3% | [1.5%,3.4%] | 4.45 | 15.79 | 0.32 |
