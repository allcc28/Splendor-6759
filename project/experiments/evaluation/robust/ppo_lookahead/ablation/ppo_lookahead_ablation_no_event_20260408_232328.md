# PPO+Lookahead Evaluation — ablation_no_event

- depth=1, top_k=15
- α=1.0, β=0.0, γ=0.0
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.1% | [94.7%,97.1%] | 15.45 | 0.82 | 2.79 |
| Greedy | 76.1% | [73.4%,78.6%] | 14.33 | 8.95 | 7.17 |
