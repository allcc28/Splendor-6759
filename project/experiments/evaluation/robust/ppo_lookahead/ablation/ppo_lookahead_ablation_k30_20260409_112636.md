# PPO+Lookahead Evaluation — ablation_k30

- depth=1, top_k=30
- α=0.3, β=0.5, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.9% | [95.6%,97.8%] | 15.21 | 0.62 | 2.33 |
| Greedy | 93.0% | [91.2%,94.4%] | 15.24 | 6.64 | 8.48 |
