# PPO+Lookahead Evaluation — ablation_k5

- depth=1, top_k=5
- α=0.3, β=0.5, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.1% | [95.9%,98.0%] | 15.14 | 0.64 | 1.13 |
| Greedy | 87.7% | [85.5%,89.6%] | 14.92 | 7.21 | 5.55 |
