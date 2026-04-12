# PPO+Lookahead Evaluation — optimized_v1

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.6% | [94.6%,97.9%] | 14.94 | 0.64 | 2.7 |
| Greedy | 89.8% | [86.8%,92.2%] | 14.86 | 7.18 | 8.37 |
