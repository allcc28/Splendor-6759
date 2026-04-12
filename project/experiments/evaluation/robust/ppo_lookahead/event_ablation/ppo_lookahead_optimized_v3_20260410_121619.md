# PPO+Lookahead Evaluation — optimized_v3

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.8% | [94.9%,98.0%] | 15.28 | 0.62 | 2.11 |
| Greedy | 90.8% | [87.9%,93.0%] | 15.05 | 6.82 | 6.41 |
