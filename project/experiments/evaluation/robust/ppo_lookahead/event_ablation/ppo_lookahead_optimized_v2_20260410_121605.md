# PPO+Lookahead Evaluation — optimized_v2

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.0% | [95.1%,98.2%] | 15.16 | 0.55 | 2.1 |
| Greedy | 91.6% | [88.8%,93.7%] | 15.09 | 6.69 | 6.39 |
