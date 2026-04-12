# PPO+Lookahead Evaluation — loo_no_reach_15

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.0% | [95.1%,98.2%] | 15.39 | 0.66 | 2.68 |
| Greedy | 89.4% | [86.4%,91.8%] | 15.05 | 6.99 | 8.54 |
