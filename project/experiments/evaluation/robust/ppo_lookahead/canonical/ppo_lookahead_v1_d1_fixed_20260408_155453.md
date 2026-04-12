# PPO+Lookahead Evaluation — v1_d1_fixed

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 200

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.5% | [94.3%,98.9%] | 15.07 | 0.56 | 1.99 |
| Greedy | 88.0% | [82.8%,91.8%] | 14.68 | 6.63 | 7.39 |
