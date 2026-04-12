# PPO+Lookahead Evaluation — v1_d0_fixed

- depth=0, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 200

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.0% | [93.6%,98.6%] | 15.99 | 1.01 | 0.41 |
| Greedy | 76.0% | [69.6%,81.4%] | 14.49 | 8.91 | 4.74 |
