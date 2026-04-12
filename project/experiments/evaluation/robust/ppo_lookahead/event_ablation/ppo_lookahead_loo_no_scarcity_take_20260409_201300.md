# PPO+Lookahead Evaluation — loo_no_scarcity_take

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.2% | [95.4%,98.3%] | 15.38 | 0.58 | 2.63 |
| Greedy | 90.6% | [87.7%,92.9%] | 14.8 | 6.38 | 8.53 |
