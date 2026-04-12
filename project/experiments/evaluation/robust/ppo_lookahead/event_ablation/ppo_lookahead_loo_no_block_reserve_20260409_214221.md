# PPO+Lookahead Evaluation — loo_no_block_reserve

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.6% | [95.9%,98.6%] | 15.33 | 0.62 | 2.47 |
| Greedy | 91.2% | [88.4%,93.4%] | 15.31 | 6.92 | 8.24 |
