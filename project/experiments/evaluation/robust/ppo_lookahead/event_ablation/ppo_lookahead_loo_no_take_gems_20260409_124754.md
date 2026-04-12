# PPO+Lookahead Evaluation — loo_no_take_gems

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 98.0% | [96.4%,98.9%] | 15.28 | 0.49 | 2.05 |
| Greedy | 92.6% | [90.0%,94.6%] | 15.14 | 6.6 | 5.8 |
