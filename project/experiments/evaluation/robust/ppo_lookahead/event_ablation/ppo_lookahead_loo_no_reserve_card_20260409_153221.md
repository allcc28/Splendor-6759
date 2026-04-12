# PPO+Lookahead Evaluation — loo_no_reserve_card

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.0% | [93.9%,97.4%] | 14.93 | 0.63 | 2.73 |
| Greedy | 93.0% | [90.4%,94.9%] | 15.22 | 6.54 | 8.81 |
