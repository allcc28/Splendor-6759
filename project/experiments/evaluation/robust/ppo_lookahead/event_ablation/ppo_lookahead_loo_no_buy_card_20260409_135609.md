# PPO+Lookahead Evaluation — loo_no_buy_card

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 98.2% | [96.6%,99.1%] | 15.2 | 0.72 | 1.9 |
| Greedy | 86.4% | [83.1%,89.1%] | 14.57 | 7.22 | 6.28 |
