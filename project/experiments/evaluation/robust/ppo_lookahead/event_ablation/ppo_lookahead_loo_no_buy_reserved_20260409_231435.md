# PPO+Lookahead Evaluation — loo_no_buy_reserved

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 95.8% | [93.7%,97.2%] | 15.02 | 0.65 | 2.57 |
| Greedy | 90.4% | [87.5%,92.7%] | 15.06 | 6.88 | 8.49 |
