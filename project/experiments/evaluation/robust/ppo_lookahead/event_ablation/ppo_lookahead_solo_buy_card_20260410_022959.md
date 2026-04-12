# PPO+Lookahead Evaluation — solo_buy_card

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.6% | [94.6%,97.9%] | 15.48 | 0.92 | 2.96 |
| Greedy | 77.8% | [74.0%,81.2%] | 14.44 | 8.83 | 8.98 |
