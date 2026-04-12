# PPO+Lookahead Evaluation — depth1_fix

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 50

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.0% | [86.5%,98.9%] | 15.04 | 0.44 | 2.16 |
| Greedy | 18.0% | [9.8%,30.8%] | 8.0 | 14.24 | 2.23 |
