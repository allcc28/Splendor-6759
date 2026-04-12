# PPO+Lookahead Evaluation — fix_test

- depth=0, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 20

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 100.0% | [83.9%,100.0%] | 16.25 | 1.6 | 0.39 |
| Greedy | 5.0% | [0.9%,23.6%] | 4.3 | 15.45 | 0.43 |
