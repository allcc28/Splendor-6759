# PPO+Lookahead Evaluation — smoke

- depth=1, top_k=10
- α=0.3, β=0.5, γ=0.2
- Games: 10

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 90.0% | [59.6%,98.2%] | 14.0 | 0.8 | 2.31 |
| Greedy | 40.0% | [16.8%,68.7%] | 10.5 | 14.3 | 2.35 |
