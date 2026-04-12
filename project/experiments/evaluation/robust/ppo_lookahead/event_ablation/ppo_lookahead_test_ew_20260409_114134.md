# PPO+Lookahead Evaluation — test_ew

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 5

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 100.0% | [56.5%,100.0%] | 15.6 | 1.4 | 2.61 |
| Greedy | 80.0% | [37.5%,96.4%] | 15.4 | 7.4 | 7.46 |
