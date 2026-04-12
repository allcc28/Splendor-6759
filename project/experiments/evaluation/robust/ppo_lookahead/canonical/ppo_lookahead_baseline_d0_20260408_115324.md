# PPO+Lookahead Evaluation — baseline_d0

- depth=0, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.3% | [94.9%,97.3%] | 15.75 | 0.93 | 0.41 |
| Greedy | 1.8% | [1.1%,2.8%] | 4.54 | 15.87 | 0.32 |
