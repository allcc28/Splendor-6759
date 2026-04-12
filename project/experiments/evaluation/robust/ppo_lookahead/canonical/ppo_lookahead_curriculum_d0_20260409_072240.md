# PPO+Lookahead Evaluation — curriculum_d0

- depth=0, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.1% | [95.9%,98.0%] | 15.9 | 0.86 | 0.39 |
| Greedy | 73.7% | [70.9%,76.3%] | 14.38 | 9.3 | 4.65 |
