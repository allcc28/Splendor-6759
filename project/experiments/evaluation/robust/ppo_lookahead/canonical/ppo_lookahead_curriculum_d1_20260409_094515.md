# PPO+Lookahead Evaluation — curriculum_d1

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.7% | [95.4%,97.6%] | 14.99 | 0.55 | 2.01 |
| Greedy | 91.5% | [89.6%,93.1%] | 15.14 | 6.77 | 6.54 |
