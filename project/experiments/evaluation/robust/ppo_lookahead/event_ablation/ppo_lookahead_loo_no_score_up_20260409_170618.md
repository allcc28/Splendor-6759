# PPO+Lookahead Evaluation — loo_no_score_up

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.0% | [95.1%,98.2%] | 15.1 | 0.71 | 2.69 |
| Greedy | 92.4% | [89.7%,94.4%] | 15.12 | 6.84 | 8.58 |
