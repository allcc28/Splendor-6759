# PPO+Lookahead Evaluation — solo_score_up

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.6% | [94.6%,97.9%] | 15.38 | 0.82 | 2.43 |
| Greedy | 86.8% | [83.5%,89.5%] | 15.24 | 7.72 | 8.56 |
