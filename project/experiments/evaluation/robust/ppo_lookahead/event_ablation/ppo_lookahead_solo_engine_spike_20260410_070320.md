# PPO+Lookahead Evaluation — solo_engine_spike

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.0% | [95.1%,98.2%] | 15.43 | 0.75 | 2.53 |
| Greedy | 85.8% | [82.5%,88.6%] | 14.86 | 7.55 | 8.31 |
