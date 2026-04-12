# PPO+Lookahead Evaluation — solo_reach_15

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.6% | [95.9%,98.6%] | 15.3 | 0.81 | 2.55 |
| Greedy | 81.2% | [77.5%,84.4%] | 14.16 | 8.12 | 8.4 |
