# PPO+Lookahead Evaluation — ablation_no_prob

- depth=1, top_k=15
- α=0.0, β=0.5, γ=0.5
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 96.5% | [95.2%,97.5%] | 14.86 | 0.56 | 2.19 |
| Greedy | 91.0% | [89.1%,92.6%] | 15.03 | 6.38 | 6.59 |
