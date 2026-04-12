# PPO+Lookahead Evaluation — ablation_no_value

- depth=1, top_k=15
- α=0.3, β=0.7, γ=0.0
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.5% | [96.3%,98.3%] | 15.35 | 0.63 | 2.07 |
| Greedy | 92.8% | [91.0%,94.2%] | 15.19 | 6.51 | 6.49 |
