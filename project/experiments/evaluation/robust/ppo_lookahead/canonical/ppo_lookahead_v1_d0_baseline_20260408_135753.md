# PPO+Lookahead Evaluation — v1_d0_baseline

- depth=0, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.2% | [95.4%,98.3%] | 16.01 | 0.86 | 0.35 |
| Greedy | 96.0% | [93.9%,97.4%] | 15.85 | 0.9 | 5.56 |
