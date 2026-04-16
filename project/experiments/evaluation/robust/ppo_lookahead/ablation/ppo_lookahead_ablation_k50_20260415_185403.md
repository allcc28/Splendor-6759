# PPO+Lookahead Evaluation — ablation_k50

- depth=1, top_k=50
- α=0.3, β=0.5, γ=0.2
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 99.4% | [98.7%,99.7%] | 15.69 | 0.61 | 3.46 |
| Greedy | 87.3% | [85.1%,89.2%] | 15.1 | 9.0 | 10.62 |
