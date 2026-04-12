# PPO+Lookahead Evaluation — loo_no_engine_spike

- depth=1, top_k=15
- α=0.3, β=0.5, γ=0.2
- Games: 500

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 97.8% | [96.1%,98.8%] | 15.33 | 0.62 | 2.72 |
| Greedy | 90.8% | [87.9%,93.0%] | 15.01 | 6.77 | 8.77 |
