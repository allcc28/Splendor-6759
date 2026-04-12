# PPO+Lookahead Evaluation — ablation_event_only

- depth=1, top_k=15
- α=0.0, β=1.0, γ=0.0
- Games: 1000

| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |
|----------|----------|--------|---------|-------------|----------|
| Random | 95.8% | [94.4%,96.9%] | 15.05 | 0.54 | 2.12 |
| Greedy | 91.4% | [89.5%,93.0%] | 14.97 | 6.35 | 6.42 |
