import json
path = "project/experiments/evaluation/robust/robust_eval_v5_event_20260309_211510.json"
d = json.load(open(path))
print("Top-level keys:", list(d.keys()))

# Find results dict — may be top-level or nested
results = d.get("results") or d
if not isinstance(results, dict):
    print(d); exit()

print(f"V5 Event-Based Robust Evaluation")
print("="*65)
for opp, r in results.items():
    if not isinstance(r, dict) or "win_rate_pooled" not in r: continue
    print(f"vs {opp}:")
    print(f"  Pooled win rate : {r.get('win_rate_pooled', r.get('win_rate', '?')):.1f}%")
    print(f"  Wilson 95% CI   : [{r.get('wilson_ci_95_lo', 0):.1f}%, {r.get('wilson_ci_95_hi', 0):.1f}%]")
    print(f"  W / L / D       : {r.get('total_wins', r.get('wins','?'))} / {r.get('total_losses', r.get('losses','?'))} / {r.get('total_draws', r.get('draws','?'))}")
    print(f"  Agent avg score : {r.get('agent_score_mean', 0):.1f}")
    print(f"  Opp avg score   : {r.get('opp_score_mean', 0):.1f}")
    print()
