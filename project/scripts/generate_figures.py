"""Generate all presentation figures for Splendor RL project."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "project/docs/figures"
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'figure.dpi': 150,
})

COLORS = {
    'blue': '#007bff',
    'red': '#dc3545',
    'green': '#28a745',
    'gold': '#ffc107',
    'purple': '#6f42c1',
    'gray': '#6c757d',
    'dark': '#343a40',
    'orange': '#fd7e14',
}

# ============================================================
# 1. PPO Progression Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
agents = ['PPO\nSparse', 'PPO\nScore (V4a)', 'PPO\nEvent (V5)', 'AlphaZero\n(best)', 'PPO+\nLookahead']
win_rates = [3, 75.8, 78.0, 24, 91.9]
colors = [COLORS['gray'], COLORS['blue'], COLORS['green'], COLORS['red'], COLORS['gold']]

bars = ax.bar(agents, win_rates, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Win Rate vs Greedy (%)', fontsize=14)
ax.set_title('Agent Progression: From Reward Engineering to Planning', fontsize=15, fontweight='bold')
ax.set_ylim(0, 105)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')

for bar, val in zip(bars, win_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{val}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add arrows showing progression
ax.annotate('', xy=(3.15, 85), xytext=(2.15, 80),
            arrowprops=dict(arrowstyle='->', color=COLORS['gold'], lw=2.5))
ax.text(2.65, 83, '+14pp', fontsize=10, ha='center', color=COLORS['gold'], fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/01_ppo_progression.png', bbox_inches='tight')
plt.close()
print("1. PPO progression done")

# ============================================================
# 2. AlphaZero Failure Table (as figure)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis('off')

data = [
    ['AlphaZero V2 (Pure, 20 iter)', '25%', '0%', 'MCTS too shallow'],
    ['AlphaZero V3 (Shaped, 30 iter)', '34%', '4%', 'Reward shaping not enough'],
    ['AlphaZero V4 (Warm-start, 60 iter)', '6%', '2%', 'PPO distillation hurt'],
    ['AlphaZero V3-long (60 iter)', '10%', '0%', 'More training did not help'],
]
cols = ['Variant', 'vs Random', 'vs Greedy', 'Diagnosis']

table = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Header styling
for j in range(len(cols)):
    table[0, j].set_facecolor(COLORS['red'])
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight low win rates
for i in range(1, len(data)+1):
    table[i, 2].set_text_props(color='red', fontweight='bold')

ax.set_title('AlphaZero: All Variants Failed (<10% vs Greedy)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT}/02_alphazero_failure.png', bbox_inches='tight')
plt.close()
print("2. AlphaZero failure done")

# ============================================================
# 3. PPO+Lookahead Architecture Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)

# Boxes
boxes = [
    (0.5, 1.5, 2, 1.5, 'Legal Actions\n(30-80)', COLORS['gray']),
    (3.2, 1.5, 2, 1.5, 'PPO Policy\nTop-K=15', COLORS['blue']),
    (6.0, 1.5, 2.2, 1.5, 'Forward\nSimulation', COLORS['purple']),
    (8.8, 1.5, 2.2, 1.5, 'Event\nEvaluation', COLORS['green']),
    (8.8, 0, 2.2, 1, 'Best Action', COLORS['gold']),
]
for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.85, edgecolor='white', linewidth=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=3)

# Arrows
arrow_style = dict(arrowstyle='->', color=COLORS['dark'], lw=2.5)
ax.annotate('', xy=(3.2, 2.25), xytext=(2.5, 2.25), arrowprops=arrow_style)
ax.annotate('', xy=(6.0, 2.25), xytext=(5.2, 2.25), arrowprops=arrow_style)
ax.annotate('', xy=(8.8, 2.25), xytext=(8.2, 2.25), arrowprops=arrow_style)
ax.annotate('', xy=(9.9, 1.5), xytext=(9.9, 1.0), arrowprops=arrow_style)

# Labels above arrows
ax.text(2.85, 2.8, 'filter', fontsize=9, ha='center', color=COLORS['dark'])
ax.text(5.6, 2.8, 'clone & step', fontsize=9, ha='center', color=COLORS['dark'])
ax.text(8.5, 2.8, 'score events', fontsize=9, ha='center', color=COLORS['dark'])

# Score formula
ax.text(6, 0.4, 'Score = 0.3 x PPO_prob + 0.5 x event_reward + 0.2 x future_value',
        fontsize=10, ha='center', style='italic', color=COLORS['dark'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))

ax.set_title('PPO+Lookahead Architecture', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/03_lookahead_architecture.png', bbox_inches='tight')
plt.close()
print("3. Architecture diagram done")

# ============================================================
# 4. Component Ablation Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
configs = ['Full\n(baseline)', 'No event\nreward', 'No PPO\nvalue', 'No PPO\nprob', 'Event\nonly', 'K=5', 'K=30']
rates = [91.9, 76.1, 92.8, 91.0, 91.4, 87.7, 93.0]
deltas = [0, -15.8, +0.9, -0.9, -0.5, -4.2, +1.1]

bar_colors = []
for d in deltas:
    if d == 0: bar_colors.append(COLORS['gold'])
    elif d < -5: bar_colors.append(COLORS['red'])
    elif d < 0: bar_colors.append(COLORS['orange'])
    else: bar_colors.append(COLORS['green'])

bars = ax.bar(configs, rates, color=bar_colors, width=0.6, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Win Rate vs Greedy (%)', fontsize=14)
ax.set_title('Component Ablation: What Matters Most?', fontsize=15, fontweight='bold')
ax.set_ylim(70, 100)
ax.axhline(y=91.9, color=COLORS['gold'], linestyle='--', alpha=0.5)

for bar, val, d in zip(bars, rates, deltas):
    label = f'{val}%'
    if d != 0:
        label += f'\n({d:+.1f}pp)'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, label,
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/04_component_ablation.png', bbox_inches='tight')
plt.close()
print("4. Component ablation done")

# ============================================================
# 5. Event Leave-One-Out Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
events = ['buy_card', 'reach_15', 'buy_reserved', 'scarcity_take', 'engine_spike',
          'block_reserve', 'score_up', 'take_gems', 'reserve_card']
loo_rates = [86.4, 89.4, 90.4, 90.6, 90.8, 91.2, 92.4, 92.6, 93.0]
loo_deltas = [-5.5, -2.5, -1.5, -1.3, -1.1, -0.7, +0.5, +0.7, +1.1]

bar_colors = []
for d in loo_deltas:
    if d < -3: bar_colors.append(COLORS['red'])
    elif d < 0: bar_colors.append(COLORS['orange'])
    else: bar_colors.append(COLORS['green'])

bars = ax.barh(events, loo_deltas, color=bar_colors, height=0.6, edgecolor='white', linewidth=1.5)
ax.set_xlabel('Impact on Win Rate (pp)', fontsize=14)
ax.set_title('Event Leave-One-Out: Which Event Matters Most?', fontsize=15, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlim(-7, 2)

for bar, d, r in zip(bars, loo_deltas, loo_rates):
    x = d - 0.3 if d < 0 else d + 0.1
    ax.text(x, bar.get_y() + bar.get_height()/2, f'{d:+.1f}pp ({r}%)',
            ha='right' if d < 0 else 'left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/05_event_leave_one_out.png', bbox_inches='tight')
plt.close()
print("5. Event leave-one-out done")

# ============================================================
# 6. Event Solo Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
solo_events = ['score_up', 'engine_spike', 'reach_15', 'buy_card']
solo_rates = [86.8, 85.8, 81.2, 77.8]

bars = ax.bar(solo_events, solo_rates, color=[COLORS['green'], COLORS['purple'], COLORS['gold'], COLORS['blue']],
              width=0.5, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Win Rate vs Greedy (%)', fontsize=14)
ax.set_title('Solo Event: Each Event Alone', fontsize=15, fontweight='bold')
ax.set_ylim(70, 95)
ax.axhline(y=91.9, color=COLORS['gold'], linestyle='--', alpha=0.5, label='Full model (91.9%)')
ax.axhline(y=78.0, color=COLORS['blue'], linestyle='--', alpha=0.5, label='PPO baseline (78.0%)')
ax.legend(fontsize=10)

for bar, val in zip(bars, solo_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/06_event_solo.png', bbox_inches='tight')
plt.close()
print("6. Event solo done")

# ============================================================
# 7. Head-to-Head Result
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
matchups = ['PPO+Lookahead\nvs PPO pure', 'Self-play\n(sanity check)']
rates = [82.2, 53.5]
colors = [COLORS['gold'], COLORS['gray']]

bars = ax.bar(matchups, rates, color=colors, width=0.4, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Win Rate (%)', fontsize=14)
ax.set_title('Head-to-Head: Direct Matchup (n=1000)', fontsize=15, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (fair)')
ax.legend(fontsize=10)

for bar, val in zip(bars, rates):
    ci = '[79.7%, 84.5%]' if val > 80 else '[46.6%, 60.3%]'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val}%\n{ci}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/07_head_to_head.png', bbox_inches='tight')
plt.close()
print("7. Head-to-head done")

# ============================================================
# 8. Final Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

data = [
    ['PPO Sparse', '~3%', '—', 'Win/loss reward only'],
    ['PPO Score (V4a)', '75.8%', '[72.9%, 78.4%]', 'Score progress reward'],
    ['PPO Event (V5)', '78.0%', '[75.3%, 80.5%]', '9-dim event reward shaping'],
    ['AlphaZero (best)', '24%', '—', 'MCTS 50 sims (failed)'],
    ['PPO+Lookahead', '91.9%', '[90.0%, 93.4%]', '1-step search + event eval'],
    ['PPO+Lookahead K=30', '93.0%', '[91.2%, 94.4%]', 'Best configuration'],
]
cols = ['Agent', 'vs Greedy', '95% CI', 'Method']

table = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

for j in range(len(cols)):
    table[0, j].set_facecolor(COLORS['dark'])
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight best
for j in range(len(cols)):
    table[5, j].set_facecolor('#fff3cd')
    table[6, j].set_facecolor('#d4edda')

# AlphaZero red
table[4, 1].set_text_props(color='red', fontweight='bold')

ax.set_title('Final Agent Comparison (n=1000, vs GreedyAgentBoost)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT}/08_final_summary.png', bbox_inches='tight')
plt.close()
print("8. Final summary done")

print(f"\nAll 8 figures saved to {OUT}/")
