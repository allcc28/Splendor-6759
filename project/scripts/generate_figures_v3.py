"""Generate Yan's figures v3 — unified blue-gray, no red/green."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = "project/docs/figures/v3"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#F5F7FA',
    'axes.facecolor': '#F5F7FA',
    'axes.grid': False,
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Pure blue-gray palette
DARK = '#2C3E50'
PRIMARY = '#2B5797'
SECONDARY = '#5B9BD5'
TERTIARY = '#85C1E9'
LIGHT_BLUE = '#BDD7EE'
LIGHT_BG = '#F5F7FA'
GRAY = '#7F8C8D'
LIGHT_GRAY = '#D5D8DC'
CHARCOAL = '#34495E'

def bottom_box(fig, text):
    fig.text(0.5, 0.02, text, ha='center', fontsize=9.5, fontweight='bold', color=PRIMARY,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT_BLUE, edgecolor=SECONDARY, alpha=0.8))

# ============================================================
# 1. Component Ablation
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
configs = ['Full\n(baseline)', 'No event\nreward', 'No PPO\nvalue', 'No PPO\nprob', 'Event\nonly', 'K=5', 'K=30']
rates = [91.9, 76.1, 92.8, 91.0, 91.4, 87.7, 93.0]
deltas = [0, -15.8, +0.9, -0.9, -0.5, -4.2, +1.1]

# Gradient: baseline=primary, negative=darker, positive=lighter
colors = []
for d in deltas:
    if d == 0: colors.append(PRIMARY)
    elif d < -10: colors.append(CHARCOAL)
    elif d < -2: colors.append(GRAY)
    elif d < 0: colors.append(SECONDARY)
    else: colors.append(TERTIARY)

bars = ax.bar(configs, rates, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Win Rate vs Greedy (%)', fontsize=12, color=DARK)
ax.set_title('Component Ablation: What Matters Most?', fontsize=14, fontweight='bold', color=DARK)
ax.set_ylim(70, 100)
ax.axhline(y=91.9, color=PRIMARY, linestyle='--', alpha=0.2)

for bar, val, d in zip(bars, rates, deltas):
    label = f'{val}%' + (f'\n({d:+.1f}pp)' if d != 0 else '')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, label,
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=DARK)

# Callout arrows instead of colored boxes
ax.annotate('Critical', xy=(1, 76.1), xytext=(1.8, 73),
            fontsize=10, fontweight='bold', color=CHARCOAL,
            arrowprops=dict(arrowstyle='->', color=CHARCOAL, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor=LIGHT_GRAY, edgecolor=CHARCOAL))

ax.annotate('Useless', xy=(2, 92.8), xytext=(2.8, 96),
            fontsize=10, fontweight='bold', color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor=LIGHT_GRAY, edgecolor=GRAY))

bottom_box(fig, 'Event reward is the core (-16pp without it)  |  PPO value has no effect  |  Event alone matches full model')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(f'{OUT}/04_component_ablation.png', bbox_inches='tight')
plt.close()
print("1. Component ablation done")

# ============================================================
# 2. Event Leave-One-Out
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
events = ['buy_card', 'reach_15', 'buy_reserved', 'scarcity_take', 'engine_spike',
          'block_reserve', 'score_up', 'take_gems', 'reserve_card']
loo_deltas = [-5.5, -2.5, -1.5, -1.3, -1.1, -0.7, +0.5, +0.7, +1.1]
loo_rates = [86.4, 89.4, 90.4, 90.6, 90.8, 91.2, 92.4, 92.6, 93.0]

# Blue gradient: more negative = darker
colors = []
for d in loo_deltas:
    if d < -3: colors.append(PRIMARY)
    elif d < -1: colors.append(SECONDARY)
    elif d < 0: colors.append(TERTIARY)
    else: colors.append(LIGHT_GRAY)

bars = ax.barh(events, loo_deltas, color=colors, height=0.6, edgecolor='white', linewidth=1.5)
ax.set_xlabel('Impact on Win Rate (pp)', fontsize=12, color=DARK)
ax.set_title('Event Leave-One-Out: Which Signal Matters Most?', fontsize=14, fontweight='bold', color=DARK)
ax.axvline(x=0, color=DARK, linewidth=1)
ax.set_xlim(-7, 2.5)

for bar, d, r in zip(bars, loo_deltas, loo_rates):
    x = d - 0.2 if d < 0 else d + 0.1
    ax.text(x, bar.get_y() + bar.get_height()/2, f'{d:+.1f}pp ({r}%)',
            ha='right' if d < 0 else 'left', va='center', fontsize=10, fontweight='bold', color=DARK)

# Callout annotations
ax.annotate('Most critical', xy=(-5.5, 0), xytext=(-6.5, 1.5),
            fontsize=9, fontweight='bold', color=PRIMARY,
            arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor=LIGHT_BLUE, edgecolor=PRIMARY))

ax.annotate('Noise\n(removing helps)', xy=(1.1, 8), xytext=(1.5, 6.5),
            fontsize=9, fontweight='bold', color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor=LIGHT_GRAY, edgecolor=GRAY))

bottom_box(fig, 'buy_card is the MVP (-5.5pp)  |  reserve_card is noise (+1.1pp when removed)')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(f'{OUT}/05_event_leave_one_out.png', bbox_inches='tight')
plt.close()
print("2. Event leave-one-out done")

# ============================================================
# 3. Lookahead Results Overview
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
matchups = ['PPO alone\nvs Greedy', 'PPO+Lookahead\nvs Greedy', 'Lookahead\nvs PPO (H2H)', 'Self-play\n(sanity)']
rates = [78.0, 91.9, 82.2, 53.5]
cis = ['[75.3%, 80.5%]', '[90.0%, 93.4%]', '[79.7%, 84.5%]', '[46.6%, 60.3%]']
colors = [GRAY, PRIMARY, SECONDARY, LIGHT_GRAY]

bars = ax.bar(matchups, rates, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Win Rate (%)', fontsize=12, color=DARK)
ax.set_title('PPO+Lookahead: All Key Results (n=1000)', fontsize=14, fontweight='bold', color=DARK)
ax.set_ylim(0, 108)
ax.axhline(y=50, color=LIGHT_GRAY, linestyle='--', alpha=0.7)

for bar, val, ci in zip(bars, rates, cis):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val}%\n{ci}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=DARK)

ax.annotate('+14pp', xy=(1, 92), xytext=(0.5, 100),
            fontsize=12, fontweight='bold', color=PRIMARY,
            arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_BLUE, edgecolor=PRIMARY))

plt.tight_layout()
plt.savefig(f'{OUT}/10_lookahead_results.png', bbox_inches='tight')
plt.close()
print("3. Lookahead results done")

# ============================================================
# 4. Curriculum Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ['PPO alone', '+ Lookahead']
v1 = [78.0, 91.9]
curriculum = [73.7, 91.5]
x = np.arange(len(labels))
w = 0.28

bars1 = ax.bar(x - w/2, v1, w, label='Original V1 (trained vs Random)', color=PRIMARY, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + w/2, curriculum, w, label='Curriculum (trained vs Greedy)', color=SECONDARY, edgecolor='white', linewidth=1.5)

ax.set_ylabel('Win Rate vs Greedy (%)', fontsize=12, color=DARK)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylim(65, 102)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.set_title('Curriculum Training: Does the Base Model Matter?', fontsize=14, fontweight='bold', color=DARK)

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height()}%', ha='center', va='bottom', fontweight='bold', fontsize=12, color=DARK)

ax.annotate('Identical!', xy=(1.3, 92), fontsize=12, fontweight='bold', color=PRIMARY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_BLUE, edgecolor=PRIMARY))
ax.annotate('-4.3pp', xy=(0.3, 71), fontsize=11, fontweight='bold', color=CHARCOAL,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=LIGHT_GRAY, edgecolor=CHARCOAL))

bottom_box(fig, 'PPO is just a candidate filter. The evaluation function does the real work.')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(f'{OUT}/09_curriculum_comparison.png', bbox_inches='tight')
plt.close()
print("4. Curriculum comparison done")

# ============================================================
# 5. Architecture — vertical top-down
# ============================================================
fig, ax = plt.subplots(figsize=(6, 8))
ax.axis('off')
ax.set_xlim(0, 6)
ax.set_ylim(-1, 10.5)

# Boxes top-to-bottom
vboxes = [
    (1.5, 8.5, 3.0, 1.0, 'Legal Actions', GRAY),
    (1.5, 6.5, 3.0, 1.0, 'PPO Filter\nTop K=15', PRIMARY),
    (1.5, 4.5, 3.0, 1.0, '1-Step Forward\nSimulation', SECONDARY),
    (1.5, 2.5, 3.0, 1.0, 'Event\nEvaluation', CHARCOAL),
    (1.5, 0.5, 3.0, 1.0, 'Best Action', PRIMARY),
]
for bx, by, bw, bh, text, color in vboxes:
    rect = mpatches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(bx + bw/2, by + bh/2, text, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')

# Downward arrows between boxes
arrow_kw = dict(arrowstyle='->', color=DARK, lw=2.5)
for y_top, y_bot in [(8.5, 7.5), (6.5, 5.5), (4.5, 3.5), (2.5, 1.5)]:
    ax.annotate('', xy=(3, y_bot), xytext=(3, y_top), arrowprops=arrow_kw)

# Score formula at the bottom
ax.text(3, -0.5, 'Score = 0.3 x PPO_prob + 0.5 x event_reward + 0.2 x future_value',
        fontsize=10, ha='center', style='italic', color=DARK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_BLUE, edgecolor=SECONDARY))

ax.set_title('PPO + Lookahead Architecture', fontsize=15, fontweight='bold', color=DARK, pad=8)
plt.subplots_adjust(top=0.94, bottom=0.02)
plt.savefig(f'{OUT}/03_lookahead_architecture.png', bbox_inches='tight')
plt.close()
print("5. Architecture done")

# ============================================================
# 6. AlphaZero Failure Table
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis('off')

data = [
    ['V3 Shaped (30 iter, 50 sims)', '34.0%', '4.0%'],
    ['V4 Warm-start (60 iter)', '6.0%', '2.0%'],
    ['V3-long (60 iter)', '10.0%', '0.0%'],
    ['Stage C (40 iter, 50 sims)', '30.0%', '5.0%'],
]
cols = ['AlphaZero Variant', 'vs Random', 'vs Greedy']

table = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center',
                 colWidths=[0.45, 0.20, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

for j in range(len(cols)):
    table[0, j].set_facecolor(PRIMARY)
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

for i in range(1, len(data)+1):
    for j in range(len(cols)):
        table[i, j].set_facecolor('white' if i % 2 == 1 else LIGHT_BG)
        table[i, j].set_edgecolor(LIGHT_GRAY)
    table[i, 2].set_text_props(color=CHARCOAL, fontweight='bold')

fig.suptitle('AlphaZero: All Variants Failed', fontsize=15, fontweight='bold', color=DARK, y=0.98)
bottom_box(fig, 'All variants < 10% vs Greedy  |  PPO Score-based already achieves 75.8%')
plt.subplots_adjust(top=0.88, bottom=0.15)
plt.savefig(f'{OUT}/02_alphazero_failure.png', bbox_inches='tight')
plt.close()
print("6. AlphaZero failure done")

# ============================================================
# 7. Final Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

data = [
    ['PPO Sparse', '~3%', '--', 'Win/loss reward only'],
    ['PPO Score (V4a)', '75.8%', '[73.0%, 78.4%]', 'Score progress reward'],
    ['PPO Event (V5)', '78.0%', '[75.3%, 80.5%]', '9-dim event reward shaping'],
    ['AlphaZero (best)', '<10%', '--', 'MCTS 50 sims (failed)'],
    ['PPO+Lookahead', '91.9%', '[90.0%, 93.4%]', '1-step search + event eval'],
    ['PPO+Lookahead K=30', '93.0%', '[91.2%, 94.4%]', 'Best configuration'],
]
cols = ['Agent', 'vs Greedy', '95% CI', 'Method']

table = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center',
                 colWidths=[0.25, 0.13, 0.20, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)

for j in range(len(cols)):
    table[0, j].set_facecolor(PRIMARY)
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(data)+1):
    for j in range(len(cols)):
        table[i, j].set_edgecolor(LIGHT_GRAY)
        if i % 2 == 0:
            table[i, j].set_facecolor(LIGHT_BG)

for j in range(len(cols)):
    table[5, j].set_facecolor(LIGHT_BLUE)
    table[6, j].set_facecolor(LIGHT_BLUE)

fig.suptitle('Final Agent Comparison (n=1000 games each)', fontsize=14, fontweight='bold', color=DARK, y=0.98)
bottom_box(fig, '35+ experiments  |  30,000+ evaluation games  |  111 human-vs-AI battles')
plt.subplots_adjust(top=0.90, bottom=0.12)
plt.savefig(f'{OUT}/08_final_summary.png', bbox_inches='tight')
plt.close()
print("7. Final summary done")

print(f"\nAll v3 figures saved to {OUT}/")
