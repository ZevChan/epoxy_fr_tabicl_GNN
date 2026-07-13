"""Figure: AD expansion paired slope + quadrant scatter."""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "translated_text2"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(HERE.parent / "experiment_predictions/ad_expansion_analysis.csv")

INK = '#333333'; GRID = '#E0E0E0'
EXPLICIT = '#1F4E79'; RESIN = '#4A7B9D'; FR = '#E08E36'; GRAPH = '#C0504D'; RED = '#C62828'; GREEN = '#2E7D32'; GRAY = '#9E9E9E'

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 13, 'axes.linewidth': 1.1,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

def add_panel_label(ax, label, x=-0.08, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=24, fontweight='bold', va='bottom', color=INK)

def style_ax(ax):
    for s in ['top','right','bottom','left']:
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.8); ax.spines[s].set_color(INK)
    ax.grid(True, color=GRID, linestyle='--', linewidth=0.5, alpha=0.7)

tgt_colors = {'LOI': EXPLICIT, 'Tg': RESIN, 'TENSILE': FR, 'UL94': GRAPH}
tgt_names = {'LOI': 'LOI', 'Tg': 'Tg', 'TENSILE': 'Tensile', 'UL94': 'UL-94'}

fig = plt.figure(figsize=(20, 9), dpi=300)
gs = GridSpec(1, 2, figure=fig, wspace=0.30)

# ── (a) Paired slope chart: AD distance before vs after ──
ax_a = fig.add_subplot(gs[0, 0])
add_panel_label(ax_a, 'a')

x_jitter = np.linspace(0.85, 1.15, len(df))
np.random.seed(42)
for _, row in df.iterrows():
    tgt = row['Target']
    base = row['AD_base']
    ft = row['AD_finetune']
    ax_a.plot([0.85, 1.15], [base, ft], '-', color=tgt_colors[tgt], alpha=0.5, linewidth=1.2)
    ax_a.scatter([0.85], [base], c=tgt_colors[tgt], s=50, zorder=5, edgecolors='white', linewidth=0.3)
    ax_a.scatter([1.15], [ft], c=tgt_colors[tgt], s=50, zorder=5, edgecolors='white', linewidth=0.3)

# Threshold line
for tgt in df['Target'].unique():
    thresh = df[df['Target']==tgt]['AD_threshold'].iloc[0]
    ax_a.axhline(y=thresh, color=tgt_colors[tgt], linestyle='--', linewidth=1.0, alpha=0.5)
    ax_a.text(1.17, thresh, f'{tgt_names[tgt]} thr', fontsize=9, color=tgt_colors[tgt], va='center')

ax_a.set_xticks([0.85, 1.15])
ax_a.set_xticklabels(['Original Model', 'Fine-tuned Model\n(OOF)'], fontsize=14, fontweight='bold')
ax_a.set_ylabel('AD Distance (kNN-5 avg)', fontweight='bold')
ax_a.set_yscale('log')
ax_a.set_title('AD Domain Expansion', fontsize=18, fontweight='bold')

# Legend
for tgt in df['Target'].unique():
    ax_a.scatter([], [], c=tgt_colors[tgt], s=50, label=tgt_names[tgt])
ax_a.legend(loc='upper left', fontsize=12)

style_ax(ax_a)

# ── (b) Quadrant scatter: Delta AD vs Delta Error ──
ax_b = fig.add_subplot(gs[0, 1])
add_panel_label(ax_b, 'b')

for _, row in df.iterrows():
    tgt = row['Target']
    ax_b.scatter(row['Delta_AD'], row['Delta_Error'], c=tgt_colors[tgt], s=120,
                zorder=5, edgecolors='white', linewidth=0.8, alpha=0.85)
    # Add target label
    ax_b.annotate(tgt_names[tgt], (row['Delta_AD'], row['Delta_Error']),
                 textcoords="offset points", xytext=(8, 8), fontsize=8, color=tgt_colors[tgt], fontweight='bold')

ax_b.axhline(y=0, color=INK, linewidth=1.2, linestyle='-', alpha=0.4)
ax_b.axvline(x=0, color=INK, linewidth=1.2, linestyle='-', alpha=0.4)

# Quadrant labels
x_lim = ax_b.get_xlim()
y_lim = ax_b.get_ylim()
ax_b.text(x_lim[0]*0.5, y_lim[1]*0.9, 'AD shrinks\nError worse', fontsize=10, color=GRAY, ha='center', fontweight='bold', alpha=0.7)
ax_b.text(x_lim[1]*0.5, y_lim[1]*0.9, 'AD expands\nError worse', fontsize=10, color=GRAY, ha='center', fontweight='bold', alpha=0.7)
ax_b.text(x_lim[0]*0.5, y_lim[0]*0.1, 'AD shrinks\nError improves', fontsize=10, color=GREEN, ha='center', fontweight='bold', alpha=0.9)
ax_b.text(x_lim[1]*0.5, y_lim[0]*0.1, 'AD expands\nError improves', fontsize=10, color=GRAY, ha='center', fontweight='bold', alpha=0.7)

ax_b.set_xlabel('Delta AD (Fine-tuned - Original)', fontweight='bold')
ax_b.set_ylabel('Delta Error (Fine-tuned - Original)', fontweight='bold')
ax_b.set_title('AD-Error Trade-off', fontsize=18, fontweight='bold')

style_ax(ax_b)

# Final
out_png = OUT_DIR / 'Figure_AD_expansion_analysis.png'
out_pdf = OUT_DIR / 'Figure_AD_expansion_analysis.pdf'
fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_png}')
print(f'Saved {out_pdf}')
