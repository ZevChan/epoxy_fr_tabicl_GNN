"""Figure: 2x3 external validation panels."""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "translated_text2"
OUT_DIR.mkdir(exist_ok=True)
df = pd.read_csv(HERE.parent / "experiment_predictions/experiment_validation_full.csv")

INK = '#333333'; GRID = '#E0E0E0'
EXPLICIT = '#1F4E79'; RESIN = '#4A7B9D'; FR = '#E08E36'; RED = '#C62828'; GRAPH = '#C0504D'

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 1.2, 'axes.labelsize': 18, 'xtick.labelsize': 15,
    'ytick.labelsize': 15, 'legend.fontsize': 14,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

def add_panel_label(ax, label, x=-0.08, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=26, fontweight='bold', va='bottom', color=INK)

def add_title_box(ax, text, color):
    ax.text(0.03, 0.93, text, transform=ax.transAxes, fontsize=18, fontweight='bold',
            va='top', ha='left', color=color)

def style_ax(ax):
    for s in ['top','right','bottom','left']:
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.8); ax.spines[s].set_color(INK)
    ax.tick_params(width=0.8, colors=INK)
    ax.grid(True, color=GRID, linestyle='--', linewidth=0.5, alpha=0.7)

def add_metric_box(ax, N, mae, rmse):
    text = f"N = {N}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=INK, alpha=0.95, linewidth=1.0)
    ax.text(0.97, 0.03, text, transform=ax.transAxes, fontsize=13, va='bottom', ha='right',
            fontweight='bold', color=INK, bbox=props, family='monospace')

# ===== BUILD FIGURE =====
fig = plt.figure(figsize=(26, 13.5), dpi=300)
gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.32)

# ── (a) LOI ──
ax_a = fig.add_subplot(gs[0, 0])
add_panel_label(ax_a, 'a')
add_title_box(ax_a, 'LOI', EXPLICIT)
v = df['LOI_exp'].notna()
x, y = df.loc[v, 'LOI_exp'].values, df.loc[v, 'LOI_pred'].values
ax_a.scatter(x, y, c=EXPLICIT, s=100, zorder=5, edgecolors='white', linewidth=0.6)
L = [20, 37]; ax_a.plot(L, L, '--', color=INK, linewidth=1.0, alpha=0.4)
ax_a.set_xlim(L); ax_a.set_ylim(L)
ax_a.set_xlabel('Experimental LOI (%)', fontweight='bold')
ax_a.set_ylabel('Predicted LOI (%)', fontweight='bold')
add_metric_box(ax_a, len(x), np.abs(y-x).mean(), np.sqrt(np.mean((y-x)**2)))
style_ax(ax_a)

# ── (b) Tg ──
ax_b = fig.add_subplot(gs[0, 1])
add_panel_label(ax_b, 'b')
add_title_box(ax_b, 'Tg', RESIN)
v = df['Tg_exp'].notna()
x, y = df.loc[v, 'Tg_exp'].values, df.loc[v, 'Tg_pred'].values
ax_b.scatter(x, y, c=RESIN, s=100, zorder=5, edgecolors='white', linewidth=0.6)
L = [75, 190]; ax_b.plot(L, L, '--', color=INK, linewidth=1.0, alpha=0.4)
ax_b.set_xlim(L); ax_b.set_ylim(L)
ax_b.set_xlabel('Experimental Tg (\u00b0C)', fontweight='bold')
ax_b.set_ylabel('Predicted Tg (\u00b0C)', fontweight='bold')
add_metric_box(ax_b, len(x), np.abs(y-x).mean(), np.sqrt(np.mean((y-x)**2)))
style_ax(ax_b)

# ── (c) Tensile ──
ax_c = fig.add_subplot(gs[0, 2])
add_panel_label(ax_c, 'c')
add_title_box(ax_c, 'Tensile Strength', FR)
v = df['Tensile_exp'].notna()
x, y = df.loc[v, 'Tensile_exp'].values, df.loc[v, 'TENSILE_pred'].values
ax_c.scatter(x, y, c=FR, s=100, zorder=5, edgecolors='white', linewidth=0.6)
L = [0, 110]; ax_c.plot(L, L, '--', color=INK, linewidth=1.0, alpha=0.4)
ax_c.set_xlim(L); ax_c.set_ylim(L)
ax_c.set_xlabel('Experimental Tensile (MPa)', fontweight='bold')
ax_c.set_ylabel('Predicted Tensile (MPa)', fontweight='bold')
add_metric_box(ax_c, len(x), np.abs(y-x).mean(), np.sqrt(np.mean((y-x)**2)))
style_ax(ax_c)

# ── (d) Tensile KDE ──
ax_d = fig.add_subplot(gs[1, 0])
add_panel_label(ax_d, 'd')
exp_v = df['Tensile_exp'].dropna().values
pred_v = df['TENSILE_pred'].dropna().values
bw = 3.0
for vals, color, label, lw, ls in [(exp_v, INK, 'Experimental', 3.0, '-'), (pred_v, RED, 'Predicted', 3.0, '--')]:
    kde = gaussian_kde(vals, bw_method=bw / np.std(vals))
    xs = np.linspace(0, 110, 200)
    ax_d.plot(xs, kde(xs), color=color, linewidth=lw, linestyle=ls, label=label, alpha=0.9)
    ax_d.plot(vals, np.zeros_like(vals) - 0.001*kde(xs).max(), '|', color=color, markersize=10, alpha=0.6)
ax_d.set_xlabel('Tensile Strength (MPa)', fontweight='bold')
ax_d.set_ylabel('Probability Density', fontweight='bold')
ax_d.legend(fontsize=14, loc='upper right')
style_ax(ax_d)

# ── (e) AD distance ──
ax_e = fig.add_subplot(gs[1, 1])
add_panel_label(ax_e, 'e')
df_s = df.sort_values('AD_Distance')
x_p = np.arange(34)
colors_e = [RED if d > df['AD_LOI_threshold'].mean() else '#2E7D32' for d in df_s['AD_Distance']]
ax_e.bar(x_p, df_s['AD_Distance'], color=colors_e, alpha=0.85, width=0.7, edgecolor='white', linewidth=0.3)
ax_e.set_yscale('log')
t = df['AD_LOI_threshold'].mean()
ax_e.axhline(y=t, color=INK, linestyle='--', linewidth=1.5)
ax_e.text(32.5, t*1.3, f'AD threshold ({t:.1f})', ha='right', fontsize=14, fontweight='bold', color=INK, va='bottom')
ax_e.set_xlabel('External sample index (sorted)', fontweight='bold')
ax_e.set_ylabel('AD distance (log scale)', fontweight='bold')
style_ax(ax_e)

# ── (f) UL94 risk (only 6 samples with experimental data) ──
ax_f = fig.add_subplot(gs[1, 2])
add_panel_label(ax_f, 'f')
df_ul = df[df['UL94_exp'].notna()].copy()
risk_counts = df_ul['UL94_Risk_Level'].value_counts()
risk_order = ['High-confidence V-0', 'Uncertain', 'Low V-0 probability']
risk_colors = {'High-confidence V-0': '#C62828', 'Uncertain': '#E57373', 'Low V-0 probability': '#FFCDD2'}
short_labels = {'High-confidence V-0': 'High V-0', 'Uncertain': 'Uncertain', 'Low V-0 probability': 'Low V-0'}
sizes = [risk_counts.get(r, 0) for r in risk_order]
colors_f = [risk_colors[r] for r in risk_order]
labels_f = [short_labels[r] for r in risk_order]

wedges, texts, autotexts = ax_f.pie(
    sizes, labels=labels_f, colors=colors_f, startangle=90,
    autopct='%1.0f%%', pctdistance=0.78,
    wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 1.5},
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
for at, sz, r in zip(autotexts, sizes, risk_order):
    if sz > 0:
        at.set_fontsize(13)
        at.set_fontweight('bold')
        at.set_color('#333333' if r == 'Low V-0 probability' else 'white')

# Simple legend
legend_labels = [f"{short_labels[r]} (n={risk_counts.get(r, 0)})" for r in risk_order if risk_counts.get(r, 0) > 0]
legend_wedges = [w for r, w in zip(risk_order, wedges) if risk_counts.get(r, 0) > 0]
ax_f.legend(legend_wedges, legend_labels, loc='lower center', fontsize=13,
           framealpha=0.95, edgecolor=INK, fancybox=False, ncol=1)

# Final
out_png = OUT_DIR / 'Figure_external_validation_2x3.png'
out_pdf = OUT_DIR / 'Figure_external_validation_2x3.pdf'
fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_png}')
print(f'Saved {out_pdf}')
