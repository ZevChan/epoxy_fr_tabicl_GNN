"""Figure: 3x3 comprehensive validation + fine-tuning + AD expansion."""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "translated_text2"
OUT_DIR.mkdir(exist_ok=True)

# ── Data ──
df_val = pd.read_csv(HERE.parent / "experiment_predictions/experiment_validation_full.csv")
df_ad = pd.read_csv(HERE.parent / "experiment_predictions/ad_expansion_analysis.csv")

cv_data = {
    'LOI':     {'Published': 0.8498, 'Reproduced': 0.8428, 'FineTuned': 0.8353, 'N_new': 10},
    'Tg':      {'Published': 0.9038, 'Reproduced': 0.8970, 'FineTuned': 0.9006, 'N_new': 6},
    'TENSILE': {'Published': 0.7223, 'Reproduced': 0.6839, 'FineTuned': 0.6652, 'N_new': 12},
    'UL94':    {'Published': 0.9030, 'Reproduced': 0.8979, 'FineTuned': 0.8956, 'N_new': 6},
}

# ── Style ──
INK = '#333333'; GRID = '#E0E0E0'; EXPLICIT = '#1F4E79'; RESIN = '#4A7B9D'
FR = '#E08E36'; GRAPH = '#C0504D'; RED = '#C62828'; GREEN = '#2E7D32'; GRAY = '#9E9E9E'
tgt_colors = {'LOI': EXPLICIT, 'Tg': RESIN, 'TENSILE': FR, 'UL94': GRAPH}
tgt_names = {'LOI': 'LOI', 'Tg': 'Tg', 'TENSILE': 'Tensile', 'UL94': 'UL-94'}

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 20, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
    'legend.fontsize': 16, 'axes.linewidth': 1.3,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

def panel_label(ax, s, x=-0.07, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=30, fontweight='bold', va='bottom', color=INK)

def style(ax):
    for sp in ['top','right','bottom','left']:
        ax.spines[sp].set_visible(True); ax.spines[sp].set_linewidth(0.8); ax.spines[sp].set_color(INK)
    ax.grid(True, color=GRID, linestyle='--', linewidth=0.5, alpha=0.7)

def metric_box(ax, N, mae, rmse):
    t = f"N={N}\nMAE={mae:.2f}\nRMSE={rmse:.2f}"
    p = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=INK, alpha=0.95, linewidth=0.8)
    ax.text(0.97, 0.03, t, transform=ax.transAxes, fontsize=15, va='bottom', ha='right',
            fontweight='bold', color=INK, bbox=p, family='monospace')

def title_text(ax, text, color):
    ax.text(0.03, 0.93, text, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', color=color)

# ── Build Figure ──
fig = plt.figure(figsize=(34, 27), dpi=300)
gs_main = GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.28)

# ============ ROW 1: Parity Plots (a, b, c) ============
for col_i, (tgt, exp_col, pred_col, L, color, title) in enumerate([
    ('LOI', 'LOI_exp', 'LOI_pred', [20, 37], EXPLICIT, 'LOI'),
    ('Tg', 'Tg_exp', 'Tg_pred', [75, 190], RESIN, 'Tg'),
    ('Tensile', 'Tensile_exp', 'TENSILE_pred', [0, 110], FR, 'Tensile Strength'),
]):
    ax = fig.add_subplot(gs_main[0, col_i])
    panel_label(ax, chr(ord('a')+col_i))
    title_text(ax, title, color)
    v = df_val[exp_col].notna()
    x, y = df_val.loc[v, exp_col].values, df_val.loc[v, pred_col].values
    ax.scatter(x, y, c=color, s=100, zorder=5, edgecolors='white', linewidth=0.6)
    ax.plot(L, L, '--', color=INK, linewidth=1.0, alpha=0.4)
    ax.set_xlim(L); ax.set_ylim(L)
    ax.set_xlabel(f'Experimental {title}', fontweight='bold')
    ax.set_ylabel(f'Predicted {title}', fontweight='bold')
    metric_box(ax, len(x), np.abs(y-x).mean(), np.sqrt(np.mean((y-x)**2)))
    style(ax)

# ============ ROW 2: d(Confusion), e(5-CV), f(KDE) ============

# ── (d) UL94 Confusion Matrix ──
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

ax_d = fig.add_subplot(gs_main[1, 0])
panel_label(ax_d, 'd')
df_ul = df_val[df_val['UL94_exp'].notna()]
y_true = df_ul['UL94_exp'].astype(int).values
y_pred_class = (df_ul['UL94_pred'] >= 0.5).astype(int).values
cm = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
cmap_cm = LinearSegmentedColormap.from_list('cmap_cm', ['#F5F5F5', GRAPH])
im = ax_d.imshow(cm, cmap=cmap_cm, aspect='auto', vmin=0, vmax=cm.max())
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        lbl = {(0,0): 'TN', (0,1): 'FP', (1,0): 'FN', (1,1): 'TP'}.get((i,j), '')
        tc = 'white' if count > cm.max()/2 else INK
        ax_d.text(j, i, f'{lbl}\n{count}', ha='center', va='center', fontsize=22, fontweight='bold', color=tc)
ax_d.set_xticks([0, 1]); ax_d.set_xticklabels(['Pred Fail', 'Pred V-0'], fontsize=15, fontweight='bold')
ax_d.set_yticks([0, 1]); ax_d.set_yticklabels(['Actual Fail', 'Actual V-0'], fontsize=15, fontweight='bold')
acc = (cm[0,0] + cm[1,1]) / cm.sum()
style(ax_d)

# ── (e) 5-CV Grouped Bar ──
ax_e = fig.add_subplot(gs_main[1, 1])
panel_label(ax_e, 'e')
targets = ['LOI', 'Tg', 'TENSILE', 'UL94']
x = np.arange(len(targets))
w = 0.22
bars_pub = ax_e.bar(x - w, [cv_data[t]['Published'] for t in targets], w, color=GRAY, label='Published 5-CV', edgecolor='white')
bars_rep = ax_e.bar(x, [cv_data[t]['Reproduced'] for t in targets], w, color=[tgt_colors[t] for t in targets], alpha=0.7, label='Reproduced 5-CV', edgecolor='white')
bars_ft  = ax_e.bar(x + w, [cv_data[t]['FineTuned'] for t in targets], w, color=RED, alpha=0.7, label='Fine-tuned 5-CV', edgecolor='white')
for i, t in enumerate(targets):
    cv = cv_data[t]
    dp = cv['FineTuned'] - cv['Published']
    dr = cv['FineTuned'] - cv['Reproduced']
    if t == 'TENSILE':
        ax_e.text(i + w, cv['FineTuned'] + 0.06, f"dPub={dp:+.3f}", ha='center', fontsize=11, fontweight='bold', color=RED)
        ax_e.text(i + w, cv['FineTuned'] + 0.03, f"dRep={dr:+.3f}\nN={cv['N_new']}", ha='center', fontsize=10, fontweight='bold', color=INK, alpha=0.7)
    else:
        ax_e.text(i + w, cv['FineTuned'] + 0.03, f"dPub={dp:+.3f}\nN={cv['N_new']}", ha='center', fontsize=11, fontweight='bold', color=RED)
        ax_e.text(i, cv['Reproduced'] + 0.015, f"dRep={dr:+.3f}", ha='center', fontsize=10, fontweight='bold', color=INK, alpha=0.6)
ax_e.set_xticks(x)
ax_e.set_xticklabels([tgt_names[t] for t in targets], fontsize=15, fontweight='bold')
ax_e.set_ylabel('Score (R2 / AUC)', fontweight='bold')
ax_e.legend(fontsize=16, loc='upper left')
ax_e.set_ylim(0.6, 1.0)
style(ax_e)

# ── (f) Tensile KDE ──
ax_f = fig.add_subplot(gs_main[1, 2])
panel_label(ax_f, 'f')
exp_v = df_val['Tensile_exp'].dropna().values
pred_v = df_val['TENSILE_pred'].dropna().values
bw = 3.0
for vals, color, label, ls in [(exp_v, INK, 'Experimental', '-'), (pred_v, RED, 'Predicted', '--')]:
    kde = gaussian_kde(vals, bw_method=bw / np.std(vals))
    xs = np.linspace(0, 110, 200)
    ax_f.plot(xs, kde(xs), color=color, linewidth=3.0, linestyle=ls, label=label, alpha=0.9)
    ax_f.plot(vals, np.zeros_like(vals) - 0.001*kde(xs).max(), '|', color=color, markersize=10, alpha=0.6)
ax_f.set_xlabel('Tensile Strength (MPa)', fontweight='bold')
ax_f.set_ylabel('Density', fontweight='bold')
ax_f.legend(fontsize=16, loc='upper right')
style(ax_f)

# ============ ROW 3: g(AD bar), h(AD scatter), i(Quadrant square) ============

# ── (g) AD bar ──
ax_g = fig.add_subplot(gs_main[2, 0])
panel_label(ax_g, 'g')
df_s = df_val.sort_values('AD_Distance')
ax_g.bar(range(34), df_s['AD_Distance'], color=RED, alpha=0.8, width=0.7, edgecolor='white', linewidth=0.2)
ax_g.set_yscale('log')
t = df_val['AD_LOI_threshold'].mean()
ax_g.axhline(y=t, color=INK, linestyle='--', linewidth=1.5)
ax_g.text(32.5, t*1.5, f'Representative AD threshold = {t:.1f}', ha='right', fontsize=14, fontweight='bold', color=INK)
ax_g.set_xlabel('Sample index (sorted)', fontweight='bold')
ax_g.set_ylabel('AD distance (log)', fontweight='bold')
style(ax_g)

# ── (h) AD Evolution Scatter ──
ax_h = fig.add_subplot(gs_main[2, 1])
panel_label(ax_h, 'h')
for tgt in df_ad['Target'].unique():
    sub = df_ad[df_ad['Target'] == tgt]
    ax_h.scatter(sub['AD_base'], sub['AD_finetune'], c=tgt_colors[tgt], s=180,
                zorder=5, edgecolors='white', linewidth=0.8, alpha=0.85, label=tgt_names[tgt])
all_vals = np.concatenate([df_ad['AD_base'].values, df_ad['AD_finetune'].values])
all_vals = all_vals[all_vals > 0]
ax_min = max(all_vals.min() * 0.5, 1e-1)
ax_max = all_vals.max() * 2
ax_h.plot([ax_min, ax_max], [ax_min, ax_max], '--', color=RED, linewidth=1.5, alpha=0.7)
ax_h.set_xscale('log'); ax_h.set_yscale('log')
ax_h.set_xlim(ax_min, ax_max); ax_h.set_ylim(ax_min, ax_max)
ax_h.set_xlabel('Base AD Distance', fontweight='bold')
ax_h.set_ylabel('Fine-tuned AD Distance (OOF)', fontweight='bold')
ax_h.legend(fontsize=16, loc='lower right')
style(ax_h)

# ── (i) Quadrant scatter (square) ──
ax_i = fig.add_subplot(gs_main[2, 2])
panel_label(ax_i, 'i')
for tgt in df_ad['Target'].unique():
    sub = df_ad[df_ad['Target'] == tgt]
    ax_i.scatter(sub['Delta_AD'], sub['Delta_Error'], c=tgt_colors[tgt], s=180,
                 zorder=5, edgecolors='white', linewidth=1.0, alpha=0.85, label=tgt_names[tgt])
ax_i.legend(fontsize=16, loc='lower left')
ax_i.axhline(y=0, color=INK, linewidth=1.2, alpha=0.4)
ax_i.axvline(x=0, color=INK, linewidth=1.2, alpha=0.4)
xl = ax_i.get_xlim(); yl = ax_i.get_ylim()
for (xq, yq, txt, c) in [
    (xl[0]*0.5, yl[1]*0.85, 'AD down / Error up', GRAY),
    (xl[1]*0.5, yl[1]*0.85, 'AD up / Error up', GRAY),
    (xl[0]*0.5, yl[0]*0.05, 'AD down / Error down', GREEN),
    (xl[1]*0.5, yl[0]*0.05, 'AD up / Error down', GRAY),
]:
    ax_i.text(xq, yq, txt, fontsize=16, color=c, ha='center', fontweight='bold', alpha=0.85)
ax_i.set_xlabel('Delta AD (Fine-tuned - Original)', fontweight='bold')
ax_i.set_ylabel('Delta |Error| (Fine-tuned - Original)', fontweight='bold')
style(ax_i)

# ── Final combined ──
out_png = OUT_DIR / 'Figure_comprehensive_3x3.png'
out_pdf = OUT_DIR / 'Figure_comprehensive_3x3.pdf'
fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_png}')
print(f'Saved {out_pdf}')

# ── Individual panels ──
panel_configs = {
    'a': ('LOI', EXPLICIT),
    'b': ('Tg', RESIN),
    'c': ('Tensile Strength', FR),
    'd': ('UL-94 Confusion', None),
    'e': ('5-CV Comparison', None),
    'f': ('Tensile KDE', None),
    'g': ('AD Distance', None),
    'h': ('AD Evolution', None),
    'i': ('AD-Error Trade-off', None),
}

# Regenerate each panel as standalone figure
for panel_id, info in panel_configs.items():
    fig_s, ax_s = plt.subplots(figsize=(8, 7), dpi=300)
    
    if panel_id == 'a':
        v = df_val['LOI_exp'].notna()
        x, y = df_val.loc[v, 'LOI_exp'].values, df_val.loc[v, 'LOI_pred'].values
        ax_s.scatter(x, y, c=EXPLICIT, s=120, zorder=5, edgecolors='white', linewidth=0.8)
        L = [20, 37]; ax_s.plot(L, L, '--', color=INK, linewidth=1.0, alpha=0.4)
        ax_s.set_xlim(L); ax_s.set_ylim(L)
        ax_s.set_xlabel('Experimental LOI (%)', fontweight='bold', fontsize=16)
        ax_s.set_ylabel('Predicted LOI (%)', fontweight='bold', fontsize=16)
        metric_box(ax_s, len(x), np.abs(y-x).mean(), np.sqrt(np.mean((y-x)**2)))
        title_text(ax_s, 'LOI', EXPLICIT)
        
    elif panel_id == 'b':
        v = df_val['Tg_exp'].notna()
        x, y = df_val.loc[v, 'Tg_exp'].values, df_val.loc[v, 'Tg_pred'].values
        ax_s.scatter(x, y, c=RESIN, s=120, zorder=5, edgecolors='white', linewidth=0.8)
        L = [75, 190]; ax_s.plot(L, L, '--', color=INK, linewidth=1.0, alpha=0.4)
        ax_s.set_xlim(L); ax_s.set_ylim(L)
        ax_s.set_xlabel('Experimental Tg (°C)', fontweight='bold', fontsize=16)
        ax_s.set_ylabel('Predicted Tg (°C)', fontweight='bold', fontsize=16)
        metric_box(ax_s, len(x), np.abs(y-x).mean(), np.sqrt(np.mean((y-x)**2)))
        title_text(ax_s, 'Tg', RESIN)
        
    elif panel_id == 'c':
        v = df_val['Tensile_exp'].notna()
        x, y = df_val.loc[v, 'Tensile_exp'].values, df_val.loc[v, 'TENSILE_pred'].values
        ax_s.scatter(x, y, c=FR, s=120, zorder=5, edgecolors='white', linewidth=0.8)
        L = [0, 110]; ax_s.plot(L, L, '--', color=INK, linewidth=1.0, alpha=0.4)
        ax_s.set_xlim(L); ax_s.set_ylim(L)
        ax_s.set_xlabel('Experimental Tensile (MPa)', fontweight='bold', fontsize=16)
        ax_s.set_ylabel('Predicted Tensile (MPa)', fontweight='bold', fontsize=16)
        metric_box(ax_s, len(x), np.abs(y-x).mean(), np.sqrt(np.mean((y-x)**2)))
        title_text(ax_s, 'Tensile Strength', FR)
        
    elif panel_id == 'd':
        cmap_cm = LinearSegmentedColormap.from_list('cmap_cm', ['#F5F5F5', GRAPH])
        im = ax_s.imshow(cm, cmap=cmap_cm, aspect='auto', vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                lbl = {(0,0): 'TN', (0,1): 'FP', (1,0): 'FN', (1,1): 'TP'}.get((i,j), '')
                tc = 'white' if count > cm.max()/2 else INK
                ax_s.text(j, i, f'{lbl}\n{count}', ha='center', va='center', fontsize=24, fontweight='bold', color=tc)
        ax_s.set_xticks([0, 1]); ax_s.set_xticklabels(['Pred Fail', 'Pred V-0'], fontsize=16, fontweight='bold')
        ax_s.set_yticks([0, 1]); ax_s.set_yticklabels(['Actual Fail', 'Actual V-0'], fontsize=16, fontweight='bold')
        
    elif panel_id == 'e':
        targets_cv = ['LOI', 'Tg', 'TENSILE', 'UL94']
        x = np.arange(len(targets_cv)); w = 0.25
        ax_s.bar(x - w, [cv_data[t]['Published'] for t in targets_cv], w, color=GRAY, label='Published 5-CV', edgecolor='white')
        ax_s.bar(x, [cv_data[t]['Reproduced'] for t in targets_cv], w, color=[tgt_colors[t] for t in targets_cv], alpha=0.7, label='Reproduced 5-CV', edgecolor='white')
        ax_s.bar(x + w, [cv_data[t]['FineTuned'] for t in targets_cv], w, color=RED, alpha=0.7, label='Fine-tuned 5-CV', edgecolor='white')
        for i, t in enumerate(targets_cv):
            cv = cv_data[t]; dp = cv['FineTuned'] - cv['Published']
            ax_s.text(i + w, cv['FineTuned'] + 0.02, f"dPub={dp:+.3f}\nN={cv['N_new']}", ha='center', fontsize=12, fontweight='bold', color=RED)
        ax_s.set_xticks(x); ax_s.set_xticklabels([tgt_names[t] for t in targets_cv], fontsize=16, fontweight='bold')
        ax_s.set_ylabel('Score (R2 / AUC)', fontweight='bold', fontsize=16)
        ax_s.legend(fontsize=16, loc='upper left'); ax_s.set_ylim(0.6, 1.0)
        
    elif panel_id == 'f':
        for vals, color, label, ls in [(exp_v, INK, 'Experimental', '-'), (pred_v, RED, 'Predicted', '--')]:
            kde = gaussian_kde(vals, bw_method=bw / np.std(vals))
            xs_kde = np.linspace(0, 110, 200)
            ax_s.plot(xs_kde, kde(xs_kde), color=color, linewidth=3.0, linestyle=ls, label=label, alpha=0.9)
            ax_s.plot(vals, np.zeros_like(vals) - 0.001*kde(xs_kde).max(), '|', color=color, markersize=12, alpha=0.6)
        ax_s.set_xlabel('Tensile Strength (MPa)', fontweight='bold', fontsize=16)
        ax_s.set_ylabel('Density', fontweight='bold', fontsize=16)
        ax_s.legend(fontsize=16, loc='upper right')
        
    elif panel_id == 'g':
        ad_thresh = df_val['AD_LOI_threshold'].mean()
        ax_s.bar(range(34), df_s['AD_Distance'], color=RED, alpha=0.8, width=0.7, edgecolor='white', linewidth=0.2)
        ax_s.set_yscale('log')
        ax_s.axhline(y=ad_thresh, color=INK, linestyle='--', linewidth=1.5)
        ax_s.text(32.5, ad_thresh*1.5, f'Representative AD threshold = {ad_thresh:.1f}', ha='right', fontsize=14, fontweight='bold', color=INK)
        ax_s.set_xlabel('Sample index (sorted)', fontweight='bold', fontsize=16)
        ax_s.set_ylabel('AD distance (log)', fontweight='bold', fontsize=16)
        
    elif panel_id == 'h':
        for tgt in df_ad['Target'].unique():
            sub = df_ad[df_ad['Target'] == tgt]
            ax_s.scatter(sub['AD_base'], sub['AD_finetune'], c=tgt_colors[tgt], s=180,
                        zorder=5, edgecolors='white', linewidth=1.0, alpha=0.85, label=tgt_names[tgt])
        ax_s.plot([ax_min, ax_max], [ax_min, ax_max], '--', color=RED, linewidth=1.5, alpha=0.7)
        ax_s.set_xscale('log'); ax_s.set_yscale('log')
        ax_s.set_xlim(ax_min, ax_max); ax_s.set_ylim(ax_min, ax_max)
        ax_s.set_xlabel('Base AD Distance', fontweight='bold', fontsize=16)
        ax_s.set_ylabel('Fine-tuned AD Distance (OOF)', fontweight='bold', fontsize=16)
        ax_s.legend(fontsize=16, loc='lower right')
        
    elif panel_id == 'i':
        for tgt in df_ad['Target'].unique():
            sub = df_ad[df_ad['Target'] == tgt]
            ax_s.scatter(sub['Delta_AD'], sub['Delta_Error'], c=tgt_colors[tgt], s=200,
                        zorder=5, edgecolors='white', linewidth=1.0, alpha=0.85, label=tgt_names[tgt])
        ax_s.legend(fontsize=16, loc='lower left')
        ax_s.axhline(y=0, color=INK, linewidth=1.2, alpha=0.4)
        ax_s.axvline(x=0, color=INK, linewidth=1.2, alpha=0.4)
        xl = ax_s.get_xlim(); yl = ax_s.get_ylim()
        for (xq, yq, txt, c) in [
            (xl[0]*0.5, yl[1]*0.85, 'AD down / Error up', GRAY),
            (xl[1]*0.5, yl[1]*0.85, 'AD up / Error up', GRAY),
            (xl[0]*0.5, yl[0]*0.05, 'AD down / Error down', GREEN),
            (xl[1]*0.5, yl[0]*0.05, 'AD up / Error down', GRAY),
        ]:
            ax_s.text(xq, yq, txt, fontsize=16, color=c, ha='center', fontweight='bold', alpha=0.85)
        ax_s.set_xlabel('Delta AD', fontweight='bold', fontsize=16)
        ax_s.set_ylabel('Delta |Error|', fontweight='bold', fontsize=16)
    
    style(ax_s)
    fig_s.tight_layout()
    fig_s.savefig(OUT_DIR / f'Figure_comprehensive_3x3_panel{panel_id}.png', dpi=600, bbox_inches='tight')
    fig_s.savefig(OUT_DIR / f'Figure_comprehensive_3x3_panel{panel_id}.pdf', bbox_inches='tight')
    plt.close(fig_s)
    print(f'  Panel {panel_id} saved')

print('All individual panels saved.')
