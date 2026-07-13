"""Figure 11: External experimental validation and AD boundary analysis."""
import warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "translated_text2"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(HERE.parent / "experiment_predictions/experiment_validation_full.csv")

# ── Style ──
INK = '#333333'
GRID = '#E6E6E6'
EXPLICIT = '#1F4E79'
GRAPH = '#C0504D'
RESIN = '#4A7B9D'
FR = '#E08E36'
GREEN = '#4CAF50'
RED = '#E53935'
GRAY = '#9E9E9E'

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 1.05, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

def add_panel_label(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='bottom', color=INK)

def style_ax(ax):
    for s in ['top','right','bottom','left']:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(INK)
    ax.grid(axis='y', color=GRID, linestyle='--', linewidth=0.5)

# ── Build Figure ──
fig = plt.figure(figsize=(18, 12), dpi=300)

# Colors per target
tgt_colors = {'LOI': EXPLICIT, 'UL94': GRAPH, 'Tg': RESIN, 'TENSILE': FR}
targets_display = {'LOI': 'LOI', 'UL94': 'UL-94', 'Tg': 'Tg', 'TENSILE': 'Tensile'}

# ============ (a) Cohort composition ============
ax_a = fig.add_subplot(3, 3, 1)
add_panel_label(ax_a, 'a')

targets = ['LOI', 'UL94', 'Tg', 'TENSILE']
exp_cols = {'LOI': 'LOI_exp', 'UL94': 'UL94_exp', 'Tg': 'Tg_exp', 'TENSILE': 'Tensile_exp'}
n_exp = [df[exp_cols[t]].notna().sum() for t in targets]
n_miss = [34 - n for n in n_exp]

x = np.arange(len(targets))
bars1 = ax_a.bar(x, n_exp, color=[tgt_colors[t] for t in targets], label='With experiment', width=0.5)
bars2 = ax_a.bar(x, n_miss, bottom=n_exp, color=GRAY, alpha=0.3, label='Missing', width=0.5)
for i, (nexp, nmiss) in enumerate(zip(n_exp, n_miss)):
    if nexp > 0:
        ax_a.text(i, nexp/2, str(nexp), ha='center', va='center', fontweight='bold', fontsize=9, color='white')
ax_a.set_xticks(x)
ax_a.set_xticklabels([targets_display[t] for t in targets], fontsize=10, fontweight='bold')
ax_a.set_ylabel('Number of samples', fontsize=10, fontweight='bold')
ax_a.set_ylim(0, 38)
ax_a.legend(fontsize=8, loc='upper right')
style_ax(ax_a)

# ============ (b) AD diagnosis ============
ax_b = fig.add_subplot(3, 3, 2)
add_panel_label(ax_b, 'b')

df_sorted = df.sort_values('AD_Distance')
colors_b = [RED if x > df_sorted['AD_Distance'].min() else RED for x in df_sorted['AD_Distance']]
# All are Outside AD, so all red
ax_b.bar(range(34), df_sorted['AD_Distance'], color=RED, alpha=0.7, width=0.7)
# AD threshold line
ad_thresh = df['AD_LOI_threshold'].mean()
ax_b.axhline(y=ad_thresh, color=INK, linestyle='--', linewidth=1.2)
ax_b.text(33, ad_thresh, f'AD threshold ({ad_thresh:.1f})', ha='right', va='bottom',
          fontsize=8, color=INK, fontweight='bold')
ax_b.set_xlabel('External samples (sorted by AD distance)', fontsize=10, fontweight='bold')
ax_b.set_ylabel('AD distance (avg kNN 5)', fontsize=10, fontweight='bold')
ax_b.set_xlim(-0.5, 33.5)
style_ax(ax_b)

# ============ (c) LOI pred vs exp ============
ax_c = fig.add_subplot(3, 3, 4)
add_panel_label(ax_c, 'c')

valid_loi = df['LOI_exp'].notna()
x_loi = df.loc[valid_loi, 'LOI_exp']
y_loi = df.loc[valid_loi, 'LOI_pred']
ax_c.scatter(x_loi, y_loi, c=EXPLICIT, s=50, zorder=5, edgecolors='white', linewidth=0.5)
lims = [min(x_loi.min(), y_loi.min())-2, max(x_loi.max(), y_loi.max())+2]
ax_c.plot(lims, lims, '--', color=INK, linewidth=0.8, alpha=0.5)
mae_loi = np.abs(y_loi - x_loi).mean()
ax_c.text(0.05, 0.95, f'MAE = {mae_loi:.2f}\nn = {len(x_loi)}', transform=ax_c.transAxes,
          fontsize=9, va='top', fontweight='bold', color=INK,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax_c.set_xlabel('Experimental LOI (%)', fontsize=10, fontweight='bold')
ax_c.set_ylabel('Predicted LOI (%)', fontsize=10, fontweight='bold')
style_ax(ax_c)

# ============ (d) Tg pred vs exp ============
ax_d = fig.add_subplot(3, 3, 5)
add_panel_label(ax_d, 'd')

valid_tg = df['Tg_exp'].notna()
x_tg = df.loc[valid_tg, 'Tg_exp']
y_tg = df.loc[valid_tg, 'Tg_pred']
ax_d.scatter(x_tg, y_tg, c=RESIN, s=50, zorder=5, edgecolors='white', linewidth=0.5)
lims_tg = [70, 200]
ax_d.plot(lims_tg, lims_tg, '--', color=INK, linewidth=0.8, alpha=0.5)
mae_tg = np.abs(y_tg - x_tg).mean()
ax_d.text(0.05, 0.95, f'MAE = {mae_tg:.1f}°C\nn = {len(x_tg)}', transform=ax_d.transAxes,
          fontsize=9, va='top', fontweight='bold', color=INK,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax_d.set_xlabel('Experimental Tg (°C)', fontsize=10, fontweight='bold')
ax_d.set_ylabel('Predicted Tg (°C)', fontsize=10, fontweight='bold')
style_ax(ax_d)

# ============ (e) TENSILE pred vs exp ============
ax_e = fig.add_subplot(3, 3, 6)
add_panel_label(ax_e, 'e')

valid_te = df['Tensile_exp'].notna()
x_te = df.loc[valid_te, 'Tensile_exp']
y_te = df.loc[valid_te, 'TENSILE_pred']
ax_e.scatter(x_te, y_te, c=FR, s=50, zorder=5, edgecolors='white', linewidth=0.5)
lims_te = [0, 110]
ax_e.plot(lims_te, lims_te, '--', color=INK, linewidth=0.8, alpha=0.5)
mae_te = np.abs(y_te - x_te).mean()
ax_e.text(0.05, 0.95, f'MAE = {mae_te:.1f} MPa\nn = {len(x_te)}', transform=ax_e.transAxes,
          fontsize=9, va='top', fontweight='bold', color=INK,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax_e.set_xlabel('Experimental Tensile (MPa)', fontsize=10, fontweight='bold')
ax_e.set_ylabel('Predicted Tensile (MPa)', fontsize=10, fontweight='bold')
style_ax(ax_e)

# ============ (f) Error vs AD distance ============
ax_f = fig.add_subplot(3, 3, 7)
add_panel_label(ax_f, 'f')

# Combine all regression errors
error_data = []
for tgt, exp_col in [('LOI', 'LOI_exp'), ('Tg', 'Tg_exp'), ('TENSILE', 'Tensile_exp')]:
    valid = df[exp_col].notna()
    for i in df[valid].index:
        err = df.loc[i, f'{tgt}_pred'] - df.loc[i, exp_col]
        error_data.append({
            'Target': tgt,
            'AbsError': abs(err),
            'Error': err,
            'AD_Distance': df.loc[i, 'AD_Distance'],
            'Sample': df.loc[i, 'Sample_ID'],
        })
err_df = pd.DataFrame(error_data)

for tgt in ['LOI', 'Tg', 'TENSILE']:
    sub = err_df[err_df['Target'] == tgt]
    ax_f.scatter(sub['AD_Distance'], sub['AbsError'], c=tgt_colors[tgt], s=40,
                 label=targets_display[tgt], zorder=5, edgecolors='white', linewidth=0.3)
ax_f.axvline(x=ad_thresh, color=INK, linestyle='--', linewidth=1.2)
ax_f.text(ad_thresh+0.3, ax_f.get_ylim()[1]*0.95, 'AD boundary', fontsize=8, color=INK)
ax_f.set_xlabel('AD distance', fontsize=10, fontweight='bold')
ax_f.set_ylabel('Absolute prediction error', fontsize=10, fontweight='bold')
ax_f.legend(fontsize=8)
style_ax(ax_f)

# ============ (g) Case studies ============
ax_g = fig.add_subplot(3, 3, 8)
add_panel_label(ax_g, 'g')

# Pick 4 cases: 2 best agreements (lowest abs LOI error), 2 worst
loi_cases = err_df[err_df['Target'] == 'LOI'].sort_values('AbsError')
best_cases = loi_cases.head(2)
worst_cases = loi_cases.tail(2)
cases = pd.concat([best_cases, worst_cases])

# Get formula info for these cases
case_data = []
for _, c in cases.iterrows():
    row = df[df['Sample_ID'] == c['Sample']].iloc[0]
    fr_name = str(row.get('FR_Name', 'N/A'))[:25]
    cur_name = str(row.get('Curing_Name', 'N/A'))[:15]
    case_data.append({
        'Sample': c['Sample'],
        'FR_Name': fr_name,
        'Curing_Name': cur_name,
        'LOI_exp': row['LOI_exp'],
        'LOI_pred': row['LOI_pred'],
        'AbsError': c['AbsError'],
        'AD_Distance': c['AD_Distance'],
    })
case_df = pd.DataFrame(case_data)
x_pos = np.arange(len(case_df))
width = 0.35
ax_g.bar(x_pos - width/2, case_df['LOI_exp'], width, color=GRAY, label='Experimental', edgecolor='white')
ax_g.bar(x_pos + width/2, case_df['LOI_pred'], width, color=EXPLICIT, label='Predicted', edgecolor='white')
for i, (exp_v, pred_v) in enumerate(zip(case_df['LOI_exp'], case_df['LOI_pred'])):
    ax_g.text(i-width/2, exp_v+0.3, f'{exp_v:.1f}', ha='center', fontsize=7, fontweight='bold')
    ax_g.text(i+width/2, pred_v+0.3, f'{pred_v:.1f}', ha='center', fontsize=7, fontweight='bold')
ax_g.set_xticks(x_pos)
ax_g.set_xticklabels(case_df['Sample'], fontsize=8)
ax_g.set_ylabel('LOI (%)', fontsize=10, fontweight='bold')
ax_g.legend(fontsize=8)
style_ax(ax_g)

# ============ UL94 summary text (right side) ============
ax_ul = fig.add_subplot(3, 3, 3)
ax_ul.axis('off')
add_panel_label(ax_ul, 'h', x=0.0, y=1.05)

ul94_text = (
    "UL-94 V-0 Screening Results\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"High-confidence V-0 candidates\n"
    f"  (probability ≥ 0.60)\n"
    f"  N = {len(df[df['UL94_Risk_Level']=='High-confidence V-0'])}\n\n"
    f"Uncertain\n"
    f"  (probability 0.40–0.60)\n"
    f"  N = {len(df[df['UL94_Risk_Level']=='Uncertain'])}\n\n"
    f"Low V-0 probability\n"
    f"  (probability < 0.40)\n"
    f"  N = {len(df[df['UL94_Risk_Level']=='Low V-0 probability'])}\n\n"
    f"No experimental UL-94 data\n"
    f"available for validation.\n"
    f"Predictions should be\n"
    f"treated as screening only."
)
ax_ul.text(0.0, 0.95, ul94_text, transform=ax_ul.transAxes,
           fontsize=9, va='top', family='monospace', color=INK,
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))

# ============ Final layout ============
fig.suptitle('Figure 11. External experimental stress test and applicability-domain boundary analysis\n'
             'of DescriptorProcess_TabICL',
             fontsize=14, fontweight='bold', color=INK, y=1.01)

fig.tight_layout()
out_png = OUT_DIR / 'Figure11_external_validation.png'
out_pdf = OUT_DIR / 'Figure11_external_validation.pdf'
fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_png}')
print(f'Saved {out_pdf}')
