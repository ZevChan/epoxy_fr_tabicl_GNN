"""Figure: OOD evidence - chemical space, formula distribution, target range."""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "translated_text2"
OUT_DIR.mkdir(exist_ok=True)

INK = '#333333'; GRID = '#E0E0E0'
EXPLICIT = '#1F4E79'; RESIN = '#4A7B9D'; FR = '#E08E36'; GRAPH = '#C0504D'
RED = '#C62828'; GRAY = '#BDBDBD'; BLUE = '#1565C0'; GREEN = '#2E7D32'

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'legend.fontsize': 15, 'axes.linewidth': 1.2,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

def style(ax):
    for s in ['top','right','bottom','left']:
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.8); ax.spines[s].set_color(INK)
    ax.grid(True, color=GRID, linestyle='--', linewidth=0.5, alpha=0.7)

def panel_label(ax, s):
    ax.text(-0.08, 1.03, s, transform=ax.transAxes, fontsize=28, fontweight='bold', va='bottom', color=INK)

# ── Load data ──
# Database samples
db_csv = HERE.parent / "94/EP+FR+CURING_SMILES+translated_text_DATASET_20260414.csv"
df_db = pd.read_csv(db_csv)

# Experimental samples
df_exp = pd.read_csv(HERE.parent / "Experiment.csv")
actual = pd.read_excel(HERE.parent / "Experiment_ActualData.xlsx")

# Compute formula features for database
temp_cols = [f'Curing_Tem{i}' for i in range(1, 10)]
time_cols = [f'Curing_Time{i}' for i in range(1, 10)]
df_db['T_max'] = df_db[temp_cols].max(axis=1)
df_db['t_total'] = df_db[time_cols].sum(axis=1).fillna(0)
df_db['Q_thermal'] = sum(df_db[tc].fillna(0) * df_db[tic].fillna(0) for tc, tic in zip(temp_cols, time_cols))
df_db['EP_wt_fraction'] = (100.0 - df_db['Flame_retardant_AdditionAmount(wt%)'].fillna(0) - df_db['Curing_agent_AdditionAmount(wt%)'].fillna(0)) / 100.0

# Compute formula features for experiment
df_exp['T_max'] = df_exp[temp_cols].max(axis=1)
df_exp['t_total'] = df_exp[time_cols].sum(axis=1).fillna(0)
df_exp['Q_thermal'] = sum(df_exp[tc].fillna(0) * df_exp[tic].fillna(0) for tc, tic in zip(temp_cols, time_cols))
df_exp['EP_wt_fraction'] = (100.0 - df_exp['Flame_retardant_AdditionAmount(wt%)'].fillna(0) - df_exp['Curing_agent_AdditionAmount(wt%)'].fillna(0)) / 100.0

# Experimental actual values
exp_loi = pd.to_numeric(actual['LOI'], errors='coerce').dropna()
exp_tg = pd.to_numeric(actual['Tg'], errors='coerce').dropna()
exp_tensile = pd.to_numeric(actual['Tensile'], errors='coerce').dropna()

# Database target values (from training data in plot_data.pkl)
import pickle as pkl
import os as _os
db_targets = {}
for tgt, path in [('LOI', HERE.parent / 'LOI/Result_Final_TabICL_Interpret'),
                   ('Tg', HERE.parent / 'Tg/Result_Final_TabICL_Interpret'),
                   ('TENSILE', HERE.parent / 'TENSILE/Result_Final_TabICL_Interpret')]:
    sd = path / 'saved_data'
    if not sd.exists():
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        sd = subdirs[0] / 'saved_data'
    with open(sd / 'plot_data.pkl', 'rb') as f:
        data = pkl.load(f)
    db_targets[tgt] = np.concatenate([data['y_train'], data['y_test']])

db_loi = db_targets['LOI']
db_tg = db_targets['Tg']
db_tensile = db_targets['TENSILE']

print(f"DB samples: {len(df_db)}")
print(f"Exp samples: {len(df_exp)}")
print(f"Exp LOI: {len(exp_loi)}, Tg: {len(exp_tg)}, Tensile: {len(exp_tensile)}")
print(f"DB LOI: {len(db_loi)}, Tg: {len(db_tg)}, Tensile: {len(db_tensile)}")

# ============================================================
# FIGURE 1: Chemical Space PCA
# ============================================================
print("\n[1/3] Chemical space PCA...")

# Select formula features for PCA
pca_features = [
    'Flame_retardant_AdditionAmount(wt%)', 'Curing_agent_AdditionAmount(wt%)',
    'EEW', 'T_max', 't_total', 'Q_thermal', 'EP_wt_fraction'
]
# Add a few descriptor features that exist in both
desc_cols = [c for c in df_db.columns if c.startswith(('FR_', 'EP_', 'CURING_')) and c in df_exp.columns]
# Use first 20 descriptor columns for PCA
use_desc = desc_cols[:20]

# Build PCA matrix
X_db_list = []
for col in pca_features + use_desc:
    if col in df_db.columns:
        X_db_list.append(pd.to_numeric(df_db[col], errors='coerce').fillna(0).values)
X_db_pca = np.column_stack(X_db_list)

X_exp_list = []
for col in pca_features + use_desc:
    if col in df_exp.columns:
        X_exp_list.append(pd.to_numeric(df_exp[col], errors='coerce').fillna(0).values)
X_exp_pca = np.column_stack(X_exp_list)

# PCA
scaler_pca = StandardScaler()
X_all = np.vstack([X_db_pca, X_exp_pca])
X_all_s = scaler_pca.fit_transform(X_all)
pca = PCA(n_components=2)
X_all_pca = pca.fit_transform(X_all_s)
n_db = len(X_db_pca)
X_db_2d = X_all_pca[:n_db]
X_exp_2d = X_all_pca[n_db:]

fig1, ax1 = plt.subplots(figsize=(12, 10), dpi=300)
ax1.scatter(X_db_2d[:, 0], X_db_2d[:, 1], c=GRAY, s=8, alpha=0.4, label='Database (N={:d})'.format(n_db), rasterized=True)
ax1.scatter(X_exp_2d[:, 0], X_exp_2d[:, 1], c=RED, s=120, marker='*', zorder=10,
           edgecolors='darkred', linewidth=1.0, label=f'External (N={len(df_exp)})')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
ax1.legend(fontsize=16, loc='upper right')
style(ax1)
fig1.tight_layout()
fig1.savefig(OUT_DIR / 'Figure_OOD_chemical_space.png', dpi=600, bbox_inches='tight')
fig1.savefig(OUT_DIR / 'Figure_OOD_chemical_space.pdf', bbox_inches='tight')
plt.close(fig1)
print(f'  PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}')

# ============================================================
# FIGURE 2: Formula/Process Distribution Comparison
# ============================================================
print("[2/3] Formula distribution comparison...")

fig2, axes2 = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
formula_vars = [
    ('Flame_retardant_AdditionAmount(wt%)', 'FR wt%', EXPLICIT),
    ('Curing_agent_AdditionAmount(wt%)', 'Curing wt%', RESIN),
    ('EEW', 'EEW', FR),
    ('T_max', 'T_max (°C)', GRAPH),
    ('t_total', 't_total (h)', RED),
    ('Q_thermal', 'Q_thermal', BLUE),
]

for idx, (col, label, color) in enumerate(formula_vars):
    ax = axes2[idx // 3, idx % 3]
    panel_label(ax, chr(ord('a') + idx))
    
    db_vals = pd.to_numeric(df_db[col], errors='coerce').dropna().values
    exp_vals = pd.to_numeric(df_exp[col], errors='coerce').dropna().values
    
    # Violin plot data
    data = [db_vals, exp_vals]
    positions = [1, 2]
    
    vp = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.6)
    vp['bodies'][0].set_facecolor(GRAY); vp['bodies'][0].set_alpha(0.4)
    vp['bodies'][1].set_facecolor(RED); vp['bodies'][1].set_alpha(0.7)
    for part in ['cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars']:
        if part in vp: vp[part].set_color(INK)
    
    # Jitter dots on Exp violin
    if len(exp_vals) > 0:
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(exp_vals))
        ax.scatter(np.full(len(exp_vals), 2) + jitter, exp_vals, c=RED, s=25, alpha=0.7,
                  zorder=5, edgecolors='darkred', linewidth=0.3)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'DB\n(N={len(db_vals)})', f'Exp\n(N={len(exp_vals)})'], fontsize=14, fontweight='bold')
    ax.set_ylabel(label, fontweight='bold')
    style(ax)

fig2.tight_layout()
fig2.savefig(OUT_DIR / 'Figure_OOD_formula_distribution.png', dpi=600, bbox_inches='tight')
fig2.savefig(OUT_DIR / 'Figure_OOD_formula_distribution.pdf', bbox_inches='tight')
plt.close(fig2)

# ============================================================
# FIGURE 3: Target Property Range Comparison
# ============================================================
print("[3/3] Target property range comparison...")

# ── UL94 from database ──
with open(HERE.parent / '94/Result_Final_TabICL_Interpret_Cls/saved_data/plot_data.pkl', 'rb') as f:
    data_ul = pkl.load(f)
db_ul94 = np.concatenate([data_ul['y_train'], data_ul['y_test']])

fig3, axes3 = plt.subplots(1, 4, figsize=(26, 8), dpi=300)
target_vars = [
    ('LOI', db_loi, exp_loi, 'LOI (%)', EXPLICIT),
    ('UL-94', db_ul94, pd.to_numeric(actual[94].map({'V0':1,'V-0':1,'NR':0,'V-2':0}), errors='coerce').dropna().values, 'UL-94 (V-0 rate)', GRAPH),
    ('Tg', db_tg, exp_tg, 'Tg (°C)', RESIN),
    ('Tensile', db_tensile, exp_tensile, 'Tensile (MPa)', FR),
]

for idx, (name, db_vals, exp_vals, label, color) in enumerate(target_vars):
    ax = axes3[idx]
    panel_label(ax, chr(ord('a') + idx))
    
    if name == 'UL-94':
        # 100% stacked bar
        db_v0_pct = (db_vals == 1).sum() / len(db_vals) * 100
        db_fail_pct = 100 - db_v0_pct
        exp_v0_pct = (exp_vals == 1).sum() / max(len(exp_vals), 1) * 100
        exp_fail_pct = 100 - exp_v0_pct
        
        x_pos = np.array([1, 2])
        w = 0.5
        ax.bar(x_pos, [db_fail_pct, exp_fail_pct], w, color=GRAY, alpha=0.5, edgecolor='white', label='Fail')
        ax.bar(x_pos, [db_v0_pct, exp_v0_pct], w, bottom=[db_fail_pct, exp_fail_pct],
               color=color, alpha=0.9, edgecolor='white', label='V-0')
        
        for i, (pos, v0, fail) in enumerate(zip(x_pos, [db_v0_pct, exp_v0_pct], [db_fail_pct, exp_fail_pct])):
            ax.text(pos, fail/2, f'{fail:.0f}%', ha='center', va='center', fontsize=14, fontweight='bold', color=INK)
            ax.text(pos, fail + v0/2, f'{v0:.0f}%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Database\n(N={len(db_vals)})', f'External\n(N={len(exp_vals)})'], fontsize=14, fontweight='bold')
        ax.set_ylabel('Proportion (%)', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(fontsize=14, loc='upper right')
    else:
        bp = ax.boxplot([db_vals, exp_vals], positions=[1, 2], widths=0.5,
                        patch_artist=True, medianprops={'color': INK, 'linewidth': 2},
                        whiskerprops={'linewidth': 1.5}, capprops={'linewidth': 1.5})
        bp['boxes'][0].set_facecolor(GRAY); bp['boxes'][0].set_alpha(0.4)
        bp['boxes'][1].set_facecolor(color); bp['boxes'][1].set_alpha(0.7)
        
        for i, (vals, pos, c) in enumerate([(db_vals, 1, GRAY), (exp_vals, 2, color)]):
            if len(vals) > 0:
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, c=c, s=25, alpha=0.5, zorder=5)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Database\n(N={len(db_vals)})', f'External\n(N={len(exp_vals)})'],
                           fontsize=14, fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
    
    style(ax)

fig3.tight_layout()
fig3.savefig(OUT_DIR / 'Figure_OOD_target_range.png', dpi=600, bbox_inches='tight')
fig3.savefig(OUT_DIR / 'Figure_OOD_target_range.pdf', bbox_inches='tight')
plt.close(fig3)

print("\nAll figures saved to translated_text2/")
