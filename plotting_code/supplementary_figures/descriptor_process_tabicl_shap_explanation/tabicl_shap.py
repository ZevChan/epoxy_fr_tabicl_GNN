import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import shap
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from scipy import stats
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

GLOBAL_OUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_text-TABICL-SHAPtranslated_text'
os.makedirs(GLOBAL_OUT_DIR, exist_ok=True)

TARGETS = {
    'LOI': {
        'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data',
        'color': '#D55E00',       # translated_text
        'color_neg': '#8491B4',   # NPG translated_text (translated_text)
        'target_name': 'LOI',
        'task': 'reg'
    },
    'UL94': {
        'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data',
        'color': '#D55E00',       
        'color_neg': '#8491B4',
        'target_name': 'UL-94',
        'task': 'cls'
    },
    'Tg': {
        'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data',
        'color': '#332288',       # translated_text
        'color_neg': '#91D1C2',   # NPG translated_text (translated_text)
        'target_name': 'Tg',
        'task': 'reg'
    },
    'TENSILE': {
        'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data',
        'color': '#0072B2',       # translated_text
        'color_neg': '#F39B7F',   # NPG translated_text (translated_text)
        'target_name': 'TENSILE',
        'task': 'reg'
    }
}

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10
})

def format_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def wrap_labels(label, max_len=30):
    return label[:max_len-3] + '...' if len(label) > max_len else label

def recolor_shap_waterfall(ax, pos_color, neg_color):
    """translated_text SHAP translated_text"""
    for patch in ax.patches:
        facecolor = patch.get_facecolor()
        if facecolor[0] > 0.5 and facecolor[2] < 0.5:
            patch.set_facecolor(pos_color); patch.set_edgecolor(pos_color)
        elif facecolor[2] > 0.5 and facecolor[0] < 0.5:
            patch.set_facecolor(neg_color); patch.set_edgecolor(neg_color)
    for text in ax.texts:
        color = text.get_color()
        if color == '#ff0052': text.set_color(pos_color)
        elif color == '#008bfb': text.set_color(neg_color)

def corrfunc(x, y, **kwargs):
    cmap = kwargs.pop('custom_cmap')
    kwargs.pop('color', None) # translated_text seaborn translated_text
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2: return
    r, p = stats.pearsonr(x, y)
    stars = "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
    ax = plt.gca()
    color = cmap((r + 1) / 2)
    ax.set_facecolor(color)
    ax.patch.set_alpha(0.7)
    ax.annotate(f"{r:.2f}\n{stars}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold', color='black')

def scatter_reg(x, y, **kwargs):
    scatter_color = kwargs.pop('scatter_color')
    line_color = kwargs.pop('line_color')
    kwargs.pop('color', None)
    ax = plt.gca()
    sns.regplot(x=x, y=y, ax=ax, 
                scatter_kws={'s': 15, 'alpha': 0.7, 'color': scatter_color, 'edgecolors': 'white', 'linewidths': 0.5},
                line_kws={'color': line_color, 'linewidth': 2})
    ax.grid(True, linestyle='--', alpha=0.3)

def dist_plot(x, **kwargs):
    hist_color = kwargs.pop('hist_color')
    kwargs.pop('color', None)
    ax = plt.gca()
    sns.histplot(x, ax=ax, kde=True, color=hist_color, edgecolor='black', alpha=0.6, bins=20)
    ax.grid(True, linestyle='--', alpha=0.3)

print("="*70)
print(">>> 🚀 translated_text TabICL translated_text")
print(f">>> 📂 translated_text: {GLOBAL_OUT_DIR}")
print("="*70)

for target_key, config in TARGETS.items():
    data_dir = config['path']
    c_primary = config['color']
    c_neg = config['color_neg']
    t_name = config['target_name']
    task = config['task']
    
    if not os.path.exists(data_dir):
        print(f"[translated_text] translated_text: {data_dir}")
        continue
    
    cmap_custom = mcolors.LinearSegmentedColormap.from_list(f"cmap_{target_key}", ["#F2F2F2", c_primary])
    cmap_diverging = mcolors.LinearSegmentedColormap.from_list(f"cmap_div_{target_key}", [c_neg, "#F2F2F2", c_primary])
    
    pkl_file = os.path.join(data_dir, 'plot_data.pkl')
    if not os.path.exists(pkl_file):
        continue
        
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    X_test, y_test = data['X_test'], data['y_test']
    X_train, y_train = data['X_train'], data['y_train']
    shap_values = data['shap_values']
    top_k_features = data['top_k_features']
    top_6_features = data['top_6_features']
    preds = data['preds']
    
    df_test_X = pd.DataFrame(X_test, columns=top_k_features)
    
    if task == 'cls':
        probs = data['probs']
        expected_value = data['expected_value']
    else:
        pred_df = pd.read_csv(os.path.join(data_dir, 'predictions.csv'))
        residuals = pred_df['Residual'].values
        inferred_base = preds - np.sum(shap_values, axis=1)
        expected_value = np.mean(inferred_base)

    print(f"\n--- translated_text {t_name} ---")

    # ------------------------------------------
    # ------------------------------------------
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top_15_idx = np.argsort(shap_abs_mean)[-15:]
    top_15_features_sorted = [wrap_labels(top_k_features[i]) for i in top_15_idx]
    values_for_bars = shap_abs_mean[top_15_idx]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.barh(top_15_features_sorted, values_for_bars, color=c_primary, edgecolor='black', linewidth=1.0)
    ax.set_xlabel(f'Mean |SHAP Value| (Impact on {t_name})', fontweight='bold', labelpad=10)
    format_spines(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_1_SHAP_Bar.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_1_SHAP_Bar.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------
    # ------------------------------------------
    for feature_name in top_6_features[:5]:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        shap.dependence_plot(
            feature_name, shap_values, df_test_X, 
            ax=ax, show=False, interaction_index=None, 
            color=c_primary, alpha=0.8, dot_size=30
        )
        ax.set_ylabel(f'SHAP Value', fontweight='bold')
        ax.set_title(f"PDP: {feature_name}", fontweight='bold')
        format_spines(ax)
        fig.tight_layout()
        safe_name = feature_name.replace('/', '_').replace('\\', '_')
        fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_2_PDP_{safe_name}.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_2_PDP_{safe_name}.png'), dpi=600, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------
    # ------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    shap_pca = pca.fit_transform(shap_values)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    if task == 'cls':
        ax.scatter(shap_pca[y_test==0, 0], shap_pca[y_test==0, 1], color=c_neg, label='Fail (0)', alpha=0.8, s=30)
        ax.scatter(shap_pca[y_test==1, 0], shap_pca[y_test==1, 1], color=c_primary, label='Pass (1)', alpha=0.8, s=30)
        ax.legend(frameon=False)
    else:
        sc = ax.scatter(shap_pca[:, 0], shap_pca[:, 1], c=y_test, cmap=cmap_custom, alpha=0.9, edgecolors='k', linewidth=0.5, s=40)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f'Actual {t_name}', fontweight='bold')

    ax.set_xlabel(f'SHAP PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'SHAP PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax.set_title('Mechanistic Clustering (SHAP Space)', fontweight='bold')
    format_spines(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_3_SHAP_PCA.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_3_SHAP_PCA.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------
    # ------------------------------------------
    print(f"   └─ translated_text Waterfall translated_text (Top 3)...")
    if task == 'reg':
        idx_target = {
            "Best_Prediction": np.argsort(np.abs(residuals))[:3],
            "Worst_Prediction": np.argsort(np.abs(residuals))[-3:][::-1],
            f"Highest_{t_name}": np.argsort(y_test)[-3:][::-1]
        }
    else:
        idx_tp = np.where((y_test == 1) & (preds == 1))[0]
        idx_fp = np.where((y_test == 0) & (preds == 1))[0]
        tp_sorted = idx_tp[np.argsort(probs[idx_tp])[::-1]] if len(idx_tp) > 0 else []
        fp_sorted = idx_fp[np.argsort(probs[idx_fp])[::-1]] if len(idx_fp) > 0 else []
        
        idx_target = {}
        if len(tp_sorted) > 0: idx_target["True_Positive"] = tp_sorted[:3]
        if len(fp_sorted) > 0: idx_target["False_Positive"] = fp_sorted[:3]

    for label, indices in idx_target.items():
        for rank, idx in enumerate(indices):
            fig = plt.figure(figsize=(7, 5), dpi=300)
            exp = shap.Explanation(
                values=shap_values[idx], base_values=expected_value, 
                data=X_test[idx], feature_names=top_k_features
            )
            shap.plots.waterfall(exp, max_display=8, show=False)
            recolor_shap_waterfall(plt.gca(), pos_color=c_primary, neg_color=c_neg)
            
            act_val = y_test[idx]
            pred_val = probs[idx] if task == 'cls' else preds[idx]
            plt.title(f"{label.replace('_', ' ')} (Top {rank+1})\n(Actual: {act_val:.2f}, Pred: {pred_val:.2f})", fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_4_Waterfall_{label}_Top{rank+1}.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_4_Waterfall_{label}_Top{rank+1}.png'), dpi=600, bbox_inches='tight')
            plt.close(fig)

    # ------------------------------------------
    # ------------------------------------------
    print(f"   └─ translated_text (translated_text PairGrid)...")
    top_5_features = top_6_features[:5]
    top_5_indices = [top_k_features.index(f) for f in top_5_features]
    
    df_pg = pd.DataFrame(X_train[:, top_5_indices], columns=top_5_features)
    df_pg[t_name] = y_train
    df_pg.columns = [wrap_labels(c, 18) for c in df_pg.columns]
    
    g = sns.PairGrid(df_pg, height=1.8, aspect=1)
    
    g.map_upper(corrfunc, custom_cmap=cmap_diverging)
    g.map_lower(scatter_reg, scatter_color=c_primary, line_color=c_neg)
    g.map_diag(dist_plot, hist_color=c_primary)
    
    for ax in g.axes.flatten():
        if ax is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            
    fig_pg = g.fig
    fig_pg.subplots_adjust(right=0.9)
    cbar_ax = fig_pg.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_diverging), cax=cbar_ax)
    cb.set_label('Pearson Correlation (r)', fontsize=12, fontweight='bold', labelpad=10)
    cb.ax.tick_params(labelsize=10)
    
    fig_pg.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_5_PairGrid.pdf'), bbox_inches='tight')
    fig_pg.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_5_PairGrid.png'), dpi=600, bbox_inches='tight')
    plt.close(fig_pg)
    
    # ------------------------------------------
    # ------------------------------------------
    print(f"   └─ translated_text SHAP Beeswarm translated_text (translated_text)...")
    fig_bee = plt.figure(figsize=(7, 6), dpi=300)
    shap.summary_plot(shap_values, df_test_X, max_display=15, show=False, cmap=cmap_diverging)
    plt.title(f"SHAP Summary ({t_name})", fontweight='bold', pad=15)
    
    fig_bee.tight_layout()
    fig_bee.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_6_SHAP_Beeswarm.pdf'), bbox_inches='tight')
    fig_bee.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_6_SHAP_Beeswarm.png'), dpi=600, bbox_inches='tight')
    plt.close(fig_bee)

    # ------------------------------------------
    # ------------------------------------------
    if task == 'reg':
        print(f"   └─ translated_text Actual vs Predicted Hexbin translated_text...")
        g_hex = sns.jointplot(
            x=preds, y=y_test,
            kind="hex", color=c_primary, 
            gridsize=20, marginal_kws=dict(bins=25, fill=True, color=c_primary)
        )
        ax_joint = g_hex.ax_joint
        ax_joint.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                      'k--', linewidth=2, label='Perfect Prediction')
        
        ax_joint.set_xlabel(f'Predicted {t_name}', fontweight='bold')
        ax_joint.set_ylabel(f'Actual {t_name}', fontweight='bold')
        ax_joint.legend(loc='upper left', frameon=False)
        g_hex.fig.suptitle(f"{t_name} Density Space", fontweight='bold', y=1.02)
        
        g_hex.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_7_Hexbin_Density.pdf'), bbox_inches='tight')
        g_hex.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_7_Hexbin_Density.png'), dpi=600, bbox_inches='tight')
        plt.close(g_hex.fig)

    # ------------------------------------------
    # ------------------------------------------
    print(f"   └─ translated_text 2D PDP translated_text (Top 1 vs Top 2~5)...")
    if len(top_6_features) >= 5:
        main_feature = top_6_features[0] 
        for secondary_feature in top_6_features[1:5]:
            fig_2d, ax_2d = plt.subplots(figsize=(7, 5), dpi=300)
            
            x_2d = df_test_X[main_feature].values
            y_2d = df_test_X[secondary_feature].values
            
            feature_idx = list(df_test_X.columns).index(main_feature)
            z_2d = shap_values[:, feature_idx]

            xi = np.linspace(x_2d.min(), x_2d.max(), 100)
            yi = np.linspace(y_2d.min(), y_2d.max(), 100)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x_2d, y_2d), z_2d, (xi, yi), method='linear')

            contour = ax_2d.contourf(xi, yi, zi, levels=20, cmap=cmap_custom, alpha=0.85)
            ax_2d.contour(xi, yi, zi, levels=10, colors='black', linewidths=0.5, alpha=0.5, linestyles='dashed')
            
            cbar = plt.colorbar(contour, ax=ax_2d)
            cbar.set_label(f'SHAP Value ({wrap_labels(main_feature, 15)})', fontweight='bold')
            
            ax_2d.set_xlabel(wrap_labels(main_feature, 25), fontweight='bold')
            ax_2d.set_ylabel(wrap_labels(secondary_feature, 25), fontweight='bold')
            ax_2d.set_title('2D Feature Synergy / Interaction', fontweight='bold')
            format_spines(ax_2d)
            
            fig_2d.tight_layout()
            safe_name1 = main_feature.replace('/', '_').replace('\\', '_')
            safe_name2 = secondary_feature.replace('/', '_').replace('\\', '_')
            
            fig_2d.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_8_2D_PDP_{safe_name1}_vs_{safe_name2}.pdf'), bbox_inches='tight')
            fig_2d.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_8_2D_PDP_{safe_name1}_vs_{safe_name2}.png'), dpi=600, bbox_inches='tight')
            plt.close(fig_2d)

print("\n🎉 translated_text！translated_text，translated_text！")