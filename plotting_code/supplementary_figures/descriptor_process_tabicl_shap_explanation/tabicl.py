import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.calibration import calibration_curve
from scipy.stats import gaussian_kde  # translated_text：translated_text KDE
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

GLOBAL_OUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_text-TABICL-SHAPtranslated_text'

TARGETS = {
    'LOI': {'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data', 'color': '#D55E00', 'color_neg': '#8491B4', 'task': 'reg'},
    'UL94': {'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 'color': '#D55E00', 'color_neg': '#8491B4', 'task': 'cls'},
    'Tg': {'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data', 'color': '#332288', 'color_neg': '#91D1C2', 'task': 'reg'},
    'TENSILE': {'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data', 'color': '#0072B2', 'color_neg': '#F39B7F', 'task': 'reg'}
}

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 1.2, 'axes.labelsize': 12, 'axes.titlesize': 12,
})

def format_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # translated_text

def wrap_labels(label, max_len=25):
    return label[:max_len-3] + '...' if len(label) > max_len else label

print("="*70)
print(">>> 🚀 translated_text: translated_text、translated_text、translated_text")
print("="*70)

for target_key, config in TARGETS.items():
    data_dir = config['path']
    c_primary = config['color']
    c_neg = config['color_neg']
    task = config['task']
    t_name = target_key
    
    pkl_file = os.path.join(data_dir, 'plot_data.pkl')
    if not os.path.exists(pkl_file): continue
        
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    X_test, y_test = data['X_test'], data['y_test']
    shap_values = data['shap_values']
    top_k_features = data['top_k_features']
    preds = data['preds']
    
    df_test_X = pd.DataFrame(X_test, columns=top_k_features)
    
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top_10_idx = np.argsort(shap_abs_mean)[-10:][::-1] # translated_text10translated_text
    top_10_features = [top_k_features[i] for i in top_10_idx]

    print(f"\n--- translated_text {t_name} ---")

    # ------------------------------------------
    # ------------------------------------------
    print(f"[{t_name}] translated_text SHAP translated_text...")
    fig_ridge, ax_ridge = plt.subplots(figsize=(6, 8), dpi=300)
    
    y_offsets = np.arange(10)[::-1] * 1.0 
    
    global_min = shap_values[:, top_10_idx].min()
    global_max = shap_values[:, top_10_idx].max()
    margin = (global_max - global_min) * 0.1
    x_grid = np.linspace(global_min - margin, global_max + margin, 500)
    
    ax_ridge.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    
    for idx, feature_name in enumerate(top_10_features):
        feat_idx = top_k_features.index(feature_name)
        feat_shap = shap_values[:, feat_idx]
        
        try:
            kde = gaussian_kde(feat_shap)
            y_dens = kde(x_grid)
            if y_dens.max() > 0:
                y_dens = (y_dens / y_dens.max()) * 0.85
        except:
            y_dens = np.zeros_like(x_grid)
        
        y_plot = y_dens + y_offsets[idx]
        
        ax_ridge.plot(x_grid, y_plot, color=c_primary, lw=1.5, zorder=idx+1)
        ax_ridge.fill_between(x_grid, y_offsets[idx], y_plot, color=c_primary, alpha=0.6, zorder=idx+1)
        
        ax_ridge.text(global_min - margin, y_offsets[idx] + 0.1, 
                      wrap_labels(feature_name), fontweight='bold', fontsize=10, ha='right', va='bottom')

    ax_ridge.set_yticks([])
    ax_ridge.set_ylabel('Density of SHAP Impact', fontweight='bold', labelpad=20)
    ax_ridge.set_xlabel(f'SHAP Value (Impact on {t_name})', fontweight='bold')
    ax_ridge.set_title(f'SHAP Impact Ridgeline ({t_name})', fontweight='bold', pad=15)
    
    ax_ridge.set_xlim(global_min - margin - (global_max-global_min)*0.4, global_max + margin)
    format_spines(ax_ridge)
    
    fig_ridge.tight_layout()
    fig_ridge.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_13_SHAP_Ridgeline.pdf'), bbox_inches='tight')
    fig_ridge.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_13_SHAP_Ridgeline.png'), dpi=600, bbox_inches='tight')
    plt.close(fig_ridge)

    # ------------------------------------------
    # ------------------------------------------
    if task == 'cls':
        print(f"[{t_name}] translated_text...")
        probs = data['probs']
        
        fig_cal, ax_cal = plt.subplots(figsize=(6, 6), dpi=300)
        
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy='uniform')
        
        ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        ax_cal.plot(prob_pred, prob_true, marker='o', color=c_primary, linewidth=2, markersize=8, label=f'TabICL Model')
        
        ax_cal.set_xlabel('Mean Predicted Probability (Confidence)', fontweight='bold')
        ax_cal.set_ylabel('Fraction of Positives (Actual Rate)', fontweight='bold')
        ax_cal.set_title('Reliability & Calibration Diagram', fontweight='bold')
        ax_cal.legend(loc='lower right', frameon=False)
        format_spines(ax_cal)
        
        fig_cal.tight_layout()
        fig_cal.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_14_Calibration_Curve.pdf'), bbox_inches='tight')
        fig_cal.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_14_Calibration_Curve.png'), dpi=600, bbox_inches='tight')
        plt.close(fig_cal)

    # ------------------------------------------
    # ------------------------------------------
    if task == 'reg':
        print(f"[{t_name}] translated_text...")
        pred_df = pd.read_csv(os.path.join(data_dir, 'predictions.csv'))
        residuals = pred_df['Residual'].values
        
        main_feature = top_10_features[0]
        x_vals = df_test_X[main_feature].values
        
        fig_res, ax_res = plt.subplots(figsize=(7, 6), dpi=300)
        
        scatter = ax_res.scatter(x_vals, residuals, s=(y_test - y_test.min() + 1)**1.5 * 2, 
                                 c=np.abs(residuals), cmap='Reds', alpha=0.7, edgecolor='black', linewidth=0.8)
        
        ax_res.axhline(0, color='gray', linestyle='--', lw=2)
        
        sns.regplot(x=x_vals, y=residuals, scatter=False, lowess=True, ax=ax_res, color=c_neg, line_kws={'linestyle':'-.', 'lw':2})
        
        cbar = plt.colorbar(scatter, ax=ax_res)
        cbar.set_label('Absolute Error Magnitude', fontweight='bold')
        
        ax_res.set_xlabel(f'{wrap_labels(main_feature)} (Top Prior)', fontweight='bold')
        ax_res.set_ylabel(f'Prediction Residual (Predicted - Actual)', fontweight='bold')
        ax_res.set_title(f'Residual Diagnosis vs Core Mechanism ({t_name})', fontweight='bold')
        format_spines(ax_res)
        
        fig_res.tight_layout()
        fig_res.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_15_Residual_Diagnosis.pdf'), bbox_inches='tight')
        fig_res.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_15_Residual_Diagnosis.png'), dpi=600, bbox_inches='tight')
        plt.close(fig_res)

print("\n🎉 translated_text！translated_text，translated_text 15 translated_text！")