import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib as mpl

ROOT_DIR = r'C:\Users\WINDOWS\Desktop\GNN'
TARGETS = ['LOI', 'Tg', 'TENSILE']
UNITS = {'LOI': '(%)', 'Tg': '(°C)', 'TENSILE': '(MPa)'}

OUTPUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_text-translated_text_translated_text'
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_ORANGE = '#D17758'   # translated_text - translated_text
C_NAVY   = '#344660'   # translated_text - translated_text
C_GREEN  = '#889A74'   # translated_text - Tg
C_GREY   = '#DDE0E7'   # translated_text - translated_text/translated_text

PALETTES = {
    'LOI':     {'primary': C_ORANGE},   # translated_text
    'Tg':      {'primary': C_GREEN},    # translated_text
    'TENSILE': {'primary': C_NAVY}      # translated_text
}

mm_to_inch = 1 / 25.4
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['savefig.dpi'] = 600

for target in TARGETS:
    print(f"\n{'='*50}\n>>> translated_text: {target} translated_text\n{'='*50}")
    
    target_dir = os.path.join(ROOT_DIR, target)
    if not os.path.exists(target_dir):
        print(f"[translated_text] translated_text: {target_dir}")
        continue
        
    result_dirs = glob.glob(os.path.join(target_dir, "Result_NoGNN_5Fold_*"))
    if not result_dirs:
        print(f"[translated_text] translated_text {target} translated_text。")
        continue
        
    latest_dir = max(result_dirs, key=os.path.getmtime)
    print(f"📁 translated_text: {os.path.basename(latest_dir)}")
    
    try:
        df_summary = pd.read_csv(os.path.join(latest_dir, 'CV_Statistical_Summary_NoGNN.csv'))
        df_folds = pd.read_csv(os.path.join(latest_dir, 'CV_All_Folds_Results_NoGNN.csv'))
        df_shap = pd.read_csv(os.path.join(latest_dir, 'Global_Top20_Features.csv'))
    except FileNotFoundError as e:
        print(f"[translated_text] translated_text: {e.filename}")
        continue

    df_summary = df_summary[df_summary['Mean_Test_R2'] > 0]
    top_models = df_summary.head(10)['Model'].tolist()
    df_folds_top = df_folds[df_folds['Model'].isin(top_models)]
    
    best_model_name = top_models[0]
    df_oof = pd.read_csv(os.path.join(latest_dir, f'OOF_Predictions_{best_model_name}.csv'))
    
    c_primary = PALETTES[target]['primary']
    unit = UNITS[target]
    
    if target == 'LOI':
        cmap_custom = mpl.colors.LinearSegmentedColormap.from_list("cmap", ["#FDF3F0", C_ORANGE])
    elif target == 'Tg':
        cmap_custom = mpl.colors.LinearSegmentedColormap.from_list("cmap", ["#F4F6F2", C_GREEN])
    else:  # TENSILE
        cmap_custom = mpl.colors.LinearSegmentedColormap.from_list("cmap", ["#E8EBF0", C_NAVY])

    # ------------------------------------------
    # ------------------------------------------
    figA, axA = plt.subplots(figsize=(8, 6))
    colors_A = [c_primary] + [C_GREY] * 9
    sns.barplot(data=df_summary.head(10), x='Mean_Test_R2', y='Model',
                palette=colors_A, ax=axA, edgecolor='black', linewidth=1.0, alpha=0.85)
    
    for i, model in enumerate(df_summary.head(10)['Model']):
        mean_val = df_summary.iloc[i]['Mean_Test_R2']
        std_val = df_summary.iloc[i]['Std_Test_R2']
        axA.errorbar(mean_val, i, xerr=std_val, color='black', capsize=4, capthick=1.2, elinewidth=1.2)
        axA.text(mean_val + std_val + 0.02, i, f"{mean_val:.3f}", va='center', fontsize=10, fontweight='bold')

    axA.set_xlabel('Mean R² Score (5-Fold CV)', fontweight='bold')
    axA.set_ylabel('')
    max_x_needed = (df_summary.head(10)['Mean_Test_R2'] + df_summary.head(10)['Std_Test_R2']).max() + 0.12
    axA.set_xlim(0, min(1.05, max_x_needed))
    axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)
    
    figA.tight_layout()
    figA.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigA_Ranking.pdf'))
    figA.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigA_Ranking.png'), dpi=600)
    plt.close(figA)

    # ------------------------------------------
    # ------------------------------------------
    figB, axB = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_folds_top, x='Test_R2', y='Model', order=top_models,
                color='white', ax=axB, width=0.6,
                boxprops={'linewidth':1.5},
                whiskerprops={'color':'black'}, medianprops={'color':'black'})
    for i, patch in enumerate(axB.patches):
        patch.set_edgecolor(c_primary if i == 0 else C_GREY)
    
    for i, model in enumerate(top_models):
        subset = df_folds_top[df_folds_top['Model'] == model]
        jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
        color = c_primary if i == 0 else C_GREY
        axB.scatter(subset['Test_R2'], np.full_like(subset['Test_R2'], i) + jitter,
                    color=color, s=36, alpha=0.7, edgecolor='none')

    axB.set_xlabel('R² Score per Fold', fontweight='bold')
    axB.set_ylabel('')
    axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)

    figB.tight_layout()
    figB.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigB_Stability.pdf'))
    figB.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigB_Stability.png'), dpi=600)
    plt.close(figB)

    # ------------------------------------------
    # ------------------------------------------
    figC, axC = plt.subplots(figsize=(8, 6))
    top_15_shap = df_shap.head(15).copy()
    top_15_shap['Display_Name'] = top_15_shap['Feature_Name'].apply(lambda x: x[:30]+'...' if len(x)>30 else x)
    
    shap_vals = top_15_shap['SHAP_Importance'].values
    norm = plt.Normalize(shap_vals.min(), shap_vals.max())
    cmap_shap = mpl.colors.LinearSegmentedColormap.from_list("shap_gradient", ["#FFFFFF", c_primary])
    colors_C = [cmap_shap(norm(val)) for val in shap_vals]

    sns.barplot(data=top_15_shap, x='SHAP_Importance', y='Display_Name',
                palette=colors_C, ax=axC, edgecolor='black', linewidth=1.0, alpha=0.85)
    axC.set_xlabel(f'Mean |SHAP Value| (Impact on {target})', fontweight='bold')
    axC.set_ylabel('')
    axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)

    figC.tight_layout()
    figC.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigC_SHAP.pdf'))
    figC.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigC_SHAP.png'), dpi=600)
    plt.close(figC)

    # ------------------------------------------
    # ------------------------------------------
    figD, axD = plt.subplots(figsize=(6.5, 6.5))
    actual_col = [c for c in df_oof.columns if 'Actual' in c][0]
    pred_col = [c for c in df_oof.columns if 'Predicted' in c][0]
    
    y_actual, y_pred = df_oof[actual_col].values, df_oof[pred_col].values
    r2_best = r2_score(y_actual, y_pred)
    rmse_best = np.sqrt(mean_squared_error(y_actual, y_pred))

    xy = np.vstack([y_actual, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_actual, y_pred, z = y_actual[idx], y_pred[idx], z[idx]

    axD.scatter(y_actual, y_pred, c=z, s=40, cmap=cmap_custom, edgecolor='white', linewidth=0.3, alpha=0.9)

    min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val)
    axD.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin],
             color=C_GREY, linestyle='--', lw=1.5)
    
    axD.fill_between([min_val-margin, max_val+margin],
                     [(min_val-margin)*0.9, (max_val+margin)*0.9],
                     [(min_val-margin)*1.1, (max_val+margin)*1.1],
                     color='gray', alpha=0.15, label='±10% Error Band')

    axD.set_xlim(min_val-margin, max_val+margin)
    axD.set_ylim(min_val-margin, max_val+margin)
    axD.set_xlabel(f'Actual {target} {unit}', fontweight='bold')
    axD.set_ylabel(f'Predicted {target} {unit} ({best_model_name})', fontweight='bold')

    textstr = f'$R^2$ = {r2_best:.4f}\nRMSE = {rmse_best:.4f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    axD.text(0.05, 0.95, textstr, transform=axD.transAxes, fontsize=12,
             verticalalignment='top', bbox=props, fontweight='bold')
    axD.spines['top'].set_visible(False); axD.spines['right'].set_visible(False)
    axD.legend(loc='lower right', frameon=False)

    figD.tight_layout()
    figD.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigD_BestModel.pdf'))
    figD.savefig(os.path.join(OUTPUT_DIR, f'{target}_Benchmark_FigD_BestModel.png'), dpi=600)
    plt.close(figD)
    
    print(f"✅ [{target}] translated_text！translated_text {OUTPUT_DIR}")

print("\n🎉 translated_text！")