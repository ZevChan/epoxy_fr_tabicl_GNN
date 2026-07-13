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

PALETTES = {
    'LOI':     {'primary': '#D55E00', 'cmap': 'Oranges_r'}, # translated_text：translated_text
    'Tg':      {'primary': '#332288', 'cmap': 'Purples_r'}, # translated_text：translated_text
    'TENSILE': {'primary': '#0072B2', 'cmap': 'Blues_r'}    # translated_text：translated_text
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
    c_cmap = PALETTES[target]['cmap']
    unit = UNITS[target]

    # ------------------------------------------
    # ------------------------------------------
    figA, axA = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_summary.head(10), x='Mean_Test_R2', y='Model', color=c_primary, ax=axA, edgecolor='black', linewidth=1.0, alpha=0.85)
    
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
    figA.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigA_Ranking.pdf'))
    figA.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigA_Ranking.png'), dpi=600)
    plt.close(figA)

    # ------------------------------------------
    # ------------------------------------------
    figB, axB = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_folds_top, x='Test_R2', y='Model', order=top_models, color='white', ax=axB, 
                width=0.6, boxprops={'edgecolor':c_primary, 'linewidth':1.5}, whiskerprops={'color':'black'}, medianprops={'color':'black'})
    sns.stripplot(data=df_folds_top, x='Test_R2', y='Model', order=top_models, color=c_primary, size=6, alpha=0.7, ax=axB, jitter=True)

    axB.set_xlabel('R² Score per Fold', fontweight='bold')
    axB.set_ylabel('')
    axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)

    figB.tight_layout()
    figB.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigB_Stability.pdf'))
    figB.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigB_Stability.png'), dpi=600)
    plt.close(figB)

    # ------------------------------------------
    # ------------------------------------------
    figC, axC = plt.subplots(figsize=(8, 6))
    top_15_shap = df_shap.head(15).copy()
    top_15_shap['Display_Name'] = top_15_shap['Feature_Name'].apply(lambda x: x[:30]+'...' if len(x)>30 else x)

    sns.barplot(data=top_15_shap, x='SHAP_Importance', y='Display_Name', color=c_primary, ax=axC, edgecolor='black', linewidth=1.0, alpha=0.85)
    axC.set_xlabel(f'Mean |SHAP Value| (Impact on {target})', fontweight='bold')
    axC.set_ylabel('')
    axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)

    figC.tight_layout()
    figC.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigC_SHAP.pdf'))
    figC.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigC_SHAP.png'), dpi=600)
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

    axD.scatter(y_actual, y_pred, c=z, s=40, cmap=c_cmap, edgecolor='white', linewidth=0.3, alpha=0.9)

    min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val)
    axD.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], color='black', linestyle='--', lw=1.5)
    
    axD.fill_between([min_val-margin, max_val+margin], [(min_val-margin)*0.9, (max_val+margin)*0.9], 
                     [(min_val-margin)*1.1, (max_val+margin)*1.1], color='gray', alpha=0.15, label='±10% Error Band')

    axD.set_xlim(min_val-margin, max_val+margin); axD.set_ylim(min_val-margin, max_val+margin)
    axD.set_xlabel(f'Actual {target} {unit}', fontweight='bold')
    axD.set_ylabel(f'Predicted {target} {unit} ({best_model_name})', fontweight='bold')

    textstr = f'$R^2$ = {r2_best:.4f}\nRMSE = {rmse_best:.4f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    axD.text(0.05, 0.95, textstr, transform=axD.transAxes, fontsize=12, verticalalignment='top', bbox=props, fontweight='bold')
    axD.spines['top'].set_visible(False); axD.spines['right'].set_visible(False)
    axD.legend(loc='lower right', frameon=False)

    figD.tight_layout()
    figD.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigD_BestModel.pdf'))
    figD.savefig(os.path.join(latest_dir, f'{target}_Benchmark_FigD_BestModel.png'), dpi=600)
    plt.close(figD)
    
    print(f"✅ [{target}] translated_text！translated_text {latest_dir}")

print("\n🎉 translated_text！")