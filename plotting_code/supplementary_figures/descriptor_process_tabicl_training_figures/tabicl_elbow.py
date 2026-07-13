import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns  # translated_text seaborn translated_text

ROOT_DIR = r'C:\Users\WINDOWS\Desktop\GNN'

TARGETS = ['LOI', 'Tg', 'TENSILE']

OPTIMAL_K_DICT = {
    'LOI': 71,
    'Tg': 72,
    'TENSILE': 40
}

PALETTES = {
    'LOI':     {'r2': '#D55E00', 'rmse': '#E69F00'}, 
    'Tg':      {'r2': '#332288', 'rmse': '#CC6677'}, 
    'TENSILE': {'r2': '#0072B2', 'rmse': '#009E73'}  
}

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

for target in TARGETS:
    print(f"\n{'='*50}\n>>> translated_text: {target}\n{'='*50}")
    
    target_dir = os.path.join(ROOT_DIR, target)
    if not os.path.exists(target_dir):
        print(f"[translated_text] translated_text: {target_dir}")
        continue
        
    result_dirs = glob.glob(os.path.join(target_dir, "Result_TabICL_Elbow_*"))
    if not result_dirs:
        print(f"[translated_text] translated_text {target} translated_text TabICL translated_text。")
        continue
        
    latest_dir = max(result_dirs, key=os.path.getmtime)
    print(f"📁 translated_text: {os.path.basename(latest_dir)}")
    
    csv_file = os.path.join(latest_dir, 'TabICL_Feature_Selection_Curve.csv')
    if not os.path.exists(csv_file):
        print(f"[translated_text] translated_text {csv_file}")
        continue
        
    df = pd.read_csv(csv_file)
    x = df['Num_Features'].values
    y_r2, std_r2 = df['Mean_R2'].values, df['Std_R2'].values
    y_rmse, std_rmse = df['Mean_RMSE'].values, df['Std_RMSE'].values
    
    target_k = OPTIMAL_K_DICT[target]
    idx = np.argmin(np.abs(x - target_k))
    opt_k = int(x[idx])
    opt_r2 = y_r2[idx]
    opt_rmse = y_rmse[idx]
    
    palette = PALETTES[target]
    c_r2, c_rmse = palette['r2'], palette['rmse']
    
    # ------------------------------------------
    # ------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=300)
    ax1.plot(x, y_r2, color=c_r2, lw=2.5, zorder=3, label='Mean R²')
    ax1.fill_between(x, y_r2 - std_r2, y_r2 + std_r2, color=c_r2, alpha=0.2, zorder=2)
    
    ax1.axvline(x=opt_k, color='black', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
    ax1.scatter([opt_k], [opt_r2], color='white', edgecolor='black', s=100, lw=2.5, zorder=5)
    ax1.annotate(f'Selected Optimal $K$\nk={opt_k}\nR²={opt_r2:.3f}', 
                 xy=(opt_k, opt_r2), xytext=(opt_k + max(x)*0.05, opt_r2 - max(y_r2)*0.05),
                 fontsize=11, fontweight='bold', arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
    
    ax1.set_xlabel('Number of Top SHAP Features', fontweight='bold')
    ax1.set_ylabel('Mean R² Score (5-Fold CV)', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(latest_dir, f'{target}_TabICL_OptimalK_R2.png'), dpi=600, bbox_inches='tight')
    fig1.savefig(os.path.join(latest_dir, f'{target}_TabICL_OptimalK_R2.pdf'), bbox_inches='tight')
    plt.close(fig1)

    # ------------------------------------------
    # ------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=300)
    ax2.plot(x, y_rmse, color=c_rmse, lw=2.5, zorder=3, label='Mean RMSE')
    ax2.fill_between(x, y_rmse - std_rmse, y_rmse + std_rmse, color=c_rmse, alpha=0.2, zorder=2)
    
    ax2.axvline(x=opt_k, color='black', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
    ax2.scatter([opt_k], [opt_rmse], color='white', edgecolor='black', s=100, lw=2.5, zorder=5)
    ax2.annotate(f'Selected Optimal $K$\nk={opt_k}\nRMSE={opt_rmse:.3f}', 
                 xy=(opt_k, opt_rmse), xytext=(opt_k + max(x)*0.05, opt_rmse + max(y_rmse)*0.05),
                 fontsize=11, fontweight='bold', arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
    
    ax2.set_xlabel('Number of Top SHAP Features', fontweight='bold')
    ax2.set_ylabel('Mean RMSE (5-Fold CV)', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(latest_dir, f'{target}_TabICL_OptimalK_RMSE.png'), dpi=600, bbox_inches='tight')
    fig2.savefig(os.path.join(latest_dir, f'{target}_TabICL_OptimalK_RMSE.pdf'), bbox_inches='tight')
    plt.close(fig2)
    
    # ------------------------------------------
    # ------------------------------------------
    shap_file = os.path.join(latest_dir, 'Global_SHAP_Features_All.csv')
    if os.path.exists(shap_file):
        df_shap = pd.read_csv(shap_file)
        top_15_shap = df_shap.head(15).copy()
        top_15_shap['Display_Name'] = top_15_shap['Feature_Name'].apply(lambda name: name[:30]+'...' if len(name)>30 else name)

        fig3, ax3 = plt.subplots(figsize=(7, 6), dpi=300)
        sns.barplot(data=top_15_shap, x='SHAP_Importance', y='Display_Name', color=c_r2, ax=ax3, edgecolor='black', linewidth=1.2)
        
        ax3.set_xlabel(f'Mean |SHAP Value| (Impact on {target})', fontweight='bold')
        ax3.set_ylabel('')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        fig3.tight_layout()
        fig3.savefig(os.path.join(latest_dir, f'{target}_TabICL_SHAP_Top15.png'), dpi=600, bbox_inches='tight')
        fig3.savefig(os.path.join(latest_dir, f'{target}_TabICL_SHAP_Top15.pdf'), bbox_inches='tight')
        plt.close(fig3)
        print(f"✅ [{target}] translated_text (K={opt_k})！translated_text R2, RMSE translated_text SHAP translated_text。")
    else:
        print(f"⚠️ [{target}] translated_text {shap_file}，translated_text SHAP translated_text。")

print("\n🎉 translated_text！translated_text。")