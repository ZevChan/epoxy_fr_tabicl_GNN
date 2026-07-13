import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

BASE_PATHS = {
    'LOI':     r'C:\Users\WINDOWS\Desktop\GNN\LOI\Group2_Fair_GNN_Outputs',
    'Tg':      r'C:\Users\WINDOWS\Desktop\GNN\Tg\Group2_Fair_GNN_Outputs',
    'Tensile': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Group2_Fair_GNN_Outputs'
}

OUTPUT_DIR = os.path.join(r'C:\Users\WINDOWS\Desktop\GNN\translated_textGNNtranslated_text_translated_text', datetime.now().strftime('%Y%m%d'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_ORANGE = '#D17758' 
C_NAVY   = '#344660' 
C_GREEN  = '#889A74' 
C_GREY   = '#B1B9C2'

PALETTES = {
    'LOI':     {'hist': '#F2E2DD', 'line': C_ORANGE}, 
    'Tg':      {'hist': '#DCE0E6', 'line': C_GREEN},   # translated_text
    'Tensile': {'hist': '#E9EDE5', 'line': C_NAVY}     # translated_text
}

mm_to_inch = 1 / 25.4
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.0
})

print(f"==================================================")
print(f"🚀 translated_text GNN translated_text (translated_text)")
print(f"==================================================")

for target_name, base_path in BASE_PATHS.items():
    if not os.path.exists(base_path): continue

    test_res_file = os.path.join(base_path, 'GNN_Test_Predictions.csv')
    if not os.path.exists(test_res_file): continue
        
    df = pd.read_csv(test_res_file)
    y_true_col = 'True_Value' if 'True_Value' in df.columns else df.columns[0]
    y_pred_col = 'Predicted_Value' if 'Predicted_Value' in df.columns else df.columns[1]
    
    residuals = df[y_pred_col] - df[y_true_col]
    palette = PALETTES[target_name]

    fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
    ax.set_box_aspect(1)
    
    ax.hist(residuals, bins=20, density=True, alpha=0.8, color=palette['hist'], edgecolor=palette['line'], linewidth=0.5)
    
    mu, std = norm.fit(residuals)
    x = np.linspace(min(residuals), max(residuals), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, color=palette['line'], linewidth=1.5)
    
    ax.axvline(0, color=C_GREY, linestyle='--', linewidth=1.0)
    
    textstr = f'Mean = {mu:.2f}\nStd = {std:.2f}'
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Residual (Pred - Actual)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_Residual_Hist.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_Residual_Hist.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  ✅ {target_name} translated_text！")