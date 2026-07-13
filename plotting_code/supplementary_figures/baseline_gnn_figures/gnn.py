import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score, mean_squared_error

BASE_PATHS = {
    'LOI': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Group2_Fair_GNN_Outputs',
    'Tg': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Group2_Fair_GNN_Outputs',
    'Tensile': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Group2_Fair_GNN_Outputs'
}

OUTPUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_textGNNtranslated_text_translated_text'
os.makedirs(OUTPUT_DIR, exist_ok=True)

UNITS = {'LOI': '(%)', 'Tg': '(°C)', 'Tensile': '(MPa)'}

C_ORANGE = '#D17758' # translated_text
C_NAVY   = '#344660' # translated_text
C_GREEN  = '#889A74' # translated_text
C_GREY   = '#B1B9C2' # translated_text

PALETTES = {
    'LOI':     {'train': C_NAVY,  'test': C_ORANGE, 'scatter': C_ORANGE, 'line': C_NAVY},
    'Tg':      {'train': C_ORANGE, 'test': C_NAVY,   'scatter': C_NAVY,   'line': C_ORANGE},
    'Tensile': {'train': C_GREY,   'test': C_GREEN,  'scatter': C_GREEN,  'line': C_NAVY}
}

mm_to_inch = 1 / 25.4
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.2
})

print(f"==================================================")
print(f"🚀 translated_text GNN translated_text (translated_text)")
print(f"==================================================")

for target_name, base_path in BASE_PATHS.items():
    if not os.path.exists(base_path): continue

    palette = PALETTES[target_name]

    test_res_file = os.path.join(base_path, 'GNN_Test_Predictions.csv')
    if os.path.exists(test_res_file):
        df = pd.read_csv(test_res_file)
        y_true_col = 'True_Value' if 'True_Value' in df.columns else df.columns[0]
        y_pred_col = 'Predicted_Value' if 'Predicted_Value' in df.columns else df.columns[1]
        y_true, y_pred = df[y_true_col], df[y_pred_col]
        
        fig, ax = plt.subplots(figsize=(40*mm_to_inch, 40*mm_to_inch), dpi=300)
        ax.scatter(y_true, y_pred, alpha=0.8, color=palette['scatter'], edgecolor='none', s=8)
        
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
                linestyle='--', color=palette['line'], linewidth=1.2, zorder=3)
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        textstr = f'$R^2$={r2:.3f}\nRMSE={rmse:.2f}'
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=6, fontweight='bold',
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel(f'Actual {target_name} {UNITS[target_name]}', fontweight='bold')
        ax.set_ylabel(f'Predicted {target_name} {UNITS[target_name]}', fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ParityPlot.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ParityPlot.png'), dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {target_name} translated_text！")

    lc_file = os.path.join(base_path, 'GNN_Learning_Curve.csv')
    if not os.path.exists(lc_file):
        lc_file = os.path.join(base_path, 'learning_curve.csv')
        
    if os.path.exists(lc_file):
        df = pd.read_csv(lc_file)
        fig, ax = plt.subplots(figsize=(45*mm_to_inch, 35*mm_to_inch), dpi=300)
        
        epochs = df['Epoch']
        train_col = 'Train_RMSE' if 'Train_RMSE' in df.columns else 'train_rmse'
        test_col = 'Test_RMSE' if 'Test_RMSE' in df.columns else ('Val_RMSE' if 'Val_RMSE' in df.columns else 'test_rmse')
        
        ax.plot(epochs, df[train_col], color=palette['train'], label='Train', linewidth=1.2)
        ax.plot(epochs, df[test_col], color=palette['test'], label='Test/Val', linewidth=1.2)
        
        best_idx = df[test_col].idxmin()
        best_epoch, best_rmse = df.loc[best_idx, 'Epoch'], df.loc[best_idx, test_col]
        ax.axvline(x=best_epoch, color='#666666', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.plot(best_epoch, best_rmse, 'o', color=palette['line'], markersize=3)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(f'RMSE {UNITS[target_name]}', fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_LearningCurve.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_LearningCurve.png'), dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {target_name} translated_text！")