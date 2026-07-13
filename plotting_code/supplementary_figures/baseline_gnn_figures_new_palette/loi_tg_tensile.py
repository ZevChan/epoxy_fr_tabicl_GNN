import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm

BASE_PATHS = {
    'LOI':     r'C:\Users\WINDOWS\Desktop\GNN\LOI\Group2_Fair_GNN_Outputs',
    'Tg':      r'C:\Users\WINDOWS\Desktop\GNN\Tg\Group2_Fair_GNN_Outputs',
    'Tensile': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Group2_Fair_GNN_Outputs'
}

OUTPUT_DIR = os.path.join(r'C:\Users\WINDOWS\Desktop\GNN\translated_textGNNtranslated_text_translated_text', datetime.now().strftime('%Y%m%d'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

UNITS = {'LOI': '(%)', 'Tg': '(°C)', 'Tensile': '(MPa)'}

C_ORANGE = '#D17758'      # translated_text
C_NAVY   = '#344660'      # translated_text
C_GREEN  = '#889A74'      # translated_text
C_GREY   = '#B1B9C2'      # translated_text
C_LIGHT_ORANGE = '#F2E2DD'
C_LIGHT_NAVY   = '#DCE0E6'
C_LIGHT_GREEN  = '#E9EDE5'

PALETTES = {
    'LOI':     {'train': C_GREY, 'test': C_ORANGE, 'scatter': C_ORANGE, 'line': C_GREY,
                'hist_fill': C_LIGHT_ORANGE, 'hist_edge': C_ORANGE},
    'Tg':      {'train': C_GREY, 'test': C_GREEN,  'scatter': C_GREEN,  'line': C_GREY,
                'hist_fill': C_LIGHT_GREEN,  'hist_edge': C_GREEN},
    'Tensile': {'train': C_GREY, 'test': C_NAVY,   'scatter': C_NAVY,   'line': C_GREY,
                'hist_fill': C_LIGHT_NAVY,   'hist_edge': C_NAVY}
}

mm_to_inch = 1 / 25.4
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'grid.linewidth': 0.3
})

def add_minor_ticks(ax):
    """translated_text"""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def darken_hex(hex_color, factor=0.55):
    """translated_text（translated_text RGB translated_text，translated_text）"""
    rgb = mcolors.to_rgb(hex_color)
    return tuple(c * factor for c in rgb)

def plot_parity(target_name, base_path, palette):
    """translated_text vs translated_text"""
    test_file = os.path.join(base_path, 'GNN_Test_Predictions.csv')
    if not os.path.exists(test_file):
        return
    df = pd.read_csv(test_file)
    y_true_col = 'True_Value' if 'True_Value' in df.columns else df.columns[0]
    y_pred_col = 'Predicted_Value' if 'Predicted_Value' in df.columns else df.columns[1]
    y_true, y_pred = df[y_true_col], df[y_pred_col]

    fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
    ax.set_box_aspect(1)
    ax.scatter(y_true, y_pred, alpha=0.8, color=palette['scatter'],
               edgecolor='none', s=8)

    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    lim = [min_val - margin, max_val + margin]
    ax.plot(lim, lim, linestyle='--', color=palette['line'], linewidth=1.2, zorder=3)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}'
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            fontweight='bold', verticalalignment='top', bbox=props)

    ax.set_xlabel(f'Actual {target_name} {UNITS[target_name]}', fontweight='bold')
    ax.set_ylabel(f'Predicted {target_name} {UNITS[target_name]}', fontweight='bold')
    add_minor_ticks(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ParityPlot.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ParityPlot.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  ✅ {target_name} translated_text (R²={r2:.3f})")

def plot_learning_curve(target_name, base_path, palette):
    """translated_text"""
    lc_file = os.path.join(base_path, 'GNN_Training_History.csv')
    if not os.path.exists(lc_file):
        lc_file = os.path.join(base_path, 'learning_curve.csv')
    if not os.path.exists(lc_file):
        return

    df = pd.read_csv(lc_file)
    epochs = df['Epoch']
    train_col = 'Train_RMSE' if 'Train_RMSE' in df.columns else 'train_rmse'
    test_col = 'Test_RMSE' if 'Test_RMSE' in df.columns else ('Val_RMSE' if 'Val_RMSE' in df.columns else 'test_rmse')

    fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
    ax.set_box_aspect(1)
    ax.plot(epochs, df[train_col], color=palette['train'], label='Train', linewidth=1.2)
    ax.plot(epochs, df[test_col], color=palette['test'], label='Test/Val', linewidth=1.2)

    best_idx = df[test_col].idxmin()
    best_epoch, best_rmse = df.loc[best_idx, 'Epoch'], df.loc[best_idx, test_col]
    ax.axvline(x=best_epoch, color=C_GREY, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.plot(best_epoch, best_rmse, 'o', color=palette['test'], markersize=3)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel(f'RMSE {UNITS[target_name]}', fontweight='bold')
    ax.legend(loc='upper right', frameon=False)
    add_minor_ticks(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_LearningCurve.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_LearningCurve.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  ✅ {target_name} translated_text (Best Epoch={best_epoch})")

def plot_residual_scatter(target_name, base_path, palette):
    """translated_text（Predicted vs Residual）—— translated_text"""
    test_file = os.path.join(base_path, 'GNN_Test_Predictions.csv')
    if not os.path.exists(test_file):
        return

    df = pd.read_csv(test_file)
    y_true_col = 'True_Value' if 'True_Value' in df.columns else df.columns[0]
    y_pred_col = 'Predicted_Value' if 'Predicted_Value' in df.columns else df.columns[1]
    residuals = df[y_true_col] - df[y_pred_col]
    y_pred = df[y_pred_col]

    fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
    ax.set_box_aspect(1)

    ax.scatter(y_pred, residuals, color=palette['scatter'], alpha=0.6,
               edgecolors='white', linewidth=0.2, s=25)

    ax.axhline(y=0, color=C_GREY, linestyle='--', linewidth=1.0, alpha=0.7)

    mean_res = residuals.mean()
    std_res = residuals.std()

    dark_color = darken_hex(palette['scatter'], factor=0.5)

    ax.fill_between([y_pred.min(), y_pred.max()],
                    mean_res - std_res, mean_res + std_res,
                    color=dark_color, alpha=0.15)

    ax.axhline(y=mean_res, color=dark_color, linewidth=0.8, linestyle='-', alpha=0.7)

    ax.set_xlabel(f'Predicted {target_name} {UNITS[target_name]}', fontweight='bold')
    ax.set_ylabel('Residual (Actual - Predicted)', fontweight='bold')
    add_minor_ticks(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ResidualScatter.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ResidualScatter.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  ✅ {target_name} translated_text (translated_text + translated_text)")

def plot_residual_histogram(target_name, base_path, palette):
    """translated_text + translated_text（translated_text）"""
    test_file = os.path.join(base_path, 'GNN_Test_Predictions.csv')
    if not os.path.exists(test_file):
        return

    df = pd.read_csv(test_file)
    y_true_col = 'True_Value' if 'True_Value' in df.columns else df.columns[0]
    y_pred_col = 'Predicted_Value' if 'Predicted_Value' in df.columns else df.columns[1]
    residuals = df[y_true_col] - df[y_pred_col]

    fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
    ax.set_box_aspect(1)

    n, bins, _ = ax.hist(residuals, bins=15, density=False, alpha=0.7,
                         color=palette['hist_fill'], edgecolor=palette['hist_edge'], linewidth=0.5)

    mean_res = residuals.mean()
    std_res = residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    bin_width = bins[1] - bins[0]
    scale = len(residuals) * bin_width
    ax.plot(x, norm.pdf(x, mean_res, std_res) * scale,
            color=palette['hist_edge'], linewidth=1.5)

    ax.axvline(x=0, color=C_GREY, linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Residual', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    add_minor_ticks(ax)

    textstr = f'Mean = {mean_res:.3f}\nStd = {std_res:.3f}'
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ResidualHistogram.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{target_name}_GNN_ResidualHistogram.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  ✅ {target_name} translated_text")

def main():
    print(f"==================================================")
    print(f"🚀 translated_text GNN translated_text (translated_text·translated_text) ")
    print(f"==================================================")

    for target_name, base_path in BASE_PATHS.items():
        if not os.path.exists(base_path):
            print(f"  ⚠️ translated_text {target_name}：translated_text")
            continue
        palette = PALETTES[target_name]

        print(f"\n--- {target_name} ---")
        plot_parity(target_name, base_path, palette)
        plot_learning_curve(target_name, base_path, palette)
        plot_residual_scatter(target_name, base_path, palette)
        plot_residual_histogram(target_name, base_path, palette)

    with open(os.path.join(OUTPUT_DIR, 'GNN_Visualization_Summary.txt'), 'w') as f:
        f.write("GNN translated_text (translated_text)\n")
        f.write("=================================\n")
        for t in BASE_PATHS:
            f.write(f"{t}: Parity Plot, Learning Curve, Residual Scatter (deep band), Residual Histogram translated_text\n")
    print(f"\n✅ translated_text {OUTPUT_DIR}")

if __name__ == "__main__":
    main()