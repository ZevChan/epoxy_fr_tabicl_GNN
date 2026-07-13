import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

BASE_PATH = r'C:\Users\WINDOWS\Desktop\GNN\94\Group2_Fair_GNN_CV_Outputs'
OUTPUT_DIR = os.path.join(r'C:\Users\WINDOWS\Desktop\GNN\translated_textGNNtranslated_text_translated_text', datetime.now().strftime('%Y%m%d'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_ORANGE = '#D17758'
C_GREY   = '#B1B9C2'      # translated_text，translated_text

mm_to_inch = 1 / 25.4
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.0
})

def format_spines(ax):
    """translated_text（translated_text，translated_text）"""
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

if not os.path.exists(BASE_PATH):
    print(f"translated_text: translated_text {BASE_PATH}")
else:
    pred_file = os.path.join(BASE_PATH, "PlainGNN_5Fold_OOF_Predictions.csv")
    if os.path.exists(pred_file):
        df_pred = pd.read_csv(pred_file)
        
        y_true = df_pred['Actual_UL94']
        y_prob = df_pred['Predicted_Probability']
        y_pred_class = df_pred['Predicted_Class']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
        ax.set_box_aspect(1)
        ax.plot(fpr, tpr, color=C_ORANGE, lw=1.5, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color=C_GREY, lw=1.0, linestyle='--')
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.legend(loc="lower right", frameon=False)
        format_spines(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_ROC_Curve.pdf"), bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_ROC_Curve.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print("  ✅ [UL94] ROC translated_text")

        cm = confusion_matrix(y_true, y_pred_class)
        fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
        ax.set_box_aspect(1)
        cmap_custom = mpl.colors.LinearSegmentedColormap.from_list("custom_orange", ["#FDF3F0", C_ORANGE])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_custom, cbar=False,
                    annot_kws={"size": 12, "weight": "bold"}, ax=ax,
                    linewidths=0.5, linecolor='black', square=True)
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xticklabels(['Fail (0)', 'Pass (1)'])
        ax.set_yticklabels(['Fail (0)', 'Pass (1)'])
        format_spines(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Confusion_Matrix.pdf"), bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Confusion_Matrix.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print("  ✅ [UL94] translated_text")
    else:
        print(f"translated_text: translated_text {pred_file}")

    metrics_file = os.path.join(BASE_PATH, "PlainGNN_5Fold_Metrics.csv")
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        
        figB, axB = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
        axB.set_box_aspect(1)
        
        box_data = [metrics_df['AUC'].values]
        bp = axB.boxplot(box_data, positions=[1], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor=C_GREY, alpha=0.5),   # translated_text
                         medianprops=dict(color="black", linewidth=1.2), 
                         showfliers=False)
        
        axB.set_xlim(0.5, 1.5)
        axB.set_xticks([1])
        axB.set_xticklabels(['Plain GNN'], fontweight='bold')
        axB.set_ylabel('Cross-validated ROC-AUC', fontweight='bold')
        
        for val in metrics_df['AUC']:
            axB.scatter(1 + np.random.uniform(-0.08, 0.08), val,
                        color=C_ORANGE, alpha=0.8, s=15,
                        edgecolors='black', linewidths=0.5)
        
        format_spines(axB)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_AUC_Stability.pdf"), bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_AUC_Stability.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print("  ✅ [UL94] AUC translated_text")
    else:
        print(f"translated_text: translated_text {metrics_file}")

    lc_file = os.path.join(BASE_PATH, "PlainGNN_BestFold_LearningCurve.csv")
    if os.path.exists(lc_file):
        df_lc = pd.read_csv(lc_file)
        
        fig, ax = plt.subplots(figsize=(85*mm_to_inch, 85*mm_to_inch), dpi=600)
        ax.set_box_aspect(1)
        
        ax.plot(df_lc['Epoch'], df_lc['Train_Loss'], color=C_GREY, label='Train Loss', lw=1.2)
        ax.plot(df_lc['Epoch'], df_lc['Val_Loss'], color=C_ORANGE, label='Val Loss', lw=1.2)
        
        best_idx = df_lc['Val_Loss'].idxmin()
        best_epoch, best_val = df_lc.loc[best_idx, 'Epoch'], df_lc.loc[best_idx, 'Val_Loss']
        
        ax.axvline(x=best_epoch, color=C_GREY, linestyle='--', linewidth=0.8, zorder=1)
        ax.plot(best_epoch, best_val, 'o', color=C_ORANGE, markersize=3, zorder=5)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('BCE Loss', fontweight='bold')
        format_spines(ax)
        ax.legend(loc='upper right', frameon=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Learning_Curve.pdf"), bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Learning_Curve.png"), dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print("  ✅ [UL94] translated_text")
    else:
        print(f"translated_text: translated_text {lc_file}")