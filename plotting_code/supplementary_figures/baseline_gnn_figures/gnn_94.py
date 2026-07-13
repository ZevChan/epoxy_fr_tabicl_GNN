import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

BASE_PATH = r'C:\Users\WINDOWS\Desktop\GNN\94\Group2_Fair_GNN_CV_Outputs'

OUTPUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_textGNNtranslated_text_translated_text'
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_ORANGE = '#D17758' # translated_text (translated_text)
C_NAVY   = '#344660' # translated_text (translated_text)
C_GREY   = '#B1B9C2' # translated_text (translated_text)

mm_to_inch = 1 / 25.4
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.2
})

print(f"==================================================")
print(f"🚀 translated_text GNN UL-94 translated_text (translated_text)")
print(f"==================================================")

if not os.path.exists(BASE_PATH):
    print(f"translated_text: translated_text {BASE_PATH}")
else:
    # ------------------------------------------
    # ------------------------------------------
    roc_file = os.path.join(BASE_PATH, "GNN_Test_ROC_Curve.csv")
    if os.path.exists(roc_file):
        df_roc = pd.read_csv(roc_file)
        fpr, tpr = df_roc['FPR'], df_roc['TPR']
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(45*mm_to_inch, 45*mm_to_inch), dpi=300)
        ax.plot(fpr, tpr, color=C_ORANGE, lw=1.5, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color=C_GREY, lw=1.0, linestyle='--')
        
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.legend(loc="lower right", frameon=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_ROC_Curve.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_ROC_Curve.png"), dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✅ [UL94] ROC translated_text")

    # ------------------------------------------
    # ------------------------------------------
    cm_file = os.path.join(BASE_PATH, "GNN_Test_Confusion_Matrix.csv")
    if os.path.exists(cm_file):
        df_cm = pd.read_csv(cm_file, index_col=0)
        cm = df_cm.values
        
        fig, ax = plt.subplots(figsize=(40*mm_to_inch, 40*mm_to_inch), dpi=300)
        
        cmap_custom = mpl.colors.LinearSegmentedColormap.from_list("custom_orange", ["#FDF3F0", C_ORANGE])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_custom, cbar=False, 
                    annot_kws={"size": 8, "weight": "bold"}, ax=ax,
                    linewidths=0.5, linecolor='black', square=True)
        
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xticklabels(['Fail (0)', 'Pass (1)'])
        ax.set_yticklabels(['Fail (0)', 'Pass (1)'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Confusion_Matrix.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Confusion_Matrix.png"), dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✅ [UL94] translated_text")

    # ------------------------------------------
    # ------------------------------------------
    lc_file = os.path.join(BASE_PATH, "GNN_Learning_Curve.csv")
    if os.path.exists(lc_file):
        df_lc = pd.read_csv(lc_file)
        fig, ax = plt.subplots(figsize=(50*mm_to_inch, 35*mm_to_inch), dpi=300)
        
        ax.plot(df_lc['Epoch'], df_lc['Train_Loss'], color=C_NAVY, label='Train Loss', lw=1.2)
        ax.plot(df_lc['Epoch'], df_lc['Val_Loss'], color=C_ORANGE, label='Val Loss', lw=1.2)
        
        best_idx = df_lc['Val_Loss'].idxmin()
        best_epoch, best_val = df_lc.loc[best_idx, 'Epoch'], df_lc.loc[best_idx, 'Val_Loss']
        ax.axvline(x=best_epoch, color=C_GREY, linestyle='--', linewidth=0.8, zorder=1)
        ax.plot(best_epoch, best_val, 'o', color=C_ORANGE, markersize=3, zorder=5)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('BCE Loss', fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Learning_Curve.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, "UL94_GNN_Learning_Curve.png"), dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✅ [UL94] translated_text")