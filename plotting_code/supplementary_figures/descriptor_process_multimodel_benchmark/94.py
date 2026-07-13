import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

ROOT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\94'

result_dirs = glob.glob(os.path.join(ROOT_DIR, "Result_NoGNN_5Fold_CLASSIFICATION_*"))

if not result_dirs:
    print(f"translated_text: translated_text {ROOT_DIR} translated_text (Result_NoGNN_5Fold_CLASSIFICATION_)。")
    exit()

data_dir = max(result_dirs, key=os.path.getmtime)
print(f">>> translated_text UL-94 (translated_text) translated_text:\n{data_dir}")

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['savefig.dpi'] = 600

C_PRIMARY = '#D55E00'  # translated_text

try:
    summary_file = glob.glob(os.path.join(data_dir, 'CV_Statistical_Summary*.csv'))[0]
    folds_file = glob.glob(os.path.join(data_dir, 'CV_All_Folds_Results*.csv'))[0]
    df_summary = pd.read_csv(summary_file)
    df_folds = pd.read_csv(folds_file)
except IndexError:
    print("translated_text：translated_text CSV translated_text。")
    exit()

df_summary = df_summary.sort_values(by='Mean_Test_AUC', ascending=False)
top_models = df_summary.head(10)['Model'].tolist()
df_folds_top = df_folds[df_folds['Model'].isin(top_models)]
best_model_name = top_models[0]

oof_file = os.path.join(data_dir, f'OOF_Predictions_{best_model_name}_Cls.csv')
df_oof = pd.read_csv(oof_file)
actual = df_oof['Actual_UL94'].values
probs = df_oof['Predicted_Probability'].values 
preds = df_oof['Predicted_Class'].values

print(">>> translated_text UL-94 translated_text...")

# ------------------------------------------
# ------------------------------------------
figA, axA = plt.subplots(figsize=(8, 6))
sns.barplot(data=df_summary.head(10), x='Mean_Test_AUC', y='Model', color=C_PRIMARY, ax=axA, edgecolor='black', linewidth=1.0, alpha=0.85)

for i, model in enumerate(df_summary.head(10)['Model']):
    mean_val = df_summary.iloc[i]['Mean_Test_AUC']
    std_val = df_summary.iloc[i]['Std_Test_AUC']
    axA.errorbar(mean_val, i, xerr=std_val, color='black', capsize=4, capthick=1.2, elinewidth=1.2)
    axA.text(mean_val + std_val + 0.01, i, f"{mean_val:.3f}", va='center', fontsize=10, fontweight='bold')

axA.set_xlabel('Mean ROC-AUC (5-Fold CV)', fontweight='bold')
axA.set_ylabel('')
axA.set_xlim(0, 1.05)
axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)

figA.tight_layout()
figA.savefig(os.path.join(data_dir, "UL94_Benchmark_FigA_Ranking.pdf"))
figA.savefig(os.path.join(data_dir, "UL94_Benchmark_FigA_Ranking.png"))
plt.close(figA)
print("  ✅ [A] AUC translated_text")

# ------------------------------------------
# ------------------------------------------
figB, axB = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df_folds_top, x='Test_AUC', y='Model', order=top_models, color='white', ax=axB, 
            width=0.6, boxprops={'edgecolor':C_PRIMARY, 'linewidth':1.5}, whiskerprops={'color':'black'}, medianprops={'color':'black'})
sns.stripplot(data=df_folds_top, x='Test_AUC', y='Model', order=top_models, color=C_PRIMARY, size=6, alpha=0.7, ax=axB, jitter=True)

axB.set_xlabel('ROC-AUC Score per Fold', fontweight='bold')
axB.set_ylabel('')
axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)

figB.tight_layout()
figB.savefig(os.path.join(data_dir, "UL94_Benchmark_FigB_Stability.pdf"))
figB.savefig(os.path.join(data_dir, "UL94_Benchmark_FigB_Stability.png"))
plt.close(figB)
print("  ✅ [B] translated_text")

# ------------------------------------------
# ------------------------------------------
shap_files = glob.glob(os.path.join(data_dir, 'Global_Top20_Features_Cls.csv')) + glob.glob(os.path.join(data_dir, 'Global_Top20_Features.csv'))
if shap_files:
    figC, axC = plt.subplots(figsize=(8, 6))
    df_shap = pd.read_csv(shap_files[0])
    top_15_shap = df_shap.head(15).copy()
    top_15_shap['Display_Name'] = top_15_shap['Feature_Name'].apply(lambda x: x[:30]+'...' if len(x)>30 else x)

    sns.barplot(data=top_15_shap, x='SHAP_Importance', y='Display_Name', color=C_PRIMARY, ax=axC, edgecolor='black', linewidth=1.0, alpha=0.85)
    axC.set_xlabel('Mean |SHAP Value| (Impact on UL-94)', fontweight='bold')
    axC.set_ylabel('')
    axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)

    figC.tight_layout()
    figC.savefig(os.path.join(data_dir, "UL94_Benchmark_FigC_SHAP.pdf"))
    figC.savefig(os.path.join(data_dir, "UL94_Benchmark_FigC_SHAP.png"))
    plt.close(figC)
    print("  ✅ [C] SHAP translated_text")

# ------------------------------------------
# ------------------------------------------
figD, (ax_roc, ax_cm) = plt.subplots(1, 2, figsize=(12, 5))

fpr, tpr, thresholds = roc_curve(actual, probs)
roc_auc = auc(fpr, tpr)
ax_roc.plot(fpr, tpr, color=C_PRIMARY, lw=2.0, label=f'AUC = {roc_auc:.3f}')
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.0)
ax_roc.set_xlim([-0.02, 1.02]); ax_roc.set_ylim([-0.02, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
ax_roc.legend(loc='lower right', frameon=False, fontsize=12)
ax_roc.spines['top'].set_visible(False); ax_roc.spines['right'].set_visible(False)

cm = confusion_matrix(actual, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False, 
            annot_kws={"size": 12, "weight": "bold"}, ax=ax_cm,
            linewidths=0.5, linecolor='black', square=True)

ax_cm.set_xlabel('Predicted Label', fontweight='bold', labelpad=5)
ax_cm.set_ylabel('True Label', fontweight='bold', labelpad=5)
ax_cm.set_xticklabels(['Fail (0)', 'Pass (1)'])
ax_cm.set_yticklabels(['Fail (0)', 'Pass (1)'])

figD.tight_layout()
figD.savefig(os.path.join(data_dir, "UL94_Benchmark_FigD_BestModel_Eval.pdf"))
figD.savefig(os.path.join(data_dir, "UL94_Benchmark_FigD_BestModel_Eval.png"))
plt.close(figD)
print(f"  ✅ [D] translated_text ({best_model_name}) ROC translated_text")

print(f"\n🎉 UL-94 translated_text！translated_text: {data_dir}")