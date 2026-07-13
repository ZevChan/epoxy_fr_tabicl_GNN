import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

GLOBAL_OUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_text-TABICL-SHAPtranslated_text'

TARGETS = {
    'UL94': {
        'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 
        'color': '#D55E00', 
        'color_neg': '#8491B4', 
        'task': 'cls'
    }
}

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 1.2, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.labelsize': 12, 'axes.titlesize': 14
})

def format_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print("="*70)
print(">>> 🚀 translated_text: translated_text (Probability CDF)")
print("="*70)

for target_key, config in TARGETS.items():
    data_dir = config['path']
    c_primary = config['color']
    c_neg = config['color_neg']
    task = config['task']
    t_name = target_key
    
    pkl_file = os.path.join(data_dir, 'plot_data.pkl')
    if not os.path.exists(pkl_file): 
        print(f"[{t_name}] translated_text，translated_text: {pkl_file}")
        continue
        
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    y_test = data['y_test']
    probs = data['probs']  # translated_text 1 (Pass) translated_text
    
    print(f"\n--- translated_text {t_name} ---")

    # ==========================================
    # ==========================================
    probs_fail = probs[y_test == 0]
    probs_pass = probs[y_test == 1]
    
    sorted_fail = np.sort(probs_fail)
    cdf_fail = np.arange(1, len(sorted_fail) + 1) / len(sorted_fail)
    
    sorted_pass = np.sort(probs_pass)
    cdf_pass = np.arange(1, len(sorted_pass) + 1) / len(sorted_pass)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    ax.plot(sorted_fail, cdf_fail, color=c_neg, linewidth=2.5, label='Actual Fail (Class 0)')
    ax.fill_between(sorted_fail, 0, cdf_fail, color=c_neg, alpha=0.15)
    
    ax.plot(sorted_pass, cdf_pass, color=c_primary, linewidth=2.5, label='Actual Pass (Class 1)')
    ax.fill_between(sorted_pass, 0, cdf_pass, color=c_primary, alpha=0.15)
    
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    ax.text(0.52, 0.4, 'Decision Boundary\n(Threshold = 0.5)', 
            color='gray', fontweight='bold', fontsize=10, va='center')
    
    ax.set_xlabel(f'Predicted Probability for {t_name} Pass (1)', fontweight='bold')
    ax.set_ylabel('Cumulative Probability (CDF)', fontweight='bold')
    ax.set_title(f'Classification Confidence Separation ({t_name})', fontweight='bold', pad=15)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), frameon=False, ncol=1)
    format_spines(ax)
    
    fig.tight_layout()
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_16_Probability_CDF.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_16_Probability_CDF.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

print("\ntranslated_text！translated_text UL94 translated_text CDF translated_text。")