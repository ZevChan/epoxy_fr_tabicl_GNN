import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# ==========================================
GLOBAL_OUT_DIR = r"C:\Users\WINDOWS\Desktop\GNN\translated_text-TABICL-optunatranslated_text"
os.makedirs(GLOBAL_OUT_DIR, exist_ok=True)

TARGETS = {
    'LOI': {
        'csv_path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_TabICL_HPO_K71_20260414_1539\Optuna_HPO_Results.csv',
        'color': '#D17758',   # translated_text
        'metric': 'CV R²',
        'task': 'reg'
    },
    'UL94': {
        'csv_path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_TabICL_HPO_Cls_K70_20260416_0852\Optuna_HPO_Results_Cls.csv',
        'color': '#D17758',   # translated_text
        'metric': 'CV ROC-AUC',
        'task': 'cls'
    },
    'Tg': {
        'csv_path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_TabICL_HPO_K72_20260415_0952\Optuna_HPO_Results.csv',
        'color': '#889A74',   # translated_text
        'metric': 'CV R²',
        'task': 'reg'
    },
    'TENSILE': {
        'csv_path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_TabICL_HPO_K40_20260415_1102\Optuna_HPO_Results.csv',
        'color': '#344660',   # translated_text
        'metric': 'CV R²',
        'task': 'reg'
    }
}

DASH_COLOR = '#DDE0E7'

# ==========================================
# ==========================================
mm_to_inch = 1 / 25.4
width_single = 89 * mm_to_inch    # Nature translated_text
width_double = 183 * mm_to_inch   # Nature translated_text
width_medium = 120 * mm_to_inch

def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7,
        'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
        'figure.dpi': 600
    })

set_nature_style()

def format_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print("="*60)
print(f"📊 translated_text Optuna HPO translated_text (4 Targets translated_text)")
print(f"📁 translated_text: {GLOBAL_OUT_DIR}")
print("="*60)

# ==========================================
# ==========================================

def plot_trace(df, name, config):
    fig, ax = plt.subplots(figsize=(width_single, width_single * 0.85))
    color = config['color']
    metric = config['metric']
    
    ax.scatter(df['number'], df['value'], color=color, alpha=0.4, s=15, label='Individual Trial', edgecolors='none')
    ax.plot(df['number'], df['best_value'], color=color, linewidth=1.5, label=f'Best {metric}', zorder=5)
    
    best_idx = df['value'].idxmax()
    best_val = df.loc[best_idx, 'value']
    ax.scatter(best_idx, best_val, color='white', edgecolor='black', s=35, zorder=6, lw=1.0)
    y_range = df['value'].max() - df['value'].min()
    y_offset = y_range * 0.05 if y_range > 0 else 0.01
    
    ax.annotate(f'{best_val:.3f}', xy=(best_idx, best_val), 
                xytext=(best_idx + len(df)*0.03, best_val - y_offset),
                fontsize=7, fontweight='bold', color='black',
                zorder=10)
    
    if config['task'] == 'cls':
        value_min = df['value'].min()
        value_max = df['value'].max()
        padding = (value_max - value_min) * 0.1
        ax.set_ylim(value_min - padding, value_max + padding)
        
    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    # ax.set_title(f'Optimization Trace ({name})', pad=10, fontweight='bold')
    
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, linestyle=':', alpha=0.5)
    format_spines(ax)
    
    fig.tight_layout()
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig1_HPO_Trace.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig1_HPO_Trace.png'), bbox_inches='tight')
    plt.close()


def plot_slices(df, name, config):
    params = ['params_n_estimators', 'params_outlier_threshold', 'params_feat_shuffle_method']
    titles = ['Number of Estimators', 'Outlier Threshold', 'Feature Shuffle Method']
    color = config['color']
    metric = config['metric']
    
    fig, axes = plt.subplots(1, 3, figsize=(width_double, width_single * 0.8))
    
    for i, (param, title) in enumerate(zip(params, titles)):
        ax = axes[i]
        
        if param == 'params_feat_shuffle_method':
            sns.boxplot(x=param, y='value', data=df, ax=ax, width=0.5,
                        boxprops={'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.6, 'facecolor': color},
                        medianprops={'color': 'white', 'linewidth': 1.2}, showfliers=False)
            sns.stripplot(x=param, y='value', data=df, ax=ax, color='black', size=3, alpha=0.4, jitter=True)
            ax.set_xlabel(title, fontweight='bold')
        else:
            x_vals = df[param].values
            jitter = (x_vals.max() - x_vals.min()) * 0.02
            x_jitter = x_vals + np.random.uniform(-jitter, jitter, size=len(x_vals))
            
            ax.scatter(x_jitter, df['value'], color=color, alpha=0.5, s=12, edgecolor='none')
            
            if len(df[param]) > 3:
                try:
                    sns.regplot(x=param, y='value', data=df, ax=ax, scatter=False, 
                                color='black', line_kws={'linestyle': '--', 'linewidth': 1.0, 'alpha': 0.6})
                except: pass
                
            ax.set_xlabel(title, fontweight='bold')
            best_val = df.loc[df['value'].idxmax(), param]
            ax.axvline(x=best_val, color=DASH_COLOR, linestyle=':', alpha=0.8, linewidth=0.8)
        
        if i == 0:
            ax.set_ylabel(f'{metric}', fontweight='bold')
        else:
            ax.set_ylabel('')
            
        ax.grid(True, linestyle=':', alpha=0.5)
        format_spines(ax)
    
    # plt.suptitle(f'Objective Value vs. Hyperparameters ({name})', y=1.05, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig2_HPO_Slices.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig2_HPO_Slices.png'), bbox_inches='tight')
    plt.close()


def plot_parallel(df, name, config):
    if len(df) < 10: return
    
    plot_data = df[['value', 'params_n_estimators', 'params_outlier_threshold']].copy()
    if 'params_feat_shuffle_method' in df.columns:
        plot_data['shuffle_method'] = df['params_feat_shuffle_method'].map({'latin': 0, 'random': 1})
    
    for col in plot_data.columns:
        if col != 'value':
            v_min, v_max = plot_data[col].min(), plot_data[col].max()
            if v_max > v_min:
                plot_data[col] = (plot_data[col] - v_min) / (v_max - v_min)
    
    fig, ax = plt.subplots(figsize=(width_medium, width_single * 0.9))
    
    norm = plt.Normalize(plot_data['value'].min(), plot_data['value'].max())
    cmap = plt.cm.viridis
    
    for idx, row in plot_data.iterrows():
        values = row.values[1:]
        x_positions = range(len(values))
        ax.plot(x_positions, values, color=cmap(norm(row['value'])), alpha=0.4, linewidth=0.8)
    
    ax.set_xticks(range(len(plot_data.columns)-1))
    ax.set_xticklabels(['n_estimators', 'outlier_threshold', 'shuffle_method'], fontweight='bold')
    ax.set_ylabel('Normalized Value', fontweight='bold')
    # ax.set_title(f'Parallel Coordinates ({name})', fontweight='bold', pad=10)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label(f'Score ({config["metric"]})', fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.5)
    format_spines(ax)
    
    fig.tight_layout()
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig3_HPO_Parallel.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig3_HPO_Parallel.png'), bbox_inches='tight')
    plt.close()


def plot_dist(df, name, config):
    fig, ax = plt.subplots(figsize=(width_single, width_single * 0.85))
    color = config['color']
    metric = config['metric']
    
    ax.hist(df['value'], bins=15, color=color, alpha=0.7, edgecolor='black', linewidth=0.6)
    
    mean_val = df['value'].mean()
    median_val = df['value'].median()
    
    ax.axvline(x=mean_val, color='black', linestyle='-', linewidth=1.2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(x=median_val, color=DASH_COLOR, linestyle='--', linewidth=1.2, label=f'Median: {median_val:.3f}')
    
    ax.set_xlabel(f'Cross-Validation {metric}', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    # ax.set_title(f'Result Distribution ({name})', fontweight='bold', pad=10)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax.legend(frameon=False)
    ax.grid(True, linestyle=':', alpha=0.5)
    format_spines(ax)
    
    fig.tight_layout()
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig4_HPO_Distribution.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(GLOBAL_OUT_DIR, f'{name}_Fig4_HPO_Distribution.png'), bbox_inches='tight')
    plt.close()

# ==========================================
# ==========================================

for name, config in TARGETS.items():
    print(f"\n--- translated_text {name} ---")
    if not os.path.exists(config['csv_path']):
        print(f"⚠️ translated_text {name} translated_text {config['csv_path']}，translated_text。")
        continue
    
    df = pd.read_csv(config['csv_path'])
    df = df[df['state'] == 'COMPLETE']
    df = df.sort_values('number').reset_index(drop=True)
    df['best_value'] = df['value'].cummax()
    
    valid_threshold = 0.0 if config['task'] == 'reg' else 0.45
    df_valid = df[df['value'] > valid_threshold].copy()
    
    if len(df_valid) == 0:
        print(f"⚠️ {name} translated_text，translated_text CSV translated_text。")
        continue

    plot_trace(df_valid, name, config)
    plot_slices(df_valid, name, config)
    plot_parallel(df_valid, name, config)
    plot_dist(df_valid, name, config)
    
    print(f"✅ {name} translated_text 4 translated_text！")

print("\n🎉 translated_text！translated_text: ", GLOBAL_OUT_DIR)