import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

GLOBAL_OUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_text-TABICL-SHAPtranslated_text'

TARGETS = {
    'LOI': {'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data', 'color': '#FDB863', 'task': 'reg'},  # translated_text
    'UL94': {'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 'color': '#FDB863', 'task': 'cls'}, # translated_text
    'Tg': {'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data', 'color': '#B2ABD2', 'task': 'reg'},   # translated_text
    'TENSILE': {'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data', 'color': '#92C5DE', 'task': 'reg'}  # translated_text
}

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

print("="*70)
print(">>> 🚀 translated_text: translated_text (Feature Network) - [translated_text & translated_text]")
print("="*70)

for target_key, config in TARGETS.items():
    data_dir = config['path']
    c_primary = config['color']
    task = config['task']
    t_name = target_key
    
    pkl_file = os.path.join(data_dir, 'plot_data.pkl')
    if not os.path.exists(pkl_file): 
        print(f"[{t_name}] translated_text，translated_text: {pkl_file}")
        continue
        
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    X_train = data['X_train']
    shap_values = data['shap_values']
    top_k_features = data['top_k_features']
    
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top_15_idx = np.argsort(shap_abs_mean)[-15:][::-1]
    top_15_features = [top_k_features[i] for i in top_15_idx]
    top_15_shap_vals = shap_abs_mean[top_15_idx]

    # ------------------------------------------
    # ------------------------------------------
    print(f"[{t_name}] translated_text...")
    df_train_top15 = pd.DataFrame(X_train[:, [top_k_features.index(f) for f in top_15_features]], columns=top_15_features)
    corr_matrix = df_train_top15.corr().abs()
    
    G = nx.Graph()
    
    for i, feature in enumerate(top_15_features):
        G.add_node(feature, importance=top_15_shap_vals[i])
        
    THRESHOLD = 0.4
    for i in range(len(top_15_features)):
        for j in range(i+1, len(top_15_features)):
            weight = corr_matrix.iloc[i, j]
            if weight > THRESHOLD:
                G.add_edge(top_15_features[i], top_15_features[j], weight=weight)
                
    fig_net, ax_net = plt.subplots(figsize=(10, 9), dpi=300)
    
    pos = nx.spring_layout(G, k=0.9, seed=42) 
    
    node_sizes = [G.nodes[n]['importance'] * 5000 for n in G.nodes]
    edge_weights = [G.edges[e]['weight'] * 3 for e in G.edges]
    
    nx.draw_networkx_edges(G, pos, ax=ax_net, width=edge_weights, alpha=0.3, edge_color='gray')
    
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=node_sizes, 
                                   node_color=c_primary, alpha=0.9, edgecolors='#777777', linewidths=1.2)
    
    labels = {n: n for n in G.nodes}
    
    text_items = nx.draw_networkx_labels(G, pos, labels, ax=ax_net, font_size=8, font_weight='bold', font_color='black')
    for _, text in text_items.items():
        text.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
    
    ax_net.set_title(f"Feature Mechanism Network ({t_name})\n(Node Size: SHAP Impact | Edge: Pearson > 0.4)", fontweight='bold', pad=20)
    ax_net.axis('off')
    
    fig_net.tight_layout()
    fig_net.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_12_FeatureNetwork_FullName.pdf'), bbox_inches='tight')
    fig_net.savefig(os.path.join(GLOBAL_OUT_DIR, f'{target_key}_12_FeatureNetwork_FullName.png'), dpi=600, bbox_inches='tight')
    plt.close(fig_net)

print("\n🎉 translated_text！translated_text Feature Network translated_text！")