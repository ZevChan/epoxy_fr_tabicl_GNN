import os, pickle, glob, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde, pearsonr
import matplotlib.colors as mcolors
import matplotlib as mpl
import shap

warnings.filterwarnings('ignore')

OUTPUT_DIR = r'C:\Users\WINDOWS\Desktop\GNN\translated_text-TABICL-SHAPtranslated_text_translated_text'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DASH_COLOR = '#DDE0E7'
# Figure 2b per-target colors
COLOR_LOI = '#1F4E79'
COLOR_UL94 = '#C0504D'
COLOR_TG = '#4A7B9D'
COLOR_TENSILE = '#E08E36'

COLOR_NEG_1 = '#8491B4'
COLOR_NEG_2 = '#91D1C2'
COLOR_NEG_3 = '#F39B7F'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 1.3,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'axes.labelsize': 16, 'axes.titlesize': 16
})

def format_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def wrap_labels(label, max_len=30):
    return label[:max_len-3] + '...' if len(label) > max_len else label

PREFERRED_FEATURE_ORDER = {
    "LOI": ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "FR_B01[C-P]", "EP_wt_fraction"],
    "UL94": ["Flame_retardant_AdditionAmount(wt%)", "FR_max_conj_path", "CURING_Mor30p", "EP_wt_fraction"],
    "Tg": ["EEW", "T_max", "Q_thermal", "CURING_RBF", "CURING_SssCH2"],
    "TENSILE": ["Q_thermal", "FR_E1m", "FR_TPSA_efficiency", "Flame_retardant_AdditionAmount(wt%)", "EP_SIC5"],
}

def feature_order_for_plot(t_key, available_features, fallback_features, max_count):
    ordered = [f for f in PREFERRED_FEATURE_ORDER.get(t_key, []) if f in available_features]
    for f in fallback_features:
        if f not in ordered:
            ordered.append(f)
        if len(ordered) >= max_count:
            break
    return ordered[:max_count]

def generate_ul94_cdf():
    print(">>> UL94 CDF translated_text")
    pkl = r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data\plot_data.pkl'
    if not os.path.exists(pkl): return
    with open(pkl, 'rb') as f: data = pickle.load(f)
    y_test, probs = data['y_test'], data['probs']
    probs_fail = probs[y_test==0]; probs_pass = probs[y_test==1]
    sorted_fail = np.sort(probs_fail); cdf_fail = np.arange(1, len(sorted_fail)+1)/len(sorted_fail)
    sorted_pass = np.sort(probs_pass); cdf_pass = np.arange(1, len(sorted_pass)+1)/len(sorted_pass)

    fig, ax = plt.subplots(figsize=(6,5), dpi=600)
    ax.plot(sorted_fail, cdf_fail, color=COLOR_NEG_1, lw=2.5, label='Actual Fail (Class 0)')
    ax.fill_between(sorted_fail, 0, cdf_fail, color=COLOR_NEG_1, alpha=0.15)
    ax.plot(sorted_pass, cdf_pass, color=COLOR_UL94, lw=2.5, label='Actual Pass (Class 1)')
    ax.fill_between(sorted_pass, 0, cdf_pass, color=COLOR_UL94, alpha=0.15)
    ax.axvline(0.5, color=DASH_COLOR, linestyle='--', lw=1.5)
    ax.text(0.52, 0.4, 'Decision Boundary\n(Threshold = 0.5)', color='black', fontweight='bold', fontsize=10, va='center')
    ax.set_xlabel('Predicted Probability for UL94 Pass (1)', fontweight='bold')
    ax.set_ylabel('Cumulative Probability (CDF)', fontweight='bold')
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([0,1.05])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,0.95), frameon=False)
    format_spines(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'UL94_Probability_CDF.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'UL94_Probability_CDF.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

def generate_feature_network():
    print(">>> translated_text")
    TARGETS_NET = {
        'LOI': {'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_LOI, 'task': 'reg'},
        'UL94': {'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 'color': COLOR_UL94, 'task': 'cls'},
        'Tg': {'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TG, 'task': 'reg'},
        'TENSILE': {'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TENSILE, 'task': 'reg'}
    }
    for t_name, cfg in TARGETS_NET.items():
        pkl = os.path.join(cfg['path'], 'plot_data.pkl')
        if not os.path.exists(pkl): continue
        with open(pkl, 'rb') as f: data = pickle.load(f)
        X_train = data['X_train']; shap_values = data['shap_values']; top_k_features = data['top_k_features']
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        top_15_idx = np.argsort(shap_abs_mean)[-15:][::-1]
        top_15_features = [top_k_features[i] for i in top_15_idx]
        top_15_shap_vals = shap_abs_mean[top_15_idx]
        df_train_top15 = pd.DataFrame(X_train[:, [top_k_features.index(f) for f in top_15_features]], columns=top_15_features)
        corr = df_train_top15.corr().abs()
        G = nx.Graph()
        for i, f in enumerate(top_15_features): G.add_node(f, importance=top_15_shap_vals[i])
        TH = 0.4
        for i in range(len(top_15_features)):
            for j in range(i+1, len(top_15_features)):
                w = corr.iloc[i,j]
                if w > TH: G.add_edge(top_15_features[i], top_15_features[j], weight=w)
        fig_net, ax_net = plt.subplots(figsize=(10,9), dpi=600)
        pos = nx.spring_layout(G, k=0.9, seed=42)
        node_sizes = [G.nodes[n]['importance']*5000 for n in G.nodes]
        edge_weights = [G.edges[e]['weight']*3 for e in G.edges]
        nx.draw_networkx_edges(G, pos, ax=ax_net, width=edge_weights, alpha=0.3, edge_color=DASH_COLOR)
        nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=node_sizes, node_color=cfg['color'], alpha=0.9, edgecolors='#777777', linewidths=1.2)
        labels = {n: n for n in G.nodes}
        text_items = nx.draw_networkx_labels(G, pos, labels, ax=ax_net, font_size=8, font_weight='bold', font_color='black')
        for _, text in text_items.items():
            text.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
        ax_net.axis('off')
        fig_net.tight_layout()
        fig_net.savefig(os.path.join(OUTPUT_DIR, f'{t_name}_FeatureNetwork_FullName.pdf'), bbox_inches='tight')
        fig_net.savefig(os.path.join(OUTPUT_DIR, f'{t_name}_FeatureNetwork_FullName.png'), dpi=600, bbox_inches='tight')
        plt.close(fig_net)

def generate_advanced_shap_corr():
    print(">>> translated_textSHAPtranslated_text")
    TARGETS_SRC = {
        'LOI': {'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_LOI, 'color_neg': COLOR_NEG_1, 'target_name': 'LOI', 'task': 'reg'},
        'UL94': {'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 'color': COLOR_UL94, 'color_neg': COLOR_NEG_1, 'target_name': 'UL-94', 'task': 'cls'},
        'Tg': {'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TG, 'color_neg': COLOR_NEG_2, 'target_name': 'Tg', 'task': 'reg'},
        'TENSILE': {'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TENSILE, 'color_neg': COLOR_NEG_3, 'target_name': 'TENSILE', 'task': 'reg'}
    }
    for t_key, cfg in TARGETS_SRC.items():
        data_dir = cfg['path']; c_primary = cfg['color']; c_neg = cfg['color_neg']; t_name = cfg['target_name']; task = cfg['task']
        pkl = os.path.join(data_dir, 'plot_data.pkl')
        if not os.path.exists(pkl): continue
        with open(pkl, 'rb') as f: data = pickle.load(f)
        X_test, y_test = data['X_test'], data['y_test']; X_train, y_train = data['X_train'], data['y_train']
        shap_values = data['shap_values']; top_k_features = data['top_k_features']; top_6_features = data['top_6_features']; preds = data['preds']
        df_test_X = pd.DataFrame(X_test, columns=top_k_features)
        if task == 'cls':
            probs = data['probs']; expected_value = data['expected_value']
        else:
            pred_df = pd.read_csv(os.path.join(data_dir, 'predictions.csv'))
            residuals = pred_df['Residual'].values
            inferred_base = preds - np.sum(shap_values, axis=1)
            expected_value = np.mean(inferred_base)

        cmap_custom = mcolors.LinearSegmentedColormap.from_list(f"cmap_{t_key}", ["#F2F2F2", c_primary])
        cmap_diverging = mcolors.LinearSegmentedColormap.from_list(f"cmap_div_{t_key}", [c_neg, "#F2F2F2", c_primary])
        cmap_light = mcolors.LinearSegmentedColormap.from_list(f"light_{t_key}", ["white", c_primary])

        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        top_15_idx_global = np.argsort(shap_abs_mean)[-15:][::-1]
        top_15_feats_global = [top_k_features[i] for i in top_15_idx_global]
        ordered_bar = feature_order_for_plot(t_key, top_k_features, top_15_feats_global, 8)
        # Reverse for barh (most important at top)
        display_order = ordered_bar[::-1]
        values_for_bars = np.array([shap_abs_mean[top_k_features.index(f)] for f in display_order])
        top_features_display = [wrap_labels(f) for f in display_order]
        norm_bar = plt.Normalize(vmin=values_for_bars.min(), vmax=values_for_bars.max())
        colors_bar = [cmap_light(norm_bar(v)) for v in values_for_bars]
        fig, ax = plt.subplots(figsize=(7,6), dpi=600)
        ax.barh(top_features_display, values_for_bars, color=colors_bar, edgecolor='black', linewidth=1.0)
        ax.set_xlabel(f'Mean |SHAP Value| (Impact on {t_name})', fontweight='bold', fontsize=14, labelpad=10)
        ax.tick_params(labelsize=13)
        for s in ax.spines.values(): s.set_linewidth(1.2)
        format_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_1_SHAP_Bar.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_1_SHAP_Bar.png'), dpi=600, bbox_inches='tight')
        plt.close(fig)

        for feat in top_6_features[:5]:
            fig, ax = plt.subplots(figsize=(6,5), dpi=600)
            shap.dependence_plot(feat, shap_values, df_test_X, ax=ax, show=False, interaction_index=None,
                                 color=c_primary, alpha=0.8, dot_size=30)
            for coll in ax.collections:
                coll.set_edgecolor('white')
                coll.set_linewidth(0.3)
            ax.set_ylabel('SHAP Value', fontweight='bold')
            format_spines(ax)
            fig.tight_layout()
            safe_name = feat.replace('/','_').replace('\\','_')
            fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_2_PDP_{safe_name}.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_2_PDP_{safe_name}.png'), dpi=600, bbox_inches='tight')
            plt.close(fig)

        pca = PCA(n_components=2, random_state=42)
        shap_pca = pca.fit_transform(shap_values)
        fig, ax = plt.subplots(figsize=(6,5), dpi=600)
        if task == 'cls':
            ax.scatter(shap_pca[y_test==0,0], shap_pca[y_test==0,1], color=c_neg, label='Fail (0)', alpha=0.8, s=30)
            ax.scatter(shap_pca[y_test==1,0], shap_pca[y_test==1,1], color=c_primary, label='Pass (1)', alpha=0.8, s=30)
            ax.legend(frameon=False)
        else:
            sc = ax.scatter(shap_pca[:,0], shap_pca[:,1], c=y_test, cmap=cmap_custom, alpha=0.9, edgecolors='white', linewidth=0.5, s=40)
            cbar = plt.colorbar(sc, ax=ax); cbar.set_label(f'Actual {t_name}', fontweight='bold')
        ax.set_xlabel(f'SHAP PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'SHAP PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        format_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_3_SHAP_PCA.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_3_SHAP_PCA.png'), dpi=600, bbox_inches='tight')
        plt.close(fig)

        # 4. Waterfall — customized with physical units and sample labels
        unit_map = {"LOI": "%", "Tg": "°C", "TENSILE": "MPa", "UL94": "probability"}
        unit = unit_map.get(t_key, "")
        display_names = {"LOI": "LOI", "Tg": "Tg", "TENSILE": "Tensile", "UL94": "UL-94"}
        display = display_names.get(t_key, t_key)

        label_descriptions = {
            "Best_Prediction": "Representative — best prediction",
            "Worst_Prediction": "Outlier — worst prediction",
            f"Highest_{t_name}": "Best real sample — highest value",
            "True_Positive": "True Positive — correctly classified V-0",
            "False_Positive": "False Positive — misclassified as V-0",
        }

        if task == 'reg':
            idx_target = {
                "Best_Prediction": np.argsort(np.abs(residuals))[:3],
                "Worst_Prediction": np.argsort(np.abs(residuals))[-3:][::-1],
                f"Highest_{t_name}": np.argsort(y_test)[-3:][::-1]
            }
        else:
            idx_tp = np.where((y_test==1)&(preds==1))[0]
            idx_fp = np.where((y_test==0)&(preds==1))[0]
            tp_sorted = idx_tp[np.argsort(probs[idx_tp])[::-1]] if len(idx_tp)>0 else []
            fp_sorted = idx_fp[np.argsort(probs[idx_fp])[::-1]] if len(idx_fp)>0 else []
            idx_target = {}
            if len(tp_sorted)>0: idx_target["True_Positive"] = tp_sorted[:3]
            if len(fp_sorted)>0: idx_target["False_Positive"] = fp_sorted[:3]
        for label, indices in idx_target.items():
            for rank, idx in enumerate(indices):
                fig = plt.figure(figsize=(8, 6), dpi=600)
                exp = shap.Explanation(values=shap_values[idx], base_values=expected_value, data=X_test[idx], feature_names=top_k_features)
                shap.plots.waterfall(exp, max_display=8, show=False)
                ax_wf = plt.gca()
                # Enlarge all text
                for txt in ax_wf.texts:
                    t = txt.get_text()
                    if "E[f(X)]" in t or "f(x)" in t:
                        txt.set_text("")
                    else:
                        txt.set_fontsize(11)
                ax_wf.tick_params(labelsize=13)
                ax_wf.xaxis.label.set_fontsize(14)
                for label in ax_wf.get_yticklabels():
                    label.set_fontsize(13)
                    label.set_fontweight("bold")
                for label in ax_wf.get_xticklabels():
                    label.set_fontsize(13)
                for patch in ax_wf.patches:
                    fc = patch.get_facecolor()
                    if fc[0] > 0.5 and fc[2] < 0.5:
                        patch.set_facecolor(c_primary); patch.set_edgecolor(c_primary)
                    elif fc[2] > 0.5 and fc[0] < 0.5:
                        patch.set_facecolor(c_neg); patch.set_edgecolor(c_neg)
                # Customize title: remove shap's "E[f(X)]/f(x)" text, add physical label
                for txt in ax_wf.texts:
                    t = txt.get_text()
                    if "E[f(X)]" in t or "f(x)" in t:
                        txt.set_text("")
                desc = label_descriptions.get(label, label)
                ax_wf.set_title(f"{display} — {desc}", fontsize=14, fontweight="bold", loc="left")
                # Add physical-unit annotation
                actual_val = exp.base_values + np.sum(exp.values)
                ax_wf.text(0.98, 0.02, f"{display} = {actual_val:.2f} {unit}".strip(),
                           transform=ax_wf.transAxes, ha="right", va="bottom",
                           fontsize=11, fontweight="bold", color=c_primary,
                           bbox=dict(facecolor="white", edgecolor=c_primary, boxstyle="round,pad=0.3", linewidth=1.0))
                fig.tight_layout()
                fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_4_Waterfall_{label}_Top{rank+1}.pdf'), bbox_inches='tight')
                fig.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_4_Waterfall_{label}_Top{rank+1}.png'), dpi=600, bbox_inches='tight')
                plt.close(fig)

        # 5. PairGrid
        top_5_features = top_6_features[:5]
        top_5_idx = [top_k_features.index(f) for f in top_5_features]
        df_pg = pd.DataFrame(X_train[:, top_5_idx], columns=top_5_features)
        df_pg[t_name] = y_train
        df_pg.columns = [wrap_labels(c,18) for c in df_pg.columns]
        g = sns.PairGrid(df_pg, height=1.8, aspect=1)
        def corrfunc(x,y,**kwargs):
            cmap = kwargs.pop('custom_cmap')
            mask = ~np.isnan(x)&~np.isnan(y); x,y=x[mask],y[mask]
            if len(x)<2: return
            r,p=pearsonr(x,y)
            stars = "***" if p<=0.001 else "**" if p<=0.01 else "*" if p<=0.05 else ""
            ax=plt.gca(); ax.set_facecolor(cmap((r+1)/2)); ax.patch.set_alpha(0.7)
            ax.annotate(f"{r:.2f}\n{stars}",xy=(0.5,0.5),xycoords=ax.transAxes,ha='center',va='center',fontsize=12,fontweight='bold',color='black')
        def scatter_reg(x,y,**kwargs):
            sc_color=kwargs.pop('scatter_color'); line_color=kwargs.pop('line_color')
            ax=plt.gca()
            sns.regplot(x=x,y=y,ax=ax,scatter_kws={'s':15,'alpha':0.7,'color':sc_color,'edgecolors':'white','linewidths':0.5},
                        line_kws={'color':line_color,'linewidth':2})
            ax.grid(True,linestyle='--',alpha=0.3)
        def dist_plot(x,**kwargs):
            hist_color=kwargs.pop('hist_color')
            ax=plt.gca()
            sns.histplot(x,ax=ax,kde=True,color=hist_color,edgecolor='black',alpha=0.6,bins=20)
            ax.grid(True,linestyle='--',alpha=0.3)
        g.map_upper(corrfunc, custom_cmap=cmap_diverging)
        g.map_lower(scatter_reg, scatter_color=c_primary, line_color=c_neg)
        g.map_diag(dist_plot, hist_color=c_primary)
        for ax in g.axes.flatten():
            if ax: ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        fig_pg = g.fig
        fig_pg.subplots_adjust(right=0.9)
        cbar_ax = fig_pg.add_axes([0.92,0.15,0.02,0.7])
        norm = mpl.colors.Normalize(vmin=-1,vmax=1)
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_diverging), cax=cbar_ax)
        cb.set_label('Pearson r', fontsize=12, fontweight='bold')
        fig_pg.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_5_PairGrid.pdf'), bbox_inches='tight')
        fig_pg.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_5_PairGrid.png'), dpi=600, bbox_inches='tight')
        plt.close(fig_pg)

        fig_bee = plt.figure(figsize=(7,6), dpi=600)
        shap.summary_plot(shap_values, df_test_X, max_display=15, show=False, cmap=cmap_diverging)
        fig_bee.tight_layout()
        fig_bee.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_6_SHAP_Beeswarm.pdf'), bbox_inches='tight')
        fig_bee.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_6_SHAP_Beeswarm.png'), dpi=600, bbox_inches='tight')
        plt.close(fig_bee)

        # 7. Hexbin
        if task == 'reg':
            g_hex = sns.jointplot(x=preds, y=y_test, kind='hex', color=c_primary, gridsize=20, marginal_kws=dict(bins=25, fill=True, color=c_primary))
            ax_j = g_hex.ax_joint
            ax_j.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect')
            ax_j.set_xlabel(f'Predicted {t_name}', fontweight='bold'); ax_j.set_ylabel(f'Actual {t_name}', fontweight='bold')
            ax_j.legend(loc='upper left', frameon=False)
            g_hex.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_7_Hexbin_Density.pdf'), bbox_inches='tight')
            g_hex.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_7_Hexbin_Density.png'), dpi=600, bbox_inches='tight')
            plt.close(g_hex.fig)

        # 8. 2D PDP
        if len(top_6_features)>=5:
            main_feat = top_6_features[0]
            for sec_feat in top_6_features[1:5]:
                fig_2d, ax_2d = plt.subplots(figsize=(7,5), dpi=600)
                x2=df_test_X[main_feat].values; y2=df_test_X[sec_feat].values
                f_idx = list(df_test_X.columns).index(main_feat); z = shap_values[:,f_idx]
                xi=np.linspace(x2.min(),x2.max(),100); yi=np.linspace(y2.min(),y2.max(),100)
                xi,yi=np.meshgrid(xi,yi)
                zi = griddata((x2,y2),z,(xi,yi),method='linear')
                contour = ax_2d.contourf(xi,yi,zi,levels=20,cmap=cmap_custom,alpha=0.85)
                ax_2d.contour(xi,yi,zi,levels=10,colors='black',linewidths=0.5,alpha=0.5,linestyles='dashed')
                cbar=plt.colorbar(contour,ax=ax_2d); cbar.set_label(f'SHAP ({wrap_labels(main_feat,15)})',fontweight='bold')
                ax_2d.set_xlabel(wrap_labels(main_feat,25),fontweight='bold'); ax_2d.set_ylabel(wrap_labels(sec_feat,25),fontweight='bold')
                format_spines(ax_2d)
                fig_2d.tight_layout()
                s1=main_feat.replace('/','_').replace('\\','_'); s2=sec_feat.replace('/','_').replace('\\','_')
                fig_2d.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_8_2D_PDP_{s1}_vs_{s2}.pdf'),bbox_inches='tight')
                fig_2d.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_8_2D_PDP_{s1}_vs_{s2}.png'),dpi=600,bbox_inches='tight')
                plt.close(fig_2d)

def generate_cheminfo_tsne():
    print(">>> t-SNE translated_text")
    TARGETS_T = {
        'LOI': {'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_LOI, 'task': 'reg'},
        'UL94': {'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 'color': COLOR_UL94, 'task': 'cls'},
        'Tg': {'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TG, 'task': 'reg'},
        'TENSILE': {'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TENSILE, 'task': 'reg'}
    }
    for t_key, cfg in TARGETS_T.items():
        pkl = os.path.join(cfg['path'], 'plot_data.pkl')
        if not os.path.exists(pkl):
            continue
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        X_test, y_test = data['X_test'], data['y_test']
        shap_values = data['shap_values']
        top_k_features = data['top_k_features']
        preds = data['preds']
        shap_abs = np.abs(shap_values).mean(axis=0)
        top_15_idx = np.argsort(shap_abs)[-15:][::-1]
        top_15_features = [top_k_features[i] for i in top_15_idx]
        top_15_shap = shap_abs[top_15_idx]

        tsne = TSNE(n_components=2, perplexity=min(30, len(X_test)-1), random_state=42)
        X_tsne = tsne.fit_transform(X_test)

        fig1, ax1 = plt.subplots(figsize=(6,5), dpi=600)
        cmap_target = mcolors.LinearSegmentedColormap.from_list(f"cmap_{t_key}", ["#F2F2F2", cfg['color']])
        if cfg['task'] == 'cls':
            ax1.scatter(X_tsne[y_test==0, 0], X_tsne[y_test==0, 1],
                       color=COLOR_NEG_1, label='Fail', alpha=0.8,
                       edgecolor='white', linewidth=0.5)
            ax1.scatter(X_tsne[y_test==1, 0], X_tsne[y_test==1, 1],
                       color=cfg['color'], label='Pass', alpha=0.8,
                       edgecolor='white', linewidth=0.5)
            ax1.legend(frameon=False)
        else:
            sc1 = ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=y_test,
                             cmap=cmap_target, s=40, alpha=0.9,
                             edgecolors='white', linewidth=0.5)
            plt.colorbar(sc1, ax=ax1).set_label(f'Actual {t_key}', fontweight='bold')
        ax1.set_xlabel('t-SNE 1', fontweight='bold')
        ax1.set_ylabel('t-SNE 2', fontweight='bold')
        format_spines(ax1)
        ax1.grid(True, linestyle='--', alpha=0.3, color=DASH_COLOR)
        fig1.tight_layout()
        fig1.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_tSNE_Class.pdf'), bbox_inches='tight')
        fig1.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_tSNE_Class.png'), dpi=600, bbox_inches='tight')
        plt.close(fig1)

        rgb_primary = mcolors.to_rgb(cfg['color'])
        light_rgb = tuple(1*0.85 + c*0.15 for c in rgb_primary)  # 85%translated_text + 15%translated_text
        light_hex = mcolors.rgb2hex(light_rgb)
        cmap_error = mcolors.LinearSegmentedColormap.from_list(
            f"err_{t_key}", [light_hex, cfg['color']])

        fig2, ax2 = plt.subplots(figsize=(6,5), dpi=600)
        if cfg['task'] == 'cls':
            err = np.abs(y_test - data['probs'])
            sc_err = ax2.scatter(X_tsne[:,0], X_tsne[:,1], c=err,
                                cmap=cmap_error, s=40, alpha=0.9,
                                edgecolors='white', linewidth=0.5)
            plt.colorbar(sc_err, ax=ax2).set_label('Uncertainty', fontweight='bold')
        else:
            ae = np.abs(y_test - preds)
            sc_err = ax2.scatter(X_tsne[:,0], X_tsne[:,1], c=ae,
                                cmap=cmap_error, s=40, alpha=0.9,
                                edgecolors='white', linewidth=0.5)
            plt.colorbar(sc_err, ax=ax2).set_label('Absolute Error', fontweight='bold')
        ax2.set_xlabel('t-SNE 1', fontweight='bold')
        ax2.set_ylabel('t-SNE 2', fontweight='bold')
        format_spines(ax2)
        ax2.grid(True, linestyle='--', alpha=0.3, color=DASH_COLOR)
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_tSNE_Error.pdf'), bbox_inches='tight')
        fig2.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_tSNE_Error.png'), dpi=600, bbox_inches='tight')
        plt.close(fig2)

def generate_advanced_diag():
    print(">>> translated_text、translated_text、translated_text")
    TARGETS_DIAG = {
        'LOI': {'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_LOI, 'color_neg': COLOR_NEG_1, 'task': 'reg'},
        'UL94': {'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 'color': COLOR_UL94, 'color_neg': COLOR_NEG_1, 'task': 'cls'},
        'Tg': {'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TG, 'color_neg': COLOR_NEG_2, 'task': 'reg'},
        'TENSILE': {'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TENSILE, 'color_neg': COLOR_NEG_3, 'task': 'reg'}
    }
    for t_key,cfg in TARGETS_DIAG.items():
        pkl = os.path.join(cfg['path'],'plot_data.pkl')
        if not os.path.exists(pkl): continue
        with open(pkl,'rb') as f: data = pickle.load(f)
        X_test,y_test=data['X_test'],data['y_test']; shap_values=data['shap_values']; top_k_features=data['top_k_features']; preds=data['preds']
        df_test_X=pd.DataFrame(X_test,columns=top_k_features)
        shap_abs=np.abs(shap_values).mean(axis=0); top_15_idx_global=np.argsort(shap_abs)[-15:][::-1]
        top_15_features=[top_k_features[i] for i in top_15_idx_global]

        top_10_features = feature_order_for_plot(t_key, top_k_features, top_15_features, 8)
        top_10_idx = [top_k_features.index(f) for f in top_10_features]

        fig_ridge, ax_ridge = plt.subplots(figsize=(6,8), dpi=300)
        y_offsets = np.arange(len(top_10_features))[::-1] * 1.0
        gmin = shap_values[:, top_10_idx].min()
        gmax = shap_values[:, top_10_idx].max()
        margin = (gmax - gmin) * 0.1
        x_grid = np.linspace(gmin - margin, gmax + margin, 500)

        ax_ridge.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.5, zorder=0)

        for idx, feat in enumerate(top_10_features):
            f_idx = top_k_features.index(feat)
            f_shap = shap_values[:, f_idx]
            try:
                kde = gaussian_kde(f_shap)
                y_dens = kde(x_grid)
                if y_dens.max() > 0:
                    y_dens = (y_dens / y_dens.max()) * 0.85
            except:
                y_dens = np.zeros_like(x_grid)
            y_plot = y_dens + y_offsets[idx]
            ax_ridge.plot(x_grid, y_plot, color=cfg['color'], lw=1.5, zorder=idx + 1)
            ax_ridge.fill_between(x_grid, y_offsets[idx], y_plot, color=cfg['color'], alpha=0.6, zorder=idx + 1)

        ax_ridge.set_yticks(y_offsets + 0.1)
        ax_ridge.set_yticklabels([wrap_labels(feat) for feat in top_10_features],
                                 fontweight='bold', fontsize=13)
        ax_ridge.tick_params(axis='y', which='both', length=0)
        ax_ridge.spines['left'].set_visible(False)

        ax_ridge.set_xlabel(f'SHAP Value (Impact on {t_key})', fontweight='bold', fontsize=14)
        ax_ridge.set_xlim(gmin - margin, gmax + margin)
        format_spines(ax_ridge)  # translated_text、translated_text

        fig_ridge.tight_layout()
        fig_ridge.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_SHAP_Ridgeline.pdf'), bbox_inches='tight')
        fig_ridge.savefig(os.path.join(OUTPUT_DIR, f'{t_key}_SHAP_Ridgeline.png'), dpi=600, bbox_inches='tight')
        plt.close(fig_ridge)

        if cfg['task']=='cls':
            probs=data['probs']
            fig_cal,ax_cal=plt.subplots(figsize=(6,6),dpi=600)
            prob_true,prob_pred=calibration_curve(y_test,probs,n_bins=10,strategy='uniform')
            ax_cal.plot([0,1],[0,1],linestyle='--',color=DASH_COLOR,label='Perfect')
            ax_cal.plot(prob_pred,prob_true,marker='s',color=cfg['color'],lw=2,markersize=8,label='Model')
            ax_cal.set_xlabel('Mean Predicted Probability',fontweight='bold')
            ax_cal.set_ylabel('Fraction of Positives',fontweight='bold')
            ax_cal.set_xlim([-0.05,1.05]); ax_cal.set_ylim([-0.05,1.05])
            ax_cal.grid(True,linestyle=':',alpha=0.6)
            ax_cal.legend(loc='lower right',frameon=False)
            format_spines(ax_cal)
            fig_cal.tight_layout()
            fig_cal.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_Calibration.pdf'),bbox_inches='tight')
            fig_cal.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_Calibration.png'),dpi=600,bbox_inches='tight')
            plt.close(fig_cal)

        if cfg['task']=='reg':
            pred_df=pd.read_csv(os.path.join(cfg['path'],'predictions.csv'))
            residuals=pred_df['Residual'].values
            main_feat=top_10_features[0]; x_vals=df_test_X[main_feat].values
            cmap_res = mcolors.LinearSegmentedColormap.from_list('res', ['white', cfg['color']])
            fig_res,ax_res=plt.subplots(figsize=(7,6),dpi=600)
            sc=ax_res.scatter(x_vals,residuals,s=(y_test-y_test.min()+1)**1.5*2,
                              c=np.abs(residuals),cmap=cmap_res,alpha=0.7,edgecolors='white',linewidth=0.5)
            ax_res.axhline(0,color=DASH_COLOR,linestyle='--',lw=2)
            sns.regplot(x=x_vals,y=residuals,scatter=False,lowess=False,ax=ax_res,
                        color=cfg['color_neg'],line_kws={'linestyle':'-.','lw':2})
            cbar=plt.colorbar(sc,ax=ax_res); cbar.set_label('Absolute Error',fontweight='bold')
            ax_res.set_xlabel(f'{wrap_labels(main_feat)} (Top Prior)',fontweight='bold')
            ax_res.set_ylabel('Prediction Residual',fontweight='bold')
            format_spines(ax_res)
            fig_res.tight_layout()
            fig_res.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_Residual_Diagnosis.pdf'),bbox_inches='tight')
            fig_res.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_Residual_Diagnosis.png'),dpi=600,bbox_inches='tight')
            plt.close(fig_res)

            abs_errors=np.abs(residuals); sorted_err=np.sort(abs_errors)
            cdf_err=np.arange(1,len(sorted_err)+1)/len(sorted_err)
            fig_cdf,ax_cdf=plt.subplots(figsize=(6,5),dpi=600)
            ax_cdf.plot(sorted_err,cdf_err,color=cfg['color'],lw=2.5)
            ax_cdf.fill_between(sorted_err,0,cdf_err,color=cfg['color'],alpha=0.15)
            err_80=sorted_err[int(len(sorted_err)*0.8)]
            ax_cdf.axhline(0.8,color=DASH_COLOR,linestyle='--',lw=1)
            ax_cdf.axvline(err_80,color=DASH_COLOR,linestyle='--',lw=1)
            ax_cdf.scatter([err_80],[0.8],color=cfg['color'],s=60,zorder=5)
            ax_cdf.text(err_80+0.02*(sorted_err.max()-sorted_err.min()),0.75,
                        f"80% samples\nError ≤ {err_80:.2f}",ha='left',va='top',
                        fontsize=12,fontweight='bold',color=cfg['color'])
            ax_cdf.set_xlim([0,sorted_err.max()]); ax_cdf.set_ylim([0,1.05])
            ax_cdf.set_xlabel(f'Absolute Error in {t_key}',fontweight='bold')
            ax_cdf.set_ylabel('Cumulative Probability (CDF)',fontweight='bold')
            ax_cdf.grid(True,linestyle=':',alpha=0.6)
            format_spines(ax_cdf)
            fig_cdf.tight_layout()
            fig_cdf.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_Error_CDF.pdf'),bbox_inches='tight')
            fig_cdf.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_Error_CDF.png'),dpi=600,bbox_inches='tight')
            plt.close(fig_cdf)

def generate_sensitivity_decision():
    print(">>> translated_text")
    TARGETS_SEN = {
        'LOI': {'path': r'C:\Users\WINDOWS\Desktop\GNN\LOI\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_LOI, 'task': 'reg'},
        'UL94': {'path': r'C:\Users\WINDOWS\Desktop\GNN\94\Result_Final_TabICL_Interpret_Cls\saved_data', 'color': COLOR_UL94, 'task': 'cls'},
        'Tg': {'path': r'C:\Users\WINDOWS\Desktop\GNN\Tg\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TG, 'task': 'reg'},
        'TENSILE': {'path': r'C:\Users\WINDOWS\Desktop\GNN\TENSILE\Result_Final_TabICL_Interpret\saved_data', 'color': COLOR_TENSILE, 'task': 'reg'}
    }
    for t_key,cfg in TARGETS_SEN.items():
        pkl=os.path.join(cfg['path'],'plot_data.pkl')
        if not os.path.exists(pkl): continue
        with open(pkl,'rb') as f: data=pickle.load(f)
        X_test,y_test=data['X_test'],data['y_test']; shap_values=data['shap_values']
        top_k_features=data['top_k_features']; preds=data['preds']
        df_test_X=pd.DataFrame(X_test,columns=top_k_features)
        shap_abs=np.abs(shap_values).mean(axis=0); top_15_idx=np.argsort(shap_abs)[-15:][::-1]
        top_15_features=[top_k_features[i] for i in top_15_idx]

        ordered_features = feature_order_for_plot(t_key, top_k_features, top_15_features, 8)

        for feat in ordered_features:
            f_idx=top_k_features.index(feat); f_vals=df_test_X[feat].values; s_vals=shap_values[:,f_idx]
            try:
                q=pd.qcut(f_vals,q=3,duplicates='drop')
                if len(q.categories)==3:
                    bins=pd.qcut(f_vals,q=3,labels=['Low','Medium','High'],duplicates='drop')
                elif len(q.categories)==2:
                    bins=pd.qcut(f_vals,q=2,labels=['Low','High'],duplicates='drop')
                else: bins=np.where(f_vals>np.median(f_vals),'High','Low')
            except: bins=np.where(f_vals>np.median(f_vals),'High','Low')
            df_imp=pd.DataFrame({'Level':bins,'SHAP':s_vals})
            order=['Low','Medium','High'] if 'Medium' in df_imp['Level'].values else ['Low','High']
            fig_q,ax_q=plt.subplots(figsize=(6,5),dpi=600)
            sns.boxplot(data=df_imp,x='Level',y='SHAP',order=order,color='white',ax=ax_q,width=0.5,
                        boxprops={'edgecolor':cfg['color'],'linewidth':2})
            sns.stripplot(data=df_imp,x='Level',y='SHAP',order=order,color=cfg['color'],alpha=0.6,ax=ax_q,jitter=True)
            ax_q.axhline(0,color='black',linestyle='--',lw=1.5,alpha=0.7)
            ax_q.set_xlabel("")
            ax_q.set_ylabel(f"SHAP on {t_key}", fontweight="bold", fontsize=18)
            ax_q.set_title(wrap_labels(feat), fontsize=17, fontweight="bold", pad=8)
            ax_q.tick_params(labelsize=15)
            for s in ax_q.spines.values():
                s.set_visible(True)
                s.set_linewidth(1.3)
                s.set_color('#333333')
            fig_q.tight_layout()
            sf=feat.replace('/','_').replace('\\','_')
            fig_q.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_QuantileImpact_{sf}.pdf'),bbox_inches='tight')
            fig_q.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_QuantileImpact_{sf}.png'),dpi=600,bbox_inches='tight')
            plt.close(fig_q)

        if cfg['task']=='reg':
            idx_h=np.argmax(y_test); idx_l=np.argmin(y_test); samples=[idx_l,idx_h]
        else:
            probs=data['probs']; idx_p=np.argmax(probs); idx_f=np.argmin(probs); samples=[idx_f,idx_p]
        wrapped=[wrap_labels(f) for f in top_k_features]
        if cfg['task']=='cls': exp_val=data['expected_value']
        else:
            inf_base=preds-np.sum(shap_values,axis=1); exp_val=np.mean(inf_base)
        fig_d,ax_d=plt.subplots(figsize=(7,6),dpi=600)
        shap.decision_plot(exp_val,shap_values[samples],features=df_test_X.iloc[samples],
                           feature_names=wrapped,feature_display_range=slice(None,-16,-1),show=False)
        plt.gca().set_title('')
        fig_d.tight_layout()
        fig_d.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_DecisionPath.pdf'),bbox_inches='tight')
        fig_d.savefig(os.path.join(OUTPUT_DIR,f'{t_key}_DecisionPath.png'),dpi=600,bbox_inches='tight')
        plt.close(fig_d)

if __name__ == '__main__':
    print("="*70)
    print(">>> translated_text TabICL translated_text (translated_text + translated_text)")
    print(f">>> translated_text: {OUTPUT_DIR}")
    generate_ul94_cdf()
    generate_feature_network()
    generate_advanced_shap_corr()
    generate_cheminfo_tsne()
    generate_advanced_diag()
    generate_sensitivity_decision()
    print("\n🎉 translated_text！")
