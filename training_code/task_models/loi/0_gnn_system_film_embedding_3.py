import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, RegressorMixin
import random
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc
import warnings
import copy

warnings.filterwarnings('ignore')

# ==========================================
# ==========================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_dir = f"Result_NoGNNFilter_SysFiLM_32D_Strict10CV_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"========================================")
print(f"🚀 translated_text！translated_text: {output_dir}/")
print(f"========================================\n")

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.dpi'] = 600

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# ==========================================
print("=== 1. translated_text ===")
df = pd.read_csv("EP+FR+CURING_SMILES+translated_text_DATASET.csv")

def extract_physical_logic(df):
    """translated_text：translated_text"""
    temp_cols = [f'Curing_Tem{i}' for i in range(1, 10)]
    time_cols = [f'Curing_Time{i}' for i in range(1, 10)]
    
    df['T_max'] = df[temp_cols].max(axis=1)
    
    df['t_total'] = df[time_cols].sum(axis=1)
    
    thermal_sum = 0
    for tem_col, time_col in zip(temp_cols, time_cols):
        thermal_sum += df[tem_col].fillna(0) * df[time_col].fillna(0)
    df['Q_thermal'] = thermal_sum
    
    cols_to_drop = temp_cols + time_cols
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    df['T_max'] = df['T_max'].fillna(0)
    df['t_total'] = df['t_total'].fillna(0)
    df['Q_thermal'] = df['Q_thermal'].fillna(0)
    
    return df

print("translated_text...")
df = extract_physical_logic(df)

valid_indices = []
for idx, row in df.iterrows():
    mol1 = Chem.MolFromSmiles(row['EP_SMILES'])
    mol2 = Chem.MolFromSmiles(row['FR_SMILES'])
    mol3 = Chem.MolFromSmiles(row['CURING_SMILES'])
    if mol1 and mol2 and mol3:
        valid_indices.append(idx)

df_valid = df.iloc[valid_indices].reset_index(drop=True)
print(f"translated_text: {len(df_valid)}")

num_features_raw = df_valid.drop(columns=['EP_SMILES', 'FR_SMILES', 'CURING_SMILES', 'LOI']).values
cond_input_dim = num_features_raw.shape[1]

try:
    fr_wt = df_valid['Flame_retardant_AdditionAmount(wt%)'].fillna(0).values
    cur_wt = df_valid['Curing_agent_AdditionAmount(wt%)'].fillna(0).values
    ep_wt = 100.0 - fr_wt - cur_wt
    wt_matrix_valid = np.vstack([ep_wt, fr_wt, cur_wt]).T / 100.0
except KeyError:
    wt_matrix_valid = np.ones((len(df_valid), 3)) / 3.0

indices = np.arange(len(df_valid))
y = df_valid['LOI'].values

print(f"translated_text（translated_text）: {cond_input_dim}")
print("translated_text：")
print(f"• translated_text {cond_input_dim} translated_text")
print(f"• translated_text Curing_Tem/Curing_Time translated_text")
print(f"• translated_text T_max, t_total, Q_thermal translated_text")

# ==========================================
# ==========================================
def create_graph_data(smiles_list):
    """translated_textSMILEStranslated_text"""
    dataset = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: 
            continue
        try: 
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except: 
            pass
        
        atom_features = []
        for atom in mol.GetAtoms():
            try:
                gc = atom.GetDoubleProp('_GasteigerCharge')
                if np.isnan(gc) or np.isinf(gc): 
                    gc = 0.0
            except: 
                gc = 0.0
            atomic_num = atom.GetAtomicNum()
            features = [
                atomic_num, atom.GetDegree(), atom.GetHybridization().real, 
                int(atom.GetIsAromatic()), atom.GetFormalCharge(), 
                atom.GetNumExplicitHs(), atom.GetNumImplicitHs(), atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(), atom.GetIsotope(), atom.GetChiralTag().real,
                mol.GetRingInfo().NumAtomRings(atom.GetIdx()), int(atom.IsInRing()), 
                int(atom.IsInRingSize(6)), atom.GetTotalValence(), len(atom.GetNeighbors()), 
                len(atom.GetBonds()), atom.GetMass()/100, gc, int(atomic_num in [7, 8]), 
                int(atomic_num in [15, 16]), int(atomic_num in [9, 17]), int(atom.IsInRingSize(5)),
            ]
            atom_features.append(features)
            
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[start, end], [end, start]])
            bf = [bond.GetBondTypeAsDouble(), bond.GetIsConjugated(), 
                  bond.IsInRing(), bond.GetStereo(), 
                  int(bond.GetBondType() == Chem.BondType.AROMATIC)]
            edge_attr.extend([bf, bf])
        
        if len(edge_index) == 0:
            for i in range(mol.GetNumAtoms()):
                edge_index.append([i, i])
                edge_attr.append([0.0, 0, 0, 0, 0])
        
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return dataset

# ==========================================
# ==========================================
class FiLMGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, edge_dim=5):
        super().__init__()
        nn_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.BatchNorm1d(out_dim), 
            nn.ReLU(), 
            nn.Linear(out_dim, out_dim)
        )
        self.conv = GINEConv(nn_mlp, edge_dim=edge_dim)
        self.film_gen = nn.Linear(cond_dim, out_dim * 2)
        self.act = nn.LeakyReLU()
        
    def forward(self, x, edge_index, edge_attr, condition_vector):
        out = self.conv(x, edge_index, edge_attr)
        gamma, beta = torch.chunk(self.film_gen(condition_vector), 2, dim=-1)
        return gamma * self.act(out) + beta

class SystemFiLMGNN(nn.Module):
    def __init__(self, node_dim=23, edge_dim=5, hidden_dim=64, sys_dim=32, cond_input_dim=100):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer1 = FiLMGNNLayer(node_dim, hidden_dim, hidden_dim, edge_dim)
        self.layer2 = FiLMGNNLayer(hidden_dim, hidden_dim, hidden_dim, edge_dim)
        self.layer3 = FiLMGNNLayer(hidden_dim, sys_dim, hidden_dim, edge_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(sys_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def extract_molecule(self, batch_data, cond_gating):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        x = self.layer1(x, edge_index, edge_attr, cond_gating[batch])
        x = self.layer2(x, edge_index, edge_attr, cond_gating[batch])
        x = self.layer3(x, edge_index, edge_attr, cond_gating[batch])
        return global_mean_pool(x, batch)

    def forward(self, ep_data, fr_data, curing_data, macro_conditions, wt_fractions):
        cond_gating = self.cond_mlp(macro_conditions)
        ep_emb = self.extract_molecule(ep_data, cond_gating)
        fr_emb = self.extract_molecule(fr_data, cond_gating)
        cur_emb = self.extract_molecule(curing_data, cond_gating)
        
        wt_ep = wt_fractions[:, 0].unsqueeze(1)
        wt_fr = wt_fractions[:, 1].unsqueeze(1)
        wt_cur = wt_fractions[:, 2].unsqueeze(1)
        
        system_embedding = (ep_emb * wt_ep) + (fr_emb * wt_fr) + (cur_emb * wt_cur)
        return self.regressor(torch.cat([system_embedding, cond_gating], dim=1)).squeeze()
        
    def extract_system_embedding(self, ep_data, fr_data, curing_data, macro_conditions, wt_fractions):
        cond_gating = self.cond_mlp(macro_conditions)
        ep_emb = self.extract_molecule(ep_data, cond_gating)
        fr_emb = self.extract_molecule(fr_data, cond_gating)
        cur_emb = self.extract_molecule(curing_data, cond_gating)
        
        wt_ep = wt_fractions[:, 0].unsqueeze(1)
        wt_fr = wt_fractions[:, 1].unsqueeze(1)
        wt_cur = wt_fractions[:, 2].unsqueeze(1)
        
        return (ep_emb * wt_ep) + (fr_emb * wt_fr) + (cur_emb * wt_cur)

# ==========================================
# ==========================================
def create_system_dataloader(ep, fr, cur, cond, wt, y, batch_size=32, shuffle=True):
    """translated_text"""
    class SystemDataset(torch.utils.data.Dataset):
        def __init__(self, ep, fr, cur, cond, wt, y):
            self.ep, self.fr, self.cur = ep, fr, cur
            self.cond = torch.tensor(cond, dtype=torch.float32)
            self.wt = torch.tensor(wt, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): 
            return len(self.ep)
        def __getitem__(self, i): 
            return self.ep[i], self.fr[i], self.cur[i], self.cond[i], self.wt[i], self.y[i]
    
    return DataLoader(SystemDataset(ep, fr, cur, cond, wt, y), batch_size=batch_size, shuffle=shuffle)

def extract_film_gnn_features(model, ep, fr, cur, cond, wt, device='cpu'):
    """translated_textGNNtranslated_text"""
    model.eval()
    model.to(device)
    loader = create_system_dataloader(ep, fr, cur, cond, wt, np.zeros(len(ep)), batch_size=32, shuffle=False)
    all_features = []
    with torch.no_grad():
        for ep_b, fr_b, cur_b, cond_b, wt_b, _ in loader:
            ep_b, fr_b, cur_b = ep_b.to(device), fr_b.to(device), cur_b.to(device)
            cond_b, wt_b = cond_b.to(device), wt_b.to(device)
            features = model.extract_system_embedding(ep_b, fr_b, cur_b, cond_b, wt_b)
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)

# ==========================================
# ==========================================
print("\n=== 5. translated_textXGBoosttranslated_text（translated_text） ===")

X_train_idx_global, X_test_idx_global, y_train_global, y_test_global = train_test_split(
    indices, y, test_size=0.2, random_state=42, 
    stratify=pd.qcut(y, q=5) if len(y) > 100 else None
)

scaler_global = StandardScaler()
X_train_num_global = scaler_global.fit_transform(num_features_raw[X_train_idx_global])

xgb_global = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_estimators=200, 
    tree_method='hist'
)
xgb_global.fit(X_train_num_global, y_train_global)

num_feature_names = df_valid.drop(columns=['EP_SMILES', 'FR_SMILES', 'CURING_SMILES', 'LOI']).columns.tolist()
feature_importance_df = pd.DataFrame({
    'Feature_Name': num_feature_names,
    'XGB_Importance': xgb_global.feature_importances_
}).sort_values(by='XGB_Importance', ascending=False)

top_k = min(450, len(feature_importance_df))
selected_global = feature_importance_df.head(top_k)

fig, ax = plt.subplots(figsize=(10, 8))
top_20 = selected_global.head(20)
ax.barh(range(20), top_20['XGB_Importance'].values[::-1])
ax.set_yticks(range(20))
ax.set_yticklabels(top_20['Feature_Name'].values[::-1])
ax.set_title('Top 20 Numerical Features (Global XGBoost)')
ax.set_xlabel('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Global_Top20_Features.png'), dpi=300)
plt.close()

selected_global.to_csv(os.path.join(output_dir, f'Global_Top_{top_k}_Features.csv'), index=False)
print(f"translated_text，translated_textTop {top_k}translated_text")

# ==========================================
# ==========================================
print("\n" + "="*60)
print("translated_text (10-Fold CV)")
print("【translated_text】translated_text：GNNtranslated_text，translated_text，translated_text！")
print("="*60)

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
    print("✓ LightGBM translated_text")
except ImportError:
    LGBM_AVAILABLE = False
    print("✗ LightGBM translated_text")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
    print("✓ CatBoost translated_text")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("✗ CatBoost translated_text")

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
    print("✓ TabPFN translated_text")
except ImportError:
    TABPFN_AVAILABLE = False
    print("✗ TabPFN translated_text")

try:
    from tabicl import TabICLRegressor
    TABICL_AVAILABLE = True
    print("✓ TabICL translated_text")
except ImportError:
    TABICL_AVAILABLE = False
    print("✗ TabICL translated_text")

class ELMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=150):
        self.n_hidden = n_hidden
    
    def fit(self, X, y):
        np.random.seed(42)
        self.input_weights = np.random.randn(X.shape[1], self.n_hidden)
        self.biases = np.random.randn(self.n_hidden)
        H = np.tanh(np.dot(X, self.input_weights) + self.biases)
        self.output_weights = np.dot(np.linalg.pinv(H), y)
        return self
    
    def predict(self, X):
        H = np.tanh(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights)

models = {
    'XGBoost': xgb.XGBRegressor(
        objective='reg:squarederror', 
        tree_method='hist', 
        random_state=42, 
        n_estimators=250, 
        learning_rate=0.05, 
        max_depth=6
    ),
    
    'RandomForest': RandomForestRegressor(
        random_state=42, 
        n_estimators=200,
        max_depth=10,
        n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesRegressor(
        random_state=42, 
        n_estimators=200,
        max_depth=10,
        n_jobs=-1
    ),
    'GBR': GradientBoostingRegressor(
        random_state=42, 
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6
    ),
    
    'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
    'KernelRidge': KernelRidge(kernel='rbf', alpha=1.0),
    
    'MLP': MLPRegressor(
        random_state=42, 
        hidden_layer_sizes=(128, 64), 
        max_iter=500, 
        early_stopping=True
    ),
    
    'BayesianRidge': BayesianRidge(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'PLS': PLSRegression(n_components=5),
    
    'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
    
    'ELM': ELMRegressor(n_hidden=150)
}

if LGBM_AVAILABLE:
    models['LightGBM'] = LGBMRegressor(
        random_state=42, 
        n_estimators=250, 
        learning_rate=0.05, 
        max_depth=6, 
        verbose=-1,
        n_jobs=-1
    )

if CATBOOST_AVAILABLE:
    models['CatBoost'] = CatBoostRegressor(
        random_state=42, 
        iterations=250, 
        learning_rate=0.05, 
        depth=6, 
        verbose=False,
        task_type='CPU'
    )

print(f"\ntranslated_text {len(models)} translated_text")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\ntranslated_text: {device}")

for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
    print(f"\n" + "-"*40)
    print(f"[Fold {fold+1}/10] translated_text...")
    print(f"translated_text: {len(train_idx)} translated_text, translated_text: {len(test_idx)} translated_text")
    print("-"*40)
    
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    # -----------------------------------------------------
    # -----------------------------------------------------
    scaler_fold = StandardScaler()
    X_train_num_fold = scaler_fold.fit_transform(num_features_raw[train_idx])
    X_test_num_fold = scaler_fold.transform(num_features_raw[test_idx])
    
    xgb_selector_fold = xgb.XGBRegressor(
        random_state=42+fold, 
        n_estimators=200, 
        tree_method='hist'
    )
    xgb_selector_fold.fit(X_train_num_fold, y_train_fold)
    top_indices_fold = np.argsort(xgb_selector_fold.feature_importances_)[::-1][:top_k]
    
    X_train_num_selected = X_train_num_fold[:, top_indices_fold]
    X_test_num_selected = X_test_num_fold[:, top_indices_fold]
    
    if fold == 0:  # translated_text
        selected_features = [num_feature_names[i] for i in top_indices_fold[:10]]
        print(f"translated_textTop 10translated_text: {selected_features}")
    
    # -----------------------------------------------------
    # -----------------------------------------------------
    gnn_train_sub_idx, gnn_val_sub_idx, y_gnn_train_sub, y_gnn_val_sub = train_test_split(
        train_idx, y_train_fold, test_size=0.1, random_state=42+fold
    )
    
    relative_gnn_train_idx = [np.where(train_idx == i)[0][0] for i in gnn_train_sub_idx]
    relative_gnn_val_idx = [np.where(train_idx == i)[0][0] for i in gnn_val_sub_idx]
    
    cond_gnn_train = X_train_num_fold[relative_gnn_train_idx]
    cond_gnn_val = X_train_num_fold[relative_gnn_val_idx]
    
    wt_gnn_train = wt_matrix_valid[gnn_train_sub_idx]
    wt_gnn_val = wt_matrix_valid[gnn_val_sub_idx]
    
    print(f"GNNtranslated_text: {len(gnn_train_sub_idx)} translated_text, GNNtranslated_text: {len(gnn_val_sub_idx)} translated_text")

    print("translated_text...")
    ep_g_tr = create_graph_data(df_valid.iloc[gnn_train_sub_idx]['EP_SMILES'].tolist())
    fr_g_tr = create_graph_data(df_valid.iloc[gnn_train_sub_idx]['FR_SMILES'].tolist())
    cu_g_tr = create_graph_data(df_valid.iloc[gnn_train_sub_idx]['CURING_SMILES'].tolist())
    
    ep_g_va = create_graph_data(df_valid.iloc[gnn_val_sub_idx]['EP_SMILES'].tolist())
    fr_g_va = create_graph_data(df_valid.iloc[gnn_val_sub_idx]['FR_SMILES'].tolist())
    cu_g_va = create_graph_data(df_valid.iloc[gnn_val_sub_idx]['CURING_SMILES'].tolist())

    # -----------------------------------------------------
    # -----------------------------------------------------
    print(">>> translated_text GNN translated_text...")
    
    fold_model = SystemFiLMGNN(cond_input_dim=cond_input_dim, sys_dim=32).to(device)
    optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    fold_train_loader = create_system_dataloader(
        ep_g_tr, fr_g_tr, cu_g_tr, cond_gnn_train, wt_gnn_train, y_gnn_train_sub, 
        batch_size=32, shuffle=True
    )
    fold_val_loader = create_system_dataloader(
        ep_g_va, fr_g_va, cu_g_va, cond_gnn_val, wt_gnn_val, y_gnn_val_sub, 
        batch_size=64, shuffle=False
    )
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(200):
        fold_model.train()
        train_loss = 0.0
        for ep_b, fr_b, cur_b, cond_b, wt_b, labels in fold_train_loader:
            ep_b, fr_b, cur_b = ep_b.to(device), fr_b.to(device), cur_b.to(device)
            cond_b, wt_b, labels = cond_b.to(device), wt_b.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = fold_model(ep_b, fr_b, cur_b, cond_b, wt_b)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            
        fold_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ep_b, fr_b, cur_b, cond_b, wt_b, labels in fold_val_loader:
                ep_b, fr_b, cur_b = ep_b.to(device), fr_b.to(device), cur_b.to(device)
                cond_b, wt_b, labels = cond_b.to(device), wt_b.to(device), labels.to(device)
                preds = fold_model(ep_b, fr_b, cur_b, cond_b, wt_b)
                val_loss += criterion(preds, labels).item() * len(labels)
                
        avg_train_loss = train_loss / len(gnn_train_sub_idx)
        avg_val_loss = val_loss / len(gnn_val_sub_idx)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(fold_model.state_dict())
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            
        if patience_counter >= 20:  # translated_text
            print(f"  translated_text epoch {epoch+1}, translated_text: {best_loss:.4f} (epoch {best_epoch})")
            break
            
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: translated_text={avg_train_loss:.4f}, translated_text={avg_val_loss:.4f}")
    
    fold_model.load_state_dict(best_model_state)
    print(f">>> translated_text GNN translated_text，translated_textepoch: {best_epoch}, translated_text: {best_loss:.4f}")

    # -----------------------------------------------------
    # -----------------------------------------------------
    print("translated_text GNN translated_text...")
    
    ep_full_tr = create_graph_data(df_valid.iloc[train_idx]['EP_SMILES'].tolist())
    fr_full_tr = create_graph_data(df_valid.iloc[train_idx]['FR_SMILES'].tolist())
    cu_full_tr = create_graph_data(df_valid.iloc[train_idx]['CURING_SMILES'].tolist())
    wt_full_tr = wt_matrix_valid[train_idx]
    
    ep_te = create_graph_data(df_valid.iloc[test_idx]['EP_SMILES'].tolist())
    fr_te = create_graph_data(df_valid.iloc[test_idx]['FR_SMILES'].tolist())
    cu_te = create_graph_data(df_valid.iloc[test_idx]['CURING_SMILES'].tolist())
    wt_te = wt_matrix_valid[test_idx]
    
    X_train_gnn_fold = extract_film_gnn_features(fold_model, ep_full_tr, fr_full_tr, cu_full_tr, X_train_num_fold, wt_full_tr, device)
    X_test_gnn_fold = extract_film_gnn_features(fold_model, ep_te, fr_te, cu_te, X_test_num_fold, wt_te, device)
    
    X_train_final = np.concatenate([X_train_gnn_fold, X_train_num_selected], axis=1)
    X_test_final = np.concatenate([X_test_gnn_fold, X_test_num_selected], axis=1)
    
    print(f">>> translated_text: GNN(32) + Num({top_k}) = translated_text {X_train_final.shape[1]}")
    
    # -----------------------------------------------------
    # -----------------------------------------------------
    fold_results = []
    
    for name, clf in models.items():
        if hasattr(clf, 'random_state'): 
            clf.random_state = 42 + fold
        
        clf_copy = copy.deepcopy(clf)
        
        try:
            clf_copy.fit(X_train_final, y_train_fold)
            preds = clf_copy.predict(X_test_final)
            
            r2 = r2_score(y_test_fold, preds)
            rmse = np.sqrt(mean_squared_error(y_test_fold, preds))
            mae = mean_absolute_error(y_test_fold, preds)
            
            fold_results.append({
                'Model': name, 
                'Fold': fold + 1, 
                'Test_R2': r2, 
                'Test_RMSE': rmse,
                'Test_MAE': mae
            })
            print(f"[{name.ljust(12)}] R²: {r2:.4f}, RMSE: {rmse:.4f}")
            
        except Exception as e:
            print(f"[{name.ljust(12)}] translated_text: {e}")
            fold_results.append({
                'Model': name, 
                'Fold': fold + 1, 
                'Test_R2': np.nan, 
                'Test_RMSE': np.nan,
                'Test_MAE': np.nan
            })

    if TABPFN_AVAILABLE:
        try:
            print("translated_text TabPFN...")
            tabpfn_model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
            tabpfn_model.fit(X_train_final, y_train_fold)
            preds_tabpfn = tabpfn_model.predict(X_test_final)
            
            fold_results.append({
                'Model': 'TabPFN', 
                'Fold': fold + 1, 
                'Test_R2': r2_score(y_test_fold, preds_tabpfn), 
                'Test_RMSE': np.sqrt(mean_squared_error(y_test_fold, preds_tabpfn)),
                'Test_MAE': mean_absolute_error(y_test_fold, preds_tabpfn)
            })
            print(f"[TabPFN      ] R²: {r2_score(y_test_fold, preds_tabpfn):.4f}")
            del tabpfn_model
            gc.collect()
        except Exception as e: 
            print(f"[TabPFN translated_text] {e}")

    if TABICL_AVAILABLE:
        try:
            print("translated_text TabICL...")
            tabicl_model = TabICLRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', kv_cache=True)
            tabicl_model.fit(X_train_final, y_train_fold)
            preds_tabicl = tabicl_model.predict(X_test_final)
            
            fold_results.append({
                'Model': 'TabICL', 
                'Fold': fold + 1, 
                'Test_R2': r2_score(y_test_fold, preds_tabicl), 
                'Test_RMSE': np.sqrt(mean_squared_error(y_test_fold, preds_tabicl)),
                'Test_MAE': mean_absolute_error(y_test_fold, preds_tabicl)
            })
            print(f"[TabICL      ] R²: {r2_score(y_test_fold, preds_tabicl):.4f}")
            del tabicl_model
            gc.collect()
        except Exception as e: 
            print(f"[TabICL translated_text] {e}")
            
    all_results.extend(fold_results)
    
    print("translated_text...")
    del fold_model, optimizer, scheduler, criterion, fold_train_loader, fold_val_loader
    del ep_g_tr, fr_g_tr, cu_g_tr, ep_g_va, fr_g_va, cu_g_va
    del ep_full_tr, fr_full_tr, cu_full_tr, ep_te, fr_te, cu_te
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# ==========================================
print("\n" + "="*60)
print("translated_text...")
print("="*60)

results_df = pd.DataFrame(all_results)

results_df.to_csv(os.path.join(output_dir, 'CV_All_Folds_Results_Strict.csv'), index=False)

summary = results_df.groupby('Model').agg(
    Mean_Test_R2=('Test_R2', 'mean'),
    Std_Test_R2=('Test_R2', 'std'),
    Mean_Test_RMSE=('Test_RMSE', 'mean'),
    Std_Test_RMSE=('Test_RMSE', 'std'),
    Mean_Test_MAE=('Test_MAE', 'mean'),
    Std_Test_MAE=('Test_MAE', 'std'),
    Count_Valid=('Test_R2', lambda x: x.notna().sum())  # translated_text
).sort_values(by='Mean_Test_R2', ascending=False)

summary = summary[summary['Count_Valid'] >= 5]
summary.to_csv(os.path.join(output_dir, 'CV_Statistical_Summary_Strict.csv'))

detailed_performance = results_df.pivot_table(
    index='Model', 
    columns='Fold', 
    values=['Test_R2', 'Test_RMSE'],
    aggfunc='first'
)
detailed_performance.to_csv(os.path.join(output_dir, 'CV_Detailed_Performance_Matrix.csv'))

best_results = results_df.loc[results_df.groupby('Model')['Test_R2'].idxmax()].sort_values(by='Test_R2', ascending=False)
best_results.to_csv(os.path.join(output_dir, 'CV_Best_Per_Fold_Results.csv'), index=False)

print("\n" + "="*60)
print(f"📊 10translated_text (translated_text)")
print(f"  translated_text {output_dir}")
print("="*60)

print("\ntranslated_text15translated_text：")
print("="*60)
top_15 = summary.head(15).copy()
top_15_formatted = top_15.copy()
for col in ['Mean_Test_R2', 'Std_Test_R2', 'Mean_Test_RMSE', 'Std_Test_RMSE', 'Mean_Test_MAE', 'Std_Test_MAE']:
    if col.startswith('Mean_'):
        top_15_formatted[col] = top_15_formatted[col].apply(lambda x: f"{x:.4f}")
    else:
        top_15_formatted[col] = top_15_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

print(top_15_formatted[['Mean_Test_R2', 'Std_Test_R2', 'Mean_Test_RMSE', 'Mean_Test_MAE', 'Count_Valid']].to_string())

print("\ntranslated_text...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sorted_models = summary.index.tolist()
r2_means = [summary.loc[m, 'Mean_Test_R2'] for m in sorted_models]
r2_stds = [summary.loc[m, 'Std_Test_R2'] for m in sorted_models]

axes[0, 0].barh(range(len(sorted_models)), r2_means, xerr=r2_stds, capsize=3, alpha=0.8)
axes[0, 0].set_yticks(range(len(sorted_models)))
axes[0, 0].set_yticklabels(sorted_models, fontsize=8)
axes[0, 0].set_xlabel('R² Score', fontsize=10)
axes[0, 0].set_title('10-Fold CV R² Performance (Mean ± Std)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')
axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

top_15_models = summary.head(15).index.tolist()
rmse_means = [summary.loc[m, 'Mean_Test_RMSE'] for m in top_15_models]
rmse_stds = [summary.loc[m, 'Std_Test_RMSE'] for m in top_15_models]

axes[0, 1].barh(range(len(top_15_models)), rmse_means, xerr=rmse_stds, capsize=3, color='orange', alpha=0.8)
axes[0, 1].set_yticks(range(len(top_15_models)))
axes[0, 1].set_yticklabels(top_15_models, fontsize=8)
axes[0, 1].set_xlabel('RMSE', fontsize=10)
axes[0, 1].set_title('Top 15 Models RMSE Performance (Mean ± Std)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

top_10_models = summary.head(10).index.tolist()
r2_box_data = []
for model in top_10_models:
    model_data = results_df[results_df['Model'] == model]['Test_R2'].dropna().values
    r2_box_data.append(model_data)

boxplot = axes[1, 0].boxplot(r2_box_data, labels=top_10_models, vert=False, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(top_10_models)))
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    
axes[1, 0].set_xlabel('R² Score', fontsize=10)
axes[1, 0].set_title('R² Distribution Across Folds (Top 10 Models)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

best_model_name = summary.index[0]
best_model_data = results_df[results_df['Model'] == best_model_name]
best_r2_per_fold = best_model_data['Test_R2'].values
best_rmse_per_fold = best_model_data['Test_RMSE'].values

x_pos = np.arange(len(best_r2_per_fold))
scatter1 = axes[1, 1].scatter(x_pos, best_r2_per_fold, color='blue', alpha=0.7, label='R²', s=80)
axes[1, 1].axhline(y=np.nanmean(best_r2_per_fold), color='blue', linestyle='--', alpha=0.7, 
                   label=f'R² Mean: {np.nanmean(best_r2_per_fold):.3f}')

ax2 = axes[1, 1].twinx()
scatter2 = ax2.scatter(x_pos, best_rmse_per_fold, color='red', alpha=0.7, label='RMSE', marker='s')
ax2.axhline(y=np.nanmean(best_rmse_per_fold), color='red', linestyle='--', alpha=0.7, 
            label=f'RMSE Mean: {np.nanmean(best_rmse_per_fold):.3f}')

axes[1, 1].set_xlabel('Fold Index', fontsize=10)
axes[1, 1].set_ylabel('R² Score', fontsize=10, color='blue')
ax2.set_ylabel('RMSE', fontsize=10, color='red')
axes[1, 1].set_title(f'{best_model_name} Performance Across Folds', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

lines1, labels1 = axes[1, 1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CV_Performance_Visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

with open(os.path.join(output_dir, 'CV_Detailed_Report.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("10translated_text (translated_text)\n")
    f.write("="*60 + "\n\n")
    f.write(f"translated_text:\n")
    f.write(f"• translated_text: {len(df_valid)}\n")
    f.write(f"• translated_text: {cond_input_dim} (translated_text)\n")
    f.write(f"• Top K translated_text: {top_k}\n")
    f.write(f"• GNN translated_text: 32\n")
    f.write(f"• translated_text: 32 + {top_k} = {32 + top_k}\n\n")
    
    f.write(f"translated_text:\n")
    f.write(f"• translated_text: 10\n")
    f.write(f"• GNN translated_text: 100 epoch, translated_text=20\n")
    f.write(f"• translated_text: GNNtranslated_text + translated_text\n\n")
    
    f.write("translated_text:\n")
    f.write("="*60 + "\n")
    f.write(summary.to_string())
    f.write("\n\n")
    
    f.write("translated_text:\n")
    f.write("="*60 + "\n")
    for i, (model, row) in enumerate(summary.head(5).iterrows()):
        f.write(f"{i+1}. {model}:\n")
        f.write(f"   • R²: {row['Mean_Test_R2']:.4f} ± {row['Std_Test_R2']:.4f}\n")
        f.write(f"   • RMSE: {row['Mean_Test_RMSE']:.4f} ± {row['Std_Test_RMSE']:.4f}\n")
        f.write(f"   • MAE: {row['Mean_Test_MAE']:.4f} ± {row['Std_Test_MAE']:.4f}\n")
        f.write(f"   • translated_text: {int(row['Count_Valid'])}\n\n")

summary_report = pd.DataFrame({
    'translated_text': ['System-FiLM GNN + translated_text (10CVtranslated_text)'],
    'translated_text': [timestamp],
    'translated_text': [len(df_valid)],
    'translated_text': [cond_input_dim],
    'Top K translated_text': [top_k],
    'GNNtranslated_text': [32],
    'translated_text': [32 + top_k],
    'translated_text': [10],
    'translated_text': [summary.index[0]],
    'translated_textR²': [f"{summary.iloc[0]['Mean_Test_R2']:.4f} ± {summary.iloc[0]['Std_Test_R2']:.4f}"],
    'translated_textRMSE': [f"{summary.iloc[0]['Mean_Test_RMSE']:.4f} ± {summary.iloc[0]['Std_Test_RMSE']:.4f}"],
    'translated_text': [len(summary)],
    'translated_text': [str(device)]
})
summary_report.to_csv(os.path.join(output_dir, 'Experiment_Summary.csv'), index=False)

print(f"\n✅ 10translated_text！")
print("="*60)
print("translated_text:")
print(f"  - CV_All_Folds_Results_Strict.csv (translated_text)")
print(f"  - CV_Statistical_Summary_Strict.csv (translated_text)")
print(f"  - CV_Detailed_Performance_Matrix.csv (translated_text)")
print(f"  - CV_Best_Per_Fold_Results.csv (translated_text)")
print(f"  - CV_Performance_Visualization.png (translated_text)")
print(f"  - CV_Detailed_Report.txt (translated_text)")
print(f"  - Experiment_Summary.csv (translated_text)")
print(f"  - Global_Top20_Features.png (translated_text)")
print(f"  - Global_Top_{top_k}_Features.csv (translated_text)")
print("="*60)
print(f"🎯 translated_text: {summary.index[0]} (R²: {summary.iloc[0]['Mean_Test_R2']:.4f} ± {summary.iloc[0]['Std_Test_R2']:.4f})")
print("="*60)

torch.cuda.empty_cache()
gc.collect()

print("\n🎉 translated_text！translated_text。")