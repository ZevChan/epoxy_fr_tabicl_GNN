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
import gc
import warnings
import copy
import json
import shap

warnings.filterwarnings('ignore')

# ==========================================
# ==========================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_dir = f"Result_NoGNNFilter_SysFiLM_32D_Strict5CV_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

run_info = {
    "output_dir": output_dir,
    "timestamp": timestamp,
    "experiment_name": "System-FiLM GNN + Multi-Model Benchmark"
}
with open("latest_run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)

print(f"========================================")
print(f"🚀 translated_text！translated_text: {output_dir}/")
print(f"========================================\n")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# ==========================================
print("=== 1. translated_text ===")
df = pd.read_csv("EP+FR+CURING_SMILES+translated_text_DATASET_20260414.csv")

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
    
    cols_to_drop = [col for col in temp_cols + time_cols if col in df.columns]
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
print("\n=== 5. translated_text SHAP translated_text ===")

scaler_global = StandardScaler()
X_num_global = scaler_global.fit_transform(num_features_raw)

xgb_global = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_estimators=200, 
    tree_method='hist'
)
xgb_global.fit(X_num_global, y)

num_feature_names = df_valid.drop(columns=['EP_SMILES', 'FR_SMILES', 'CURING_SMILES', 'LOI']).columns.tolist()
if wt_matrix_valid is not None and len(num_feature_names) < X_num_global.shape[1]:
    num_feature_names.extend(['EP_wt_fraction', 'FR_wt_fraction', 'CURING_wt_fraction'])

print(">>> translated_text SHAP translated_text (translated_text，translated_text)...")
explainer = shap.TreeExplainer(xgb_global)
shap_values = explainer.shap_values(X_num_global)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance_df = pd.DataFrame({
    'Feature_Name': num_feature_names,
    'SHAP_Importance': mean_abs_shap
}).sort_values(by='SHAP_Importance', ascending=False)

# 32(GNN) + 450(SHAP) = 482 < 500
top_k = min(450, len(shap_importance_df))
selected_global = shap_importance_df.head(top_k)

top_feature_names = selected_global['Feature_Name'].tolist()
top_feature_indices = [num_feature_names.index(feat) for feat in top_feature_names]

selected_global.to_csv(os.path.join(output_dir, f'Global_Top_{top_k}_SHAP_Features.csv'), index=False)
top_20_features = selected_global.head(20)
top_20_features.to_csv(os.path.join(output_dir, 'Global_Top20_SHAP_Features.csv'), index=False)
print(f"✅ SHAP translated_text！translated_text {cond_input_dim} translated_text {top_k} translated_text。")

# ==========================================
# ==========================================
print("\n" + "="*60)
print("translated_text (5-Fold CV)")
print(f"【translated_text】translated_text SHAP translated_text Top {top_k} translated_text GNN translated_text，translated_text。")
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

sample_predictions = {name: [] for name in models.keys()}
sample_actuals = {name: [] for name in models.keys()}
all_sample_predictions = {name: np.zeros(len(y)) * np.nan for name in models.keys()}
for special_model in ['TabPFN', 'TabICL']:
    if special_model not in sample_predictions:
        sample_predictions[special_model] = []
        sample_actuals[special_model] = []
        all_sample_predictions[special_model] = np.zeros(len(y)) * np.nan

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\ntranslated_text: {device}")

for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
    print(f"\n" + "-"*40)
    print(f"[Fold {fold+1}/5] translated_text...")
    print(f"translated_text: {len(train_idx)} translated_text, translated_text: {len(test_idx)} translated_text")
    print("-"*40)
    
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    # -----------------------------------------------------
    # -----------------------------------------------------
    scaler_fold = StandardScaler()
    X_train_num_fold = scaler_fold.fit_transform(num_features_raw[train_idx])
    X_test_num_fold = scaler_fold.transform(num_features_raw[test_idx])
    
    X_train_num_selected = X_train_num_fold[:, top_feature_indices]
    X_test_num_selected = X_test_num_fold[:, top_feature_indices]
    
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
    print(">>> translated_text System-FiLM GNN translated_text...")
    
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
    
    ep_te = create_graph_data(df_valid.iloc[test_idx]['EP_SMILES'].tolist())
    fr_te = create_graph_data(df_valid.iloc[test_idx]['FR_SMILES'].tolist())
    cu_te = create_graph_data(df_valid.iloc[test_idx]['CURING_SMILES'].tolist())
    
    X_train_gnn_fold = extract_film_gnn_features(fold_model, ep_full_tr, fr_full_tr, cu_full_tr, X_train_num_fold, wt_matrix_valid[train_idx], device)
    X_test_gnn_fold = extract_film_gnn_features(fold_model, ep_te, fr_te, cu_te, X_test_num_fold, wt_matrix_valid[test_idx], device)
    
    X_train_final = np.concatenate([X_train_gnn_fold, X_train_num_selected], axis=1)
    X_test_final = np.concatenate([X_test_gnn_fold, X_test_num_selected], axis=1)
    
    print(f">>> translated_text: GNN(32) + SHAP_Top({top_k}) = translated_text {X_train_final.shape[1]}")
    
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
            
            sample_predictions[name].extend(preds.tolist())
            sample_actuals[name].extend(y_test_fold.tolist())
            for i, t_idx in enumerate(test_idx):
                all_sample_predictions[name][t_idx] = preds[i]
            
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
            tabpfn_model = TabPFNRegressor(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                ignore_pretraining_limits=True  # translated_text，translated_text
            )
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
            
            sample_predictions['TabPFN'].extend(preds_tabpfn.tolist())
            sample_actuals['TabPFN'].extend(y_test_fold.tolist())
            for i, t_idx in enumerate(test_idx):
                all_sample_predictions['TabPFN'][t_idx] = preds_tabpfn[i]
                
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
            
            sample_predictions['TabICL'].extend(preds_tabicl.tolist())
            sample_actuals['TabICL'].extend(y_test_fold.tolist())
            for i, t_idx in enumerate(test_idx):
                all_sample_predictions['TabICL'][t_idx] = preds_tabicl[i]
                
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
print("translated_text OOF translated_text...")
print("="*60)

for model_name in models.keys():
    if len(sample_predictions[model_name]) > 0:
        pd.DataFrame({
            'Actual_LOI': sample_actuals[model_name],
            'Predicted_LOI': sample_predictions[model_name]
        }).to_csv(os.path.join(output_dir, f'OOF_Predictions_{model_name}.csv'), index=False)

if TABPFN_AVAILABLE and 'TabPFN' in sample_predictions:
    pd.DataFrame({
        'Actual_LOI': sample_actuals['TabPFN'],
        'Predicted_LOI': sample_predictions['TabPFN']
    }).to_csv(os.path.join(output_dir, f'OOF_Predictions_TabPFN.csv'), index=False)

if TABICL_AVAILABLE and 'TabICL' in sample_predictions:
    pd.DataFrame({
        'Actual_LOI': sample_actuals['TabICL'],
        'Predicted_LOI': sample_predictions['TabICL']
    }).to_csv(os.path.join(output_dir, f'OOF_Predictions_TabICL.csv'), index=False)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(output_dir, 'CV_All_Folds_Results_Strict.csv'), index=False)

summary = results_df.groupby('Model').agg(
    Mean_Test_R2=('Test_R2', 'mean'), 
    Std_Test_R2=('Test_R2', 'std'),
    Mean_Test_RMSE=('Test_RMSE', 'mean'), 
    Std_Test_RMSE=('Test_RMSE', 'std'),
    Mean_Test_MAE=('Test_MAE', 'mean'), 
    Std_Test_MAE=('Test_MAE', 'std'),
    Count_Valid=('Test_R2', lambda x: x.notna().sum()) 
).sort_values(by='Mean_Test_R2', ascending=False)

summary = summary[summary['Count_Valid'] >= 5]
summary.to_csv(os.path.join(output_dir, 'CV_Statistical_Summary_Strict.csv'))

print(f"\n✅ 5translated_text！translated_text {output_dir}")