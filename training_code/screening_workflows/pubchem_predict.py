import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import warnings
from tabicl import TabICLRegressor, TabICLClassifier
import pickle
import json
from alvadesccliwrapper.alvadesc import AlvaDesc
from rdkit import Chem
from rdkit.Chem import Descriptors
import gc
from tqdm import tqdm
import time
import re

warnings.filterwarnings('ignore')

# ==========================================
# 0. Configuration Parameters
# ==========================================
# Epoxy resin and curing agent info
EP_SMILES = "CC(C1=CC=C(C=C1)OCC2CO2)(C3=CC=C(C=C3)OCC4CO4)C"
CURING_SMILES = "NC1=CC=C(CC2=CC=C(N)C=C2)C=C1"

# Epoxy equivalent weight fixed at 190
EEW = 190.0
print(f"环氧树脂EEW: {EEW:.2f}")

# FR loading levels (wt%)
FR_ADDITIONS = [1, 2, 5, 10, 15, 20, 25, 30]

# Curing process parameters
CURING_PARAMS = {
    'Curing_Tem1': 80, 'Curing_Time1': 2,
    'Curing_Tem2': 120, 'Curing_Time2': 2,
    'Curing_Tem3': 150, 'Curing_Time3': 2,
    'Curing_Tem4': 0, 'Curing_Time4': 0,
    'Curing_Tem5': 0, 'Curing_Time5': 0,
    'Curing_Tem6': 0, 'Curing_Time6': 0,
    'Curing_Tem7': 0, 'Curing_Time7': 0,
    'Curing_Tem8': 0, 'Curing_Time8': 0,
    'Curing_Tem9': 0, 'Curing_Time9': 0,
    'Curing_Pressure': 0, 'Curing_UV': 0
}

# CalculateT_max, t_total, Q_thermal
temp_cols = [f'Curing_Tem{i}' for i in range(1, 10)]
time_cols = [f'Curing_Time{i}' for i in range(1, 10)]
T_max = max([CURING_PARAMS[col] for col in temp_cols])
t_total = sum([CURING_PARAMS[col] for col in time_cols])
Q_thermal = sum([CURING_PARAMS[temp_cols[i]] * CURING_PARAMS[time_cols[i]] for i in range(9)])

# Model paths and corresponding training datasets
MODEL_PATHS = {
    'LOI': {
        'features': r"results\shap_features\loi_global_shap_features.csv",
        'params': r"results\shap_features\loi_best_params.txt",
        'best_k': 71,
        'train_data': r"data\task_datasets\loi_dataset.csv"
    },
    'Tg': {
        'features': r"results\shap_features\tg_global_shap_features.csv",
        'params': r"results\shap_features\tg_best_params.txt",
        'best_k': 72,
        'train_data': r"data\task_datasets\tg_dataset.csv"
    },
    'Tensile': {
        'features': r"results\shap_features\tensile_global_shap_features.csv",
        'params': r"results\shap_features\tensile_best_params.txt",
        'best_k': 40,
        'train_data': r"data\task_datasets\tensile_dataset.csv"
    },
    'UL94': {
        'features': r"results\shap_features\ul94_global_shap_features.csv",
        'params': r"results\shap_features\ul94_best_params.txt",
        'best_k': 70,
        'train_data': r"data\task_datasets\ul94_dataset.csv"
    }
}

# Similarity-filtered SMILES file paths
SIMILARITY_FILES = [
    'unique_similar_smiles_0.40.smi',
    'unique_similar_smiles_0.50.smi',
    'unique_similar_smiles_0.60.smi',
    'unique_similar_smiles_0.70.smi',
    'unique_similar_smiles_0.80.smi'
]

OUTPUT_DIR = r"outputs\pubchem_screening"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# alvaDescPath
ALVADESC_PATH = r"C:\Program Files\Alvascience\alvaDesc\alvaDescCLI.exe"  # Set your alvaDesc CLI path here

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. Helper Functions
# ==========================================
def load_best_params(param_file):
    """Load best parameters, extract model-relevant params only"""
    params = {}
    
    with open(param_file, 'r') as f:
        content = f.read()
    
    # Extract parameters using regex
    # TabICL accepted params: n_estimators, feat_shuffle_method, outlier_threshold
    param_patterns = {
        'n_estimators': r'n_estimators:\s*(\d+)',
        'feat_shuffle_method': r'feat_shuffle_method:\s*(\w+)',
        'outlier_threshold': r'outlier_threshold:\s*([\d\.]+)'
    }
    
    for param_name, pattern in param_patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # Convert to appropriate types
            if param_name == 'n_estimators':
                params[param_name] = int(value)
            elif param_name == 'outlier_threshold':
                params[param_name] = float(value)
            else:
                params[param_name] = value
    
    # Set defaults if params not found
    if 'n_estimators' not in params:
        params['n_estimators'] = 8
    if 'feat_shuffle_method' not in params:
        params['feat_shuffle_method'] = 'latin'
    if 'outlier_threshold' not in params:
        params['outlier_threshold'] = 3.0
    
    print(f"Loaded parameters: {params}")
    return params

def calculate_all_descriptors(smiles_list, prefix):
    """Calculate all descriptors using alvaDesc"""
    try:
        aDesc = AlvaDesc(ALVADESC_PATH)
        aDesc.set_input_SMILES(smiles_list)
        
        # Calculate all descriptors
        if not aDesc.calculate_descriptors('ALL'):
            print(f'Error: {aDesc.get_error()}')
            return None
        
        descriptors = aDesc.get_output_descriptors()
        values = aDesc.get_output()
        
        # Convert to DataFrame and add prefix
        df = pd.DataFrame(values, columns=descriptors)
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"Calculate描述符时出错: {e}")
        return None

def get_required_descriptors(models_info):
    """Get all required descriptor names (exclude computed features)"""
    all_descriptors = set()
    
    for target, info in models_info.items():
        # Load feature list
        shap_df = pd.read_csv(info['features'])
        best_k = info['best_k']
        features = shap_df['Feature_Name'].tolist()[:best_k]
        
        # Extract descriptor features (exclude base/computed)
        for feat in features:
            # Only include alvaDesc descriptors, exclude computed
            if (feat.startswith('EP_') or feat.startswith('FR_') or feat.startswith('CURING_')) and \
               not any(x in feat for x in ['wt_fraction', 'T_max', 't_total', 'Q_thermal', 'EEW', 'AdditionAmount']):
                all_descriptors.add(feat)
    
    return list(all_descriptors)

def calculate_specific_descriptors(smiles_list, prefix, required_descriptors):
    """Calculate specific descriptors"""
    try:
        aDesc = AlvaDesc(ALVADESC_PATH)
        aDesc.set_input_SMILES(smiles_list)
        
        # Extract descriptor names, strip prefix
        desc_names = []
        for desc in required_descriptors:
            if desc.startswith(f"{prefix}_"):
                desc_names.append(desc.replace(f"{prefix}_", ""))
        
        if not desc_names:
            return pd.DataFrame()
        
        # Calculate描述符
        if not aDesc.calculate_descriptors(desc_names):
            print(f'Error: {aDesc.get_error()}')
            return pd.DataFrame()
        
        descriptors = aDesc.get_output_descriptors()
        values = aDesc.get_output()
        
        # Convert to DataFrame and add prefix
        df = pd.DataFrame(values, columns=descriptors)
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"Calculate描述符时出错: {e}")
        return pd.DataFrame()

# ==========================================
# 2. Load Models and Features
# ==========================================
def load_and_train_models():
    """Load and train all models"""
    models = {}
    
    for target, paths in MODEL_PATHS.items():
        print(f"\n=== Load并训练{target}模型 ===")
        
        # Load features
        shap_df = pd.read_csv(paths['features'])
        best_k = paths['best_k']
        top_k_features = shap_df['Feature_Name'].tolist()[:best_k]
        
        # Load parameters
        params = load_best_params(paths['params'])
        
        # Create model using global device variable
        if target == 'UL94':
            model = TabICLClassifier(**params, device=device)
        else:
            model = TabICLRegressor(**params, device=device)
        
        # Load training data
        train_data_path = paths['train_data']
        print(f"Load training data: {train_data_path}")
        
        try:
            df_train = pd.read_csv(train_data_path)
        except Exception as e:
            print(f"Failed to load training data: {e}")
            # If failed, try default path
            default_path = "EP+FR+CURING_SMILES+描述符_DATASET_20260414.csv"
            print(f"Try loading default training data: {default_path}")
            try:
                df_train = pd.read_csv(default_path)
            except Exception as e2:
                print(f"Default data load also failed: {e2}")
                # Create empty DataFrame
                df_train = pd.DataFrame()
        
        if df_train.empty:
            print(f"警告: {target}Training data empty, cannot train")
            models[target] = {
                'model': model,
                'features': top_k_features,
                'scaler': None,
                'type': 'classification' if target == 'UL94' else 'regression'
            }
            continue
        
        print(f"Original data shape: {df_train.shape}")
        
        # Data preprocessing (consistent with training)
        print("Perform data preprocessing...")
        
        # Process curing parameters
        temp_cols = [f'Curing_Tem{i}' for i in range(1, 10)]
        time_cols = [f'Curing_Time{i}' for i in range(1, 10)]
        
        # Check and add necessary columns
        if all(col in df_train.columns for col in temp_cols + time_cols):
            df_train['T_max'] = df_train[temp_cols].max(axis=1)
            df_train['t_total'] = df_train[time_cols].sum(axis=1)
            thermal_sum = sum([df_train[tem_col].fillna(0) * df_train[time_col].fillna(0) 
                              for tem_col, time_col in zip(temp_cols, time_cols)])
            df_train['Q_thermal'] = thermal_sum
            
            # Drop raw temperature and time columns
            cols_to_drop = [col for col in temp_cols + time_cols if col in df_train.columns]
            df_train = df_train.drop(columns=cols_to_drop)
            
            # Fill NaN values
            df_train[['T_max', 't_total', 'Q_thermal']] = df_train[['T_max', 't_total', 'Q_thermal']].fillna(0)
        
        # Calculate mass fractions
        if 'Flame_retardant_AdditionAmount(wt%)' in df_train.columns:
            df_train['EP_wt_fraction'] = (100.0 - df_train['Flame_retardant_AdditionAmount(wt%)'].fillna(0) - 
                                         df_train['Curing_agent_AdditionAmount(wt%)'].fillna(0)) / 100.0
            df_train['FR_wt_fraction'] = df_train['Flame_retardant_AdditionAmount(wt%)'].fillna(0) / 100.0
            df_train['CURING_wt_fraction'] = df_train['Curing_agent_AdditionAmount(wt%)'].fillna(0) / 100.0
        
        # Check if all required features exist in data
        missing_features = []
        for feat in top_k_features:
            if feat not in df_train.columns:
                missing_features.append(feat)
        
        if missing_features:
            print(f"警告: 训练数据中缺少以下特征: {missing_features}")
            print("Try adding zero-filled columns...")
            for feat in missing_features:
                df_train[feat] = 0.0
        
        # Extract target by task type
        if target == 'UL94':
            target_col = 'UL94'
        elif target == 'LOI':
            target_col = 'LOI'
        elif target == 'Tg':
            target_col = 'Tg'
        elif target == 'Tensile':
            target_col = 'Tensile'
        
        # Drop rows with NaN target
        if target_col in df_train.columns:
            df_train = df_train.dropna(subset=[target_col])
        else:
            print(f"警告: 训练数据中没有找到{target}target column, cannot train")
            models[target] = {
                'model': model,
                'features': top_k_features,
                'scaler': None,
                'type': 'classification' if target == 'UL94' else 'regression'
            }
            continue
        
        print(f"Data shape after NaN target drop: {df_train.shape}")
        
        # Extract features and targets from training data
        X_train = df_train[top_k_features].values
        y_train = df_train[target_col].values
        
        print(f"X_train形状: {X_train.shape}")
        print(f"y_train形状: {y_train.shape}")
        
        # Check for NaN values
        if np.isnan(X_train).any():
            print("警告: X_train中包含NaN值，进行填充...")
            X_train = np.nan_to_num(X_train, nan=0.0)
        
        if np.isnan(y_train).any():
            print("警告: y_train中包含NaN值，进行填充...")
            y_train = np.nan_to_num(y_train, nan=0.0)
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        print(f"训练{target}模型...")
        model.fit(X_train_scaled, y_train)
        
        models[target] = {
            'model': model,
            'features': top_k_features,
            'scaler': scaler,
            'type': 'classification' if target == 'UL94' else 'regression'
        }
        
        print(f"  成功训练{target}模型，特征维度: {X_train.shape}")
    
    return models

# ==========================================
# 3. Calculate环氧和固化剂描述符
# ==========================================
def calculate_ep_curing_descriptors(models):
    """Calculate环氧树脂和固化剂的描述符"""
    print("\n=== Calculate环氧树脂和固化剂描述符 ===")
    
    # 获取All需要的描述符（排除Calculate特征）
    all_descriptors = get_required_descriptors(MODEL_PATHS)
    
    # Calculate EP descriptors
    print("Calculate EP descriptors...")
    ep_descriptors = calculate_specific_descriptors([EP_SMILES], 'EP', all_descriptors)
    
    # Calculate curing agent descriptors
    print("Calculate curing agent descriptors...")
    curing_descriptors = calculate_specific_descriptors([CURING_SMILES], 'CURING', all_descriptors)
    
    return ep_descriptors, curing_descriptors

# ==========================================
# 4. Prediction Functions
# ==========================================
def predict_for_fr(fr_smiles, fr_descriptors, ep_descriptors, curing_descriptors, addition_pct, models):
    """预测单个阻燃剂在特定添加量下的性能"""
    results = {}
    
    # Calculate mass fractions
    fr_wt_fraction = addition_pct / 100.0
    curing_wt_fraction = (100 - addition_pct) / 6 / 100.0
    ep_wt_fraction = (100 - addition_pct) * 5 / 6 / 100.0
    
    # 基础特征（包括Calculate特征）
    base_features = {
        'EEW': EEW,
        'Flame_retardant_AdditionAmount(wt%)': addition_pct,
        'Curing_agent_AdditionAmount(wt%)': (100 - addition_pct) / 6,
        'EP_wt_fraction': ep_wt_fraction,
        'FR_wt_fraction': fr_wt_fraction,
        'CURING_wt_fraction': curing_wt_fraction,
        'T_max': T_max,
        't_total': t_total,
        'Q_thermal': Q_thermal,
        'Curing_Pressure': 0,
        'Curing_UV': 0
    }
    
    # 为每个目标进行预测
    for target, model_info in models.items():
        features = model_info['features']
        
        # Build feature vector
        feature_vector = []
        for feat in features:
            if feat in base_features:
                feature_vector.append(base_features[feat])
            elif feat.startswith('FR_'):
                # 从阻燃剂描述符中获取
                fr_feat = feat.replace('FR_', '')
                if not fr_descriptors.empty and fr_feat in fr_descriptors.columns:
                    feature_vector.append(fr_descriptors[fr_feat].iloc[0])
                else:
                    feature_vector.append(0)  # 缺失值填充为0
            elif feat.startswith('EP_'):
                # 从环氧树脂描述符中获取
                ep_feat = feat.replace('EP_', '')
                if not ep_descriptors.empty and ep_feat in ep_descriptors.columns:
                    feature_vector.append(ep_descriptors[ep_feat].iloc[0])
                else:
                    feature_vector.append(0)
            elif feat.startswith('CURING_'):
                # 从固化剂描述符中获取
                curing_feat = feat.replace('CURING_', '')
                if not curing_descriptors.empty and curing_feat in curing_descriptors.columns:
                    feature_vector.append(curing_descriptors[curing_feat].iloc[0])
                else:
                    feature_vector.append(0)
            else:
                feature_vector.append(0)
        
        # 转换为numpy数组并Standardize
        X = np.array(feature_vector).reshape(1, -1)
        
        # Checkscaler是否存在
        if model_info['scaler'] is None:
            print(f"警告: {target}模型的scaler不存在，跳过预测")
            if model_info['type'] == 'classification':
                results[f'{target}_pred'] = 0
                results[f'{target}_prob'] = 0
            else:
                results[f'{target}_pred'] = 0
            continue
            
        X_scaled = model_info['scaler'].transform(X)
        
        # 预测
        try:
            if model_info['type'] == 'classification':
                prob = model_info['model'].predict_proba(X_scaled)[:, 1]
                pred = 1 if prob > 0.5 else 0
                results[f'{target}_pred'] = pred
                results[f'{target}_prob'] = prob[0]
            else:
                pred = model_info['model'].predict(X_scaled)[0]
                results[f'{target}_pred'] = pred
        except Exception as e:
            print(f"Prediction failed: {e}")
            if model_info['type'] == 'classification':
                results[f'{target}_pred'] = 0
                results[f'{target}_prob'] = 0
            else:
                results[f'{target}_pred'] = 0
    
    return results

# ==========================================
# 5. Main Function
# ==========================================
def main():
    print("=" * 80)
    print("Start screening similarity-filtered FRs")
    print("=" * 80)
    
    # 1. Load and train models
    print("\n1. Load and train models...")
    models = load_and_train_models()
    
    # 2. Calculate EP and curing agent descriptors
    ep_descriptors, curing_descriptors = calculate_ep_curing_descriptors(models)
    
    if ep_descriptors.empty:
        print("警告: 环氧树脂描述符Calculate失败，使用空描述符")
    if curing_descriptors.empty:
        print("警告: 固化剂描述符Calculate失败，使用空描述符")
    
    # 3. Get required FR descriptors (exclude computed)
    all_descriptors = get_required_descriptors(MODEL_PATHS)
    
    # 4. Process each similarity file
    for similarity_file in SIMILARITY_FILES:
        file_path = os.path.join(OUTPUT_DIR, similarity_file)
        
        if not os.path.exists(file_path):
            print(f"\n警告: File {similarity_file} 不存在，跳过")
            continue
        
        print(f"\n{'='*60}")
        print(f"ProcessFile: {similarity_file}")
        print(f"{'='*60}")
        
        # Read SMILES
        with open(file_path, 'r', encoding='utf-8') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        print(f"找到 {len(smiles_list)} 个SMILES")
        
        if len(smiles_list) == 0:
            print("File empty, skipping")
            continue
        
        # Batch processing to avoid memory overflow
        batch_size = 50  # Reduce batch size to avoid alvaDesc memory issues
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  Calculate批次 {batch_num} 的描述符 ({len(batch_smiles)}个分子)...")
            
            # Calculate FR descriptors
            fr_descriptors_batch = calculate_specific_descriptors(batch_smiles, 'FR', all_descriptors)
            
            if fr_descriptors_batch.empty:
                print(f"  批次 {batch_num} 描述符Calculate失败，跳过")
                continue
            
            # Create result DataFrame per loading level
            for addition_pct in FR_ADDITIONS:
                results_list = []
                
                for idx, fr_smiles in enumerate(batch_smiles):
                    # Get descriptors for this FR
                    if idx < len(fr_descriptors_batch):
                        fr_desc = fr_descriptors_batch.iloc[[idx]]
                    else:
                        continue
                    
                    # Predict properties
                    try:
                        pred_results = predict_for_fr(fr_smiles, fr_desc, ep_descriptors, 
                                                     curing_descriptors, addition_pct, models)
                        
                        # Build result row
                        result_row = {
                            'FR_SMILES': fr_smiles,
                            'FR_Addition_pct': addition_pct,
                            'LOI_pred': pred_results.get('LOI_pred', None),
                            'Tg_pred': pred_results.get('Tg_pred', None),
                            'Tensile_pred': pred_results.get('Tensile_pred', None),
                            'UL94_pred': pred_results.get('UL94_pred', None),
                            'UL94_prob': pred_results.get('UL94_prob', None)
                        }
                        results_list.append(result_row)
                        
                    except Exception as e:
                        print(f"Prediction failed: {fr_smiles}, 错误: {e}")
                        continue
                
                # Save results for this loading level
                if results_list:
                    results_df = pd.DataFrame(results_list)
                    # Use similarity filename as prefix
                    similarity_prefix = similarity_file.replace('.smi', '')
                    output_file = os.path.join(OUTPUT_DIR, f'predictions_{similarity_prefix}_addition_{addition_pct}pct_batch_{batch_num}.csv')
                    results_df.to_csv(output_file, index=False)
                    print(f"  Save添加量{addition_pct}%的Results到: {output_file}")
            
            # Clear memory
            del fr_descriptors_batch
            gc.collect()
            
            # Add delay to avoid alvaDesc overload
            time.sleep(1)
    
    print("\n" + "=" * 80)
    print("预测Done！")
    print("=" * 80)

# ==========================================
# 6. Filter and Summarize (Multi-Criteria)
# ==========================================
def filter_and_summarize_multiple_criteria():
    """Filter and summarize predictions (multi-criteria)"""
    print("\nFilter and summarize predictions (multi-criteria)...")
    
    # Define multiple filter criteria
    FILTER_CRITERIA_LIST = [
        {
            'name': 'Criterion1_HighFR_HighMechanics',
            'criteria': {
                'LOI_min': 32,  # 最小LOI值
                'Tg_min': 160,  # 最小Tg值
                'Tensile_min': 60,  # 最小拉伸强度
                'UL94_min': 1  # UL94等级为V-0
            }
        },
        {
            'name': 'Criterion2_HighTg_ModerateFR',
            'criteria': {
                'LOI_min': 28,  # 最小LOI值
                'Tg_min': 160,  # 最小Tg值
                'Tensile_min': 60,  # 最小拉伸强度
                'UL94_min': 0  # UL94等级为V-0
            }
        },
        {
            'name': 'Criterion3_UltraHighTg_HighMechanics',
            'criteria': {
                'LOI_min': 28,  # 最小LOI值
                'Tg_min': 180,  # 最小Tg值
                'Tensile_min': 70,  # 最小拉伸强度
                'UL94_min': 1  # UL94等级为V-0
            }
        }
    ]
    
    # Create summary per filter criterion
    all_summary_results = {}
    
    for filter_info in FILTER_CRITERIA_LIST:
        filter_name = filter_info['name']
        filter_criteria = filter_info['criteria']
        
        print(f"\n{'='*60}")
        print(f"Apply filter criteria: {filter_name}")
        print(f"LOI_min: {filter_criteria['LOI_min']}, Tg_min: {filter_criteria['Tg_min']}, "
              f"Tensile_min: {filter_criteria['Tensile_min']}, UL94_min: {filter_criteria['UL94_min']}")
        print(f"{'='*60}")
        
        filter_summary = {}
        
        for addition_pct in FR_ADDITIONS:
            print(f"\nProcess loading level: {addition_pct}%")
            
            # Collect all result files for this loading
            pattern = f'predictions_*_addition_{addition_pct}pct_*.csv'
            result_files = []
            for file in os.listdir(OUTPUT_DIR):
                if file.startswith(f'predictions_') and f'addition_{addition_pct}pct' in file:
                    result_files.append(os.path.join(OUTPUT_DIR, file))
            
            if not result_files:
                print(f"  未找到添加量{addition_pct}%的ResultsFile")
                filter_summary[addition_pct] = {
                    'total_molecules': 0,
                    'filtered_molecules': 0,
                    'filtered_percentage': 0
                }
                continue
            
            # Merge all results
            all_results = []
            for file in tqdm(result_files, desc=f"MergeFile"):
                try:
                    df = pd.read_csv(file)
                    all_results.append(df)
                except Exception as e:
                    print(f"  ReadFile失败: {file}, 错误: {e}")
                    continue
            
            if not all_results:
                filter_summary[addition_pct] = {
                    'total_molecules': 0,
                    'filtered_molecules': 0,
                    'filtered_percentage': 0
                }
                continue
            
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Save merged file (once only)
            if filter_name == FILTER_CRITERIA_LIST[0]['name']:  # 只在第一个条件时SaveMergeFile
                combined_file = os.path.join(OUTPUT_DIR, f'all_predictions_addition_{addition_pct}pct.csv')
                combined_df.to_csv(combined_file, index=False)
                print(f"  MergeFile已Save: {combined_file}")
            
            # Apply filter criteria
            filtered_df = combined_df[
                (combined_df['LOI_pred'] >= filter_criteria['LOI_min']) &
                (combined_df['Tg_pred'] >= filter_criteria['Tg_min']) &
                (combined_df['Tensile_pred'] >= filter_criteria['Tensile_min']) &
                (combined_df['UL94_pred'] >= filter_criteria['UL94_min'])
            ]
            
            # Save filtered results
            filtered_dir = os.path.join(OUTPUT_DIR, filter_name)
            os.makedirs(filtered_dir, exist_ok=True)
            filtered_file = os.path.join(filtered_dir, f'filtered_addition_{addition_pct}pct.csv')
            filtered_df.to_csv(filtered_file, index=False)
            
            # Statistics
            filter_summary[addition_pct] = {
                'total_molecules': len(combined_df),
                'filtered_molecules': len(filtered_df),
                'filtered_percentage': len(filtered_df) / len(combined_df) * 100 if len(combined_df) > 0 else 0
            }
            
            print(f"  总分子数: {len(combined_df)}")
            print(f"  Filter后分子数: {len(filtered_df)}")
            print(f"  Filter比例: {filter_summary[addition_pct]['filtered_percentage']:.2f}%")
            
            # Save top N best molecules
            if len(filtered_df) > 0:
                # Sort by composite score
                filtered_df['综合评分'] = (
                    filtered_df['LOI_pred'] / filter_criteria['LOI_min'] * 0.3 +
                    filtered_df['Tg_pred'] / filter_criteria['Tg_min'] * 0.3 +
                    filtered_df['Tensile_pred'] / filter_criteria['Tensile_min'] * 0.3 +
                    filtered_df['UL94_pred'] * 0.1
                )
                top_n = min(100, len(filtered_df))
                top_molecules = filtered_df.nlargest(top_n, '综合评分')
                top_file = os.path.join(filtered_dir, f'top_{top_n}_addition_{addition_pct}pct.csv')
                top_molecules.to_csv(top_file, index=False)
                print(f"  前{top_n}个最佳分子已Save: {top_file}")
        
        # Save summary for this criterion
        summary_df = pd.DataFrame.from_dict(filter_summary, orient='index')
        summary_df.index.name = 'Addition_pct'
        summary_file = os.path.join(OUTPUT_DIR, f'summary_{filter_name}.csv')
        summary_df.to_csv(summary_file)
        
        all_summary_results[filter_name] = summary_df
        
        print(f"\n{filter_name}SummaryResults已Save到: {summary_file}")
    
    # Create comprehensive comparison table
    print(f"\n{'='*60}")
    print("Create comprehensive comparison table")
    print(f"{'='*60}")
    
    comparison_data = []
    for filter_info in FILTER_CRITERIA_LIST:
        filter_name = filter_info['name']
        summary_df = all_summary_results.get(filter_name)
        
        if summary_df is not None and not summary_df.empty:
            for addition_pct in FR_ADDITIONS:
                if addition_pct in summary_df.index:
                    row_data = summary_df.loc[addition_pct]
                    comparison_data.append({
                        'Filter条件': filter_name,
                        '添加量(%)': addition_pct,
                        '总分子数': int(row_data['total_molecules']),
                        'Filter分子数': int(row_data['filtered_molecules']),
                        'Filter比例(%)': round(row_data['filtered_percentage'], 2)
                    })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = os.path.join(OUTPUT_DIR, 'Filter条件综合比较.csv')
        comparison_df.to_csv(comparison_file, index=False)
        print(f"综合比较表已Save到: {comparison_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("FilterResultsSummary")
        print(f"{'='*60}")
        
        for filter_info in FILTER_CRITERIA_LIST:
            filter_name = filter_info['name']
            total_filtered = 0
            for addition_pct in FR_ADDITIONS:
                filtered_count = comparison_df[
                    (comparison_df['Filter条件'] == filter_name) & 
                    (comparison_df['添加量(%)'] == addition_pct)
                ]['Filter分子数'].sum()
                total_filtered += filtered_count
            
            print(f"{filter_name}: 总共Filter出 {total_filtered} 个分子")
    
    return all_summary_results

# ==========================================
# Execute Main Program
# ==========================================
if __name__ == "__main__":
    # Run predictions
    main()
    
    # Filter and Summarize Results (multi-criteria)
    summary = filter_and_summarize_multiple_criteria()
    
    print("\n" + "=" * 80)
    print("All任务Done！")
    print("=" * 80)