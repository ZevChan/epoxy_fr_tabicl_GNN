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
# 预测数据File
PREDICTION_CSV_FILE = "EP+FR+CURING_SMILES+描述符_预测.csv"

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

# OutputDirectory
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
# 3. 预测函数
# ==========================================
def predict_for_row(row_data, ep_descriptors, fr_descriptors, curing_descriptors, models):
    """预测单行数据的性能"""
    results = {}
    
    # 从行数据中提取Parameters
    eew = row_data.get('EEW', 0)
    fr_addition = row_data.get('Flame_retardant_AdditionAmount(wt%)', 0)
    curing_addition = row_data.get('Curing_agent_AdditionAmount(wt%)', 0)
    
    # Calculate mass fractions
    fr_wt_fraction = fr_addition / 100.0
    curing_wt_fraction = curing_addition / 100.0
    ep_wt_fraction = (100 - fr_addition - curing_addition) / 100.0
    
    # 提取Curing process parameters
    temp_cols = [f'Curing_Tem{i}' for i in range(1, 10)]
    time_cols = [f'Curing_Time{i}' for i in range(1, 10)]
    
    # CalculateT_max, t_total, Q_thermal
    temps = [row_data.get(col, 0) for col in temp_cols]
    times = [row_data.get(col, 0) for col in time_cols]
    
    T_max = max(temps) if temps else 0
    t_total = sum(times) if times else 0
    Q_thermal = sum([t * d for t, d in zip(temps, times)]) if temps and times else 0
    
    # 基础特征（包括Calculate特征）
    base_features = {
        'EEW': eew,
        'Flame_retardant_AdditionAmount(wt%)': fr_addition,
        'Curing_agent_AdditionAmount(wt%)': curing_addition,
        'EP_wt_fraction': ep_wt_fraction,
        'FR_wt_fraction': fr_wt_fraction,
        'CURING_wt_fraction': curing_wt_fraction,
        'T_max': T_max,
        't_total': t_total,
        'Q_thermal': Q_thermal,
        'Curing_Pressure': row_data.get('Curing_Pressure', 0),
        'Curing_UV': row_data.get('Curing_UV', 0)
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
# 4. 主函数
# ==========================================
def main():
    print("=" * 80)
    print(f"Start预测 {PREDICTION_CSV_FILE}")
    print("=" * 80)
    
    # 1. Load and train models
    print("\n1. Load and train models...")
    models = load_and_train_models()
    
    # 2. Read预测数据
    print(f"\n2. Read预测数据: {PREDICTION_CSV_FILE}")
    
    if not os.path.exists(PREDICTION_CSV_FILE):
        print(f"错误: File {PREDICTION_CSV_FILE} 不存在")
        print("请确保File存在，然后重新运行程序")
        return
    
    try:
        df_pred = pd.read_csv(PREDICTION_CSV_FILE)
        print(f"预测数据形状: {df_pred.shape}")
        print(f"列名: {df_pred.columns.tolist()}")
    except Exception as e:
        print(f"ReadCSVFile失败: {e}")
        return
    
    # Check必要的列是否存在
    required_cols = ['EP_SMILES', 'FR_SMILES', 'CURING_SMILES']
    missing_cols = [col for col in required_cols if col not in df_pred.columns]
    if missing_cols:
        print(f"错误: CSVFile中缺少必要的列: {missing_cols}")
        return
    
    # 3. 获取All需要的描述符（排除Calculate特征）
    all_descriptors = get_required_descriptors(MODEL_PATHS)
    
    # 4. 提取唯一的SMILES，避免重复Calculate
    print("\n3. 提取唯一的SMILES...")
    unique_ep_smiles = df_pred['EP_SMILES'].dropna().unique().tolist()
    unique_fr_smiles = df_pred['FR_SMILES'].dropna().unique().tolist()
    unique_curing_smiles = df_pred['CURING_SMILES'].dropna().unique().tolist()
    
    print(f"  唯一环氧树脂SMILES: {len(unique_ep_smiles)}")
    print(f"  唯一阻燃剂SMILES: {len(unique_fr_smiles)}")
    print(f"  唯一固化剂SMILES: {len(unique_curing_smiles)}")
    
    # 5. Calculate描述符
    print("\n4. Calculate描述符...")
    
    # Calculate EP descriptors
    print("  Calculate EP descriptors...")
    ep_descriptors_dict = {}
    if unique_ep_smiles:
        ep_descriptors = calculate_specific_descriptors(unique_ep_smiles, 'EP', all_descriptors)
        if not ep_descriptors.empty:
            for i, smiles in enumerate(unique_ep_smiles):
                if i < len(ep_descriptors):
                    ep_descriptors_dict[smiles] = ep_descriptors.iloc[[i]]
        else:
            print("  警告: 环氧树脂描述符Calculate失败")
    
    # Calculate FR descriptors
    print("  Calculate FR descriptors...")
    fr_descriptors_dict = {}
    if unique_fr_smiles:
        # 分批Process，避免内存问题
        batch_size = 50
        for i in range(0, len(unique_fr_smiles), batch_size):
            batch_smiles = unique_fr_smiles[i:i+batch_size]
            fr_descriptors_batch = calculate_specific_descriptors(batch_smiles, 'FR', all_descriptors)
            if not fr_descriptors_batch.empty:
                for j, smiles in enumerate(batch_smiles):
                    if j < len(fr_descriptors_batch):
                        fr_descriptors_dict[smiles] = fr_descriptors_batch.iloc[[j]]
            time.sleep(1)  # 避免alvaDesc过载
    
    # Calculate curing agent descriptors
    print("  Calculate curing agent descriptors...")
    curing_descriptors_dict = {}
    if unique_curing_smiles:
        curing_descriptors = calculate_specific_descriptors(unique_curing_smiles, 'CURING', all_descriptors)
        if not curing_descriptors.empty:
            for i, smiles in enumerate(unique_curing_smiles):
                if i < len(curing_descriptors):
                    curing_descriptors_dict[smiles] = curing_descriptors.iloc[[i]]
        else:
            print("  警告: 固化剂描述符Calculate失败")
    
    # 6. 进行预测
    print("\n5. 进行预测...")
    all_results = []
    
    for idx, row in tqdm(df_pred.iterrows(), total=len(df_pred), desc="预测进度"):
        try:
            # 获取SMILES
            ep_smiles = row['EP_SMILES']
            fr_smiles = row['FR_SMILES']
            curing_smiles = row['CURING_SMILES']
            
            # 获取对应的描述符
            ep_desc = ep_descriptors_dict.get(ep_smiles, pd.DataFrame())
            fr_desc = fr_descriptors_dict.get(fr_smiles, pd.DataFrame())
            curing_desc = curing_descriptors_dict.get(curing_smiles, pd.DataFrame())
            
            # Predict properties
            pred_results = predict_for_row(row, ep_desc, fr_desc, curing_desc, models)
            
            # Build result row
            result_row = {
                'Row_Index': idx,
                'EP_SMILES': ep_smiles,
                'FR_SMILES': fr_smiles,
                'CURING_SMILES': curing_smiles,
                'EEW': row.get('EEW', ''),
                'Flame_retardant_AdditionAmount(wt%)': row.get('Flame_retardant_AdditionAmount(wt%)', ''),
                'Curing_agent_AdditionAmount(wt%)': row.get('Curing_agent_AdditionAmount(wt%)', ''),
                'LOI_pred': pred_results.get('LOI_pred', None),
                'Tg_pred': pred_results.get('Tg_pred', None),
                'Tensile_pred': pred_results.get('Tensile_pred', None),
                'UL94_pred': pred_results.get('UL94_pred', None),
                'UL94_prob': pred_results.get('UL94_prob', None)
            }
            
            # 添加工艺Parameters
            for col in df_pred.columns:
                if col not in result_row and col not in ['EP_SMILES', 'FR_SMILES', 'CURING_SMILES']:
                    result_row[col] = row[col]
            
            all_results.append(result_row)
            
        except Exception as e:
            print(f"第{idx}行Prediction failed: {e}")
            continue
    
    # 7. SaveResults
    print("\n6. SaveResults...")
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # 使用CSVFile名作为OutputFile名
        csv_name = os.path.splitext(os.path.basename(PREDICTION_CSV_FILE))[0]
        output_file = os.path.join(OUTPUT_DIR, f'{csv_name}_预测Results.csv')
        results_df.to_csv(output_file, index=False)
        print(f"预测Results已Save到: {output_file}")
        
        # 打印Statistics
        print(f"\n预测DoneStatistics:")
        print(f"  总行数: {len(df_pred)}")
        print(f"  成功预测: {len(all_results)}")
        print(f"  失败: {len(df_pred) - len(all_results)}")
        
        # 如果有预测Results，显示一些Statistics
        if len(all_results) > 0:
            print(f"\n预测ResultsStatistics:")
            if 'LOI_pred' in results_df.columns:
                print(f"  LOI预测范围: {results_df['LOI_pred'].min():.2f} - {results_df['LOI_pred'].max():.2f}")
            if 'Tg_pred' in results_df.columns:
                print(f"  Tg预测范围: {results_df['Tg_pred'].min():.2f} - {results_df['Tg_pred'].max():.2f}")
            if 'Tensile_pred' in results_df.columns:
                print(f"  拉伸强度预测范围: {results_df['Tensile_pred'].min():.2f} - {results_df['Tensile_pred'].max():.2f}")
            if 'UL94_pred' in results_df.columns:
                ul94_v0_count = results_df['UL94_pred'].sum() if 'UL94_pred' in results_df.columns else 0
                print(f"  UL94 V-0预测数量: {ul94_v0_count} ({ul94_v0_count/len(results_df)*100:.1f}%)")
    else:
        print("没有成功预测的Results")
    
    print("\n" + "=" * 80)
    print("预测Done！")
    print("=" * 80)

# ==========================================
# Execute Main Program
# ==========================================
if __name__ == "__main__":
    main()